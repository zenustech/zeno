#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "../geometry/linear_system/mfcg.hpp"

#include "../geometry/kernel/calculate_facet_normal.hpp"
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/compute_characteristic_length.hpp"
#include "../geometry/kernel/calculate_bisector_normal.hpp"

#include "../geometry/kernel/tiled_vector_ops.hpp"
#include "../geometry/kernel/geo_math.hpp"

#include "../geometry/kernel/calculate_edge_normal.hpp"

#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"

#include "collision_energy/vertex_face_sqrt_collision.hpp"
#include "collision_energy/vertex_face_collision.hpp"
#include "collision_energy/evaluate_collision.hpp"


namespace zeno {

struct StrainLimitRepulsion : INode {

    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec2 = zs::vec<T,2>;
    using vec3 = zs::vec<T, 3>;
    using mat3 = zs::vec<T, 3, 3>;
    using mat9 = zs::vec<T,9,9>;
    using mat12 = zs::vec<T,12,12>;

    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;

    using pair3_t = zs::vec<Ti,3>;
    using pair4_t = zs::vec<Ti,4>;

    // we assume that all the vertex has the same mass = 1.0kg
    virtual void apply() override {
        using namespace zs;
        auto zssurf = get_input<ZenoParticles>("zssurf");
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto& verts = zssurf->getParticles();
        auto& tris = zssurf->getQuadraturePoints();

        auto restShapeTag = get_param<std::string>("restShapeTag");
        auto defShapeTag = get_param<std::string>("defShapeTag");
        auto repulsionTag = get_param<std::string>("repulsionTag");

        if(!verts.hasProperty(restShapeTag)){
            fmt::print(fg(fmt::color::red),"the input zssurf has no restShapeTag{}\n",restShapeTag);
            throw std::runtime_error("the input zssurf has no restShapeTag");
        }
        if(!verts.hasProperty(defShapeTag)){
            fmt::print(fg(fmt::color::red),"the input zssurf has no defShapeTag{}\n",defShapeTag);
            throw std::runtime_error("the input zssurf has no defShapeTag");
        }
        if(verts.hasProperty(repulsionTag) && verts.getChannelSize(repulsionTag) != 3){
            fmt::print(fg(fmt::color::red),"wrong repulsion {} channel size = {}\n",repulsionTag,verts.getChannelSize(repulsionTag));
            throw std::runtime_error("wrong repulsion channel size");
        }
        if(!verts.hasProperty(repulsionTag)){
            verts.append_channels(cudaPol,{{repulsionTag,3}});
        }

        auto strainStretchLimit = get_input2<float>("strainStretchLimit");
        auto strainShrinkLimit = get_input2<float>("strainShrinkLimit");

        if(strainStretchLimit < 1.0)
            throw std::runtime_error("invalid strainStretchLimit(< 1.0)");
        if(strainShrinkLimit > 1.0)
            throw std::runtime_error("invalid strainShrinkLImit(> 1.0)");
        // strainLimit = strainLimit > 0.9 ? 0.9 : strainLimit;
        // strainLimit = strainLimit < 0.1 ? 0.1 : strainLimit;
        auto limitStrength = get_input2<float>("limitStrength");

        zs::Vector<T> constraint_count{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(verts.size()),
            [constraint_count = proxy<space>(constraint_count)] ZS_LAMBDA(int vi) mutable {
                constraint_count[vi] = (T)0.0;
        });

        zs::Vector<zs::vec<T,3,3>> constraint_repulsion{tris.get_allocator(),tris.size()};

        TILEVEC_OPS::fill(cudaPol,verts,repulsionTag,(T)0.0);
        cudaPol(zs::range(tris.size()),[
                constraint_count = proxy<space>(constraint_count),
                constraint_repulsion = proxy<space>(constraint_repulsion),
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts),
                strainStretchLimit = strainStretchLimit,
                strainShrinkLimit = strainShrinkLimit,
                limitStrength = limitStrength,
                restShapeTag = zs::SmallString(restShapeTag),
                defShapeTag = zs::SmallString(defShapeTag),
                repulsionTag = zs::SmallString(repulsionTag)] ZS_LAMBDA(int ti) mutable {
            auto inds = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            constraint_repulsion[ti] = zs::vec<T,3,3>::zeros();
            for(int i = 0;i != 3;++i) {
                auto idx0 = inds[(i + 0) % 3];
                auto idx1 = inds[(i + 1) % 3];
                auto E = verts.pack(dim_c<3>,restShapeTag,idx1) - verts.pack(dim_c<3>,restShapeTag,idx0);
                auto e = verts.pack(dim_c<3>,defShapeTag,idx1) - verts.pack(dim_c<3>,defShapeTag,idx0);

                auto En = E.norm();
                auto en = e.norm();

                auto strain = en / En;
                auto repulsion = vec3::zeros();
                auto alpha = (T)1.0;
                if(verts("bou_tag",idx0) > 0.5 || verts("bou_tag",idx1) > 0.5)
                    alpha = (T)0.5;

                if(strain > strainStretchLimit)
                    repulsion += En * e * (strain - strainStretchLimit) / alpha / en;
                else if(strain < strainShrinkLimit)
                    repulsion += En * e * (strain - strainShrinkLimit) / alpha / en;
                else
                    return;
                repulsion *= limitStrength;

                atomic_add(exec_cuda,&constraint_count[idx0],(T)1.0);
                atomic_add(exec_cuda,&constraint_count[idx1],(T)1.0);

                for(int d = 0;d != 3;++d)
                    constraint_repulsion[ti](i,d) = repulsion[d];
                // for(int d = 0;d != 3;++d){
                //     atomic_add(exec_cuda,&verts(repulsionTag,d,idx0),repulsion[d]);
                //     atomic_add(exec_cuda,&verts(repulsionTag,d,idx1),-repulsion[d]);
                // }
            }
        });

        cudaPol(zs::range(tris.size()),
            [constraint_count = proxy<space>(constraint_count),
                tris = proxy<space>({},tris),
                repulsionTag = zs::SmallString(repulsionTag),
                verts = proxy<space>({},verts),
                constraint_repulsion = proxy<space>(constraint_repulsion)] ZS_LAMBDA(int ti) mutable {
            auto inds = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            for(int i = 0;i != 3;++i){
                auto idx0 = inds[(i + 0) % 3];
                auto idx1 = inds[(i + 1) % 3];
                T max_c_num = (T)0;
                max_c_num = constraint_count[idx0] > max_c_num ? constraint_count[idx0] : max_c_num;
                max_c_num = constraint_count[idx1] > max_c_num ? constraint_count[idx1] : max_c_num;

                auto repulsion = row(constraint_repulsion[ti],i);
                if(max_c_num > (T)0.5){
                    repulsion /= max_c_num;
                    for(int d = 0;d != 3;++d){
                        atomic_add(exec_cuda,&verts(repulsionTag,d,idx0),repulsion[d]);
                        atomic_add(exec_cuda,&verts(repulsionTag,d,idx1),-repulsion[d]);
                    }
                }
            }
        });

        // cudaPol(zs::range(verts.size()),
        //     [constraint_count = proxy<space>(constraint_count),
        //         repulsionTag = zs::SmallString(repulsionTag),
        //         verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
        //         if(constraint_count[vi] > (T)0.5)
        //             verts.tuple(dim_c<3>,repulsionTag,vi) = verts.pack(dim_c<3>,repulsionTag,vi)/(T)constraint_count[vi];
        // });

        set_output("zssurf",zssurf);
    }

};


ZENDEFNODE(StrainLimitRepulsion, {{"zssurf",
                                    {"float","strainStretchLimit","1.5"},
                                    {"float","strainShrinkLimit","0.5"},
                                    {"float","limitStrength","1.0"}
                                    },
                                  {"zssurf"},
                                  {
                                    {"string","restShapeTag","X"},
                                    {"string","defShapeTag","x"},
                                    {"string","repulsionTag","srep"}
                                  },
                                  {"FEM"}});

};