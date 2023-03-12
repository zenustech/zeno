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
#define MAX_FP_COLLISION_PAIRS 4

struct KinematicCollisionRepulsion : zeno::INode {
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

    virtual void apply() override {
        using namespace zs;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zssurf = get_input<ZenoParticles>("zssurf");
        auto kboundary = get_input<ZenoParticles>("kboundary");
        auto& verts = zssurf->getParticles();

        const auto& kverts = kboundary->getParticles();
        const auto& kb_tris = kboundary->getQuadraturePoints();
        auto ktris = typename ZenoParticles::particles_t({
                {"inds",3},{"nrm",3},{"ff_inds",3}},0,zs::memsrc_e::device,kb_tris.size());
        if(kb_tris.getChannelSize("inds") != 3)
            throw std::runtime_error("the input kboundary should be triangulate mesh");
        TILEVEC_OPS::copy<3>(cudaPol,kb_tris,"inds",ktris,"inds");
        TILEVEC_OPS::copy<3>(cudaPol,kb_tris,"ff_inds",ktris,"ff_inds");        
        // calculate ktris normal
        cudaPol(zs::range(ktris.size()),
            [kverts = proxy<space>({},kverts),
                ktris = proxy<space>({},ktris)] ZS_LAMBDA(int ti) mutable {
            auto tri = ktris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            auto v0 = kverts.template pack<3>("x",tri[0]);
            auto v1 = kverts.template pack<3>("x",tri[1]);
            auto v2 = kverts.template pack<3>("x",tri[2]);

            auto e01 = v1 - v0;
            auto e02 = v2 - v0;

            auto nrm = e01.cross(e02);
            auto nrm_norm = nrm.norm();
            if(nrm_norm < 1e-8)
                nrm = zs::vec<T,3>::zeros();
            else
                nrm = nrm / nrm_norm;

            ktris.tuple(dim_c<3>,"nrm",ti) = nrm;
        });
           
        auto dvTag = get_param<std::string>("dvTag");
        if(verts.hasProperty(dvTag) && verts.getChannelSize(dvTag) != 3){
            fmt::print(fg(fmt::color::red),"wrong dvTag {} channel size = {}\n",dvTag,verts.getChannelSize(dvTag));
            throw std::runtime_error("wrong dvTag channel size");
        }
        if(!verts.hasProperty(dvTag)){
            verts.append_channels(cudaPol,{{dvTag,3}});
        }
        auto dpTag = get_param<std::string>("dpTag");
        if(verts.hasProperty(dpTag) && verts.getChannelSize(dpTag) != 3){
            fmt::print(fg(fmt::color::red),"wrong dpTag {} channel size = {}\n",dpTag,verts.getChannelSize(dpTag));
            throw std::runtime_error("wrong dpTag channel size");
        }
        if(!verts.hasProperty(dpTag)){
            verts.append_channels(cudaPol,{{dpTag,3}});
        }

        auto kinInCollisionEps = get_input2<float>("kinInCollisionEps");
        auto kinOutCollisionEps = get_input2<float>("kinOutCollisionEps");

        // do collision detection
        dtiles_t kc_buffer{verts.get_allocator(),
            {
                {"inds",2}
            },verts.size() * MAX_FP_COLLISION_PAIRS};   
        TILEVEC_OPS::fill<2>(cudaPol,kc_buffer,"inds",zs::vec<int,2>::uniform(-1).template reinterpret_bits<T>());     
        
        auto stBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,kverts,ktris,wrapv<3>{},(T)0.0,"x");
        stBvh.build(cudaPol,bvs);
        auto thickness = kinInCollisionEps + kinOutCollisionEps;

        cudaPol(zs::range(verts.size()),
            [verts = proxy<space>({},verts),
                thickness = thickness,
                kc_buffer = proxy<space>({},kc_buffer),
                stBvh = proxy<space>(stBvh),
                ktris = proxy<space>({},ktris),
                kverts = proxy<space>({},kverts),
                in_collisionEps = kinInCollisionEps,
                out_collisionEps = kinOutCollisionEps] ZS_LAMBDA(int vi) mutable {
            auto p = verts.pack(dim_c<3>,"x",vi);
            auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};

            int nm_collision_pairs = 0;
            auto process_kinematic_vertex_face_collision_pairs = [&](int kti) {
                if(nm_collision_pairs >= MAX_FP_COLLISION_PAIRS)
                    return;
                auto ktri = ktris.pack(dim_c<3>,"inds",kti).reinterpret_bits(int_c);
                T dist = (T)0.0;

                auto nrm = ktris.pack(dim_c<3>,"nrm",kti);
                auto seg = p - kverts.pack(dim_c<3>,"x",ktri[0]);

                auto kt0 = kverts.pack(dim_c<3>,"x",ktri[0]);
                auto kt1 = kverts.pack(dim_c<3>,"x",ktri[1]);
                auto kt2 = kverts.pack(dim_c<3>,"x",ktri[2]);   

                T barySum = (T)0.0;
                T distance = LSL_GEO::pointTriangleDistance(kt0,kt1,kt2,p,barySum);   

                auto collisionEps = dist < 0 ? out_collisionEps : in_collisionEps;
                // if(barySum > 1.1)
                //     return;
                if(distance > collisionEps)
                    return;

                // if(!LSL_GEO::pointProjectsInsideTriangle(kt0,kt1,kt2,p)){
                auto ntris = ktris.pack(dim_c<3>,"ff_inds",kti).reinterpret_bits(int_c);
                for(int i = 0;i != 3;++i) {
                    auto nti = ntris[i];
                    auto edge_normal = ktris.pack(dim_c<3>,"nrm",kti) + ktris.pack(dim_c<3>,"nrm",nti);
                    edge_normal = (edge_normal)/(edge_normal.norm() + (T)1e-6);
                    auto e0 = kverts.pack(dim_c<3>,"x",ktri[(i+0)%3]);
                    auto e1 = kverts.pack(dim_c<3>,"x",ktri[(i+1)%3]);
                    auto e10 = e1 - e0;
                    auto bisector_normal = edge_normal.cross(e10).normalized();

                    seg = p - kverts.pack(dim_c<3>,"x",ktri[i]);
                    if(bisector_normal.dot(seg) < 0)
                        return;
                }
                // }
                kc_buffer.tuple(dim_c<2>,"inds",vi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = 
                    zs::vec<int,2>(vi,kti).reinterpret_bits<T>();
                
                nm_collision_pairs++;
            };
            stBvh.iter_neighbors(bv,process_kinematic_vertex_face_collision_pairs);
        });

        zs::Vector<T> constraint_weight_sum{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(verts.size()),
            [constraint_weight_sum = proxy<space>(constraint_weight_sum)] ZS_LAMBDA(int vi) mutable {
                constraint_weight_sum[vi] = (T)0.0;
        });
        // evaluate collision repulsion
        TILEVEC_OPS::fill(cudaPol,verts,dvTag,(T)0.0);
        TILEVEC_OPS::fill(cudaPol,verts,dpTag,(T)0.0);
        cudaPol(zs::range(kc_buffer.size()),
            [kc_buffer = proxy<space>({},kc_buffer),
                constraint_weight_sum = proxy<space>(constraint_weight_sum),
                verts = proxy<space>({},verts),
                kverts = proxy<space>({},kverts),
                ktris = proxy<space>({},ktris),
                dvTag = zs::SmallString(dvTag),
                dpTag = zs::SmallString(dpTag)] ZS_LAMBDA(int cpi) mutable {
            auto inds = kc_buffer.pack(dim_c<2>,"inds",cpi).reinterpret_bits(int_c);
            for(int i = 0;i != 2;++i)
                if(inds[i] < 0)
                    return;
            vec3 cp[4] = {}; 
            cp[0] = verts.pack(dim_c<3>,"x",inds[0]);
            auto ktri = ktris.pack(dim_c<3>,"inds",inds[1]).reinterpret_bits(int_c);
            cp[1] = kverts.pack(dim_c<3>,"x",ktri[0]);
            cp[2] = kverts.pack(dim_c<3>,"x",ktri[1]);
            cp[3] = kverts.pack(dim_c<3>,"x",ktri[2]);   
    
            auto bary = LSL_GEO::getInsideBarycentricCoordinates(cp);
            auto project_p = cp[1] * bary[0] + cp[2] * bary[1] + cp[3] * bary[2];

            auto project_v = vec3::zeros();
            for(int i = 0;i != 3;++i)
                project_v += kverts.pack(dim_c<3>,"v",ktri[2]) * bary[i]; 

            auto dp = project_p - cp[0];
            auto dv = project_v - verts.pack(dim_c<3>,"v",inds[0]);
            // project out the normal part
            auto nrm = ktris.pack(dim_c<3>,"nrm",inds[1]);
            dv -= dv.dot(nrm) * nrm;
            auto w = (T)1.0;
            atomic_add(exec_cuda,&constraint_weight_sum[inds[0]],(T)1.0);
            for(int d = 0;d != 3;++d){
                atomic_add(exec_cuda,&verts(dpTag,d,inds[0]),(T)dp[d] * (T)w);
                atomic_add(exec_cuda,&verts(dvTag,d,inds[0]),(T)dv[d] * (T)w);
            }
        });

        cudaPol(zs::range(verts.size()),
            [verts = proxy<space>({},verts),
                constraint_weight_sum = proxy<space>(constraint_weight_sum),
                dpTag = zs::SmallString(dpTag),
                dvTag = zs::SmallString(dvTag)] ZS_LAMBDA(int vi) mutable {
            verts.tuple(dim_c<3>,dvTag,vi) = verts.pack(dim_c<3>,dvTag,vi) / constraint_weight_sum[vi];
            verts.tuple(dim_c<3>,dpTag,vi) = verts.pack(dim_c<3>,dpTag,vi) / constraint_weight_sum[vi];
        });
    }
};

ZENDEFNODE(KinematicCollisionRepulsion, {
                                  {"zssurf","kboundary",
                                    {"float","kinInCollisionEps","0.1"},
                                    {"float","kinOutCollisionEps","0.01"}
                                  },
                                  {"zssurf"},
                                  {
                                    {"string","dvTag","dv"},
                                    {"string","dpTag","dp"}
                                  },
                                  {"FEM"}});

};