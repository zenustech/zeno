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

#include "kernel/tiled_vector_ops.hpp"
#include "zensim/container/Bvh.hpp"


namespace zeno {

struct ZSMarkInvertedTet : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;

    virtual void apply() override {
        using namespace zs;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zsparticles = get_input<ZenoParticles>("ZSParticles");
        auto& verts = zsparticles->getParticles();
        auto& eles = zsparticles->getQuadraturePoints();
        auto use_append = get_param<bool>("append");
        if(eles.getPropertySize("inds") != 4)
            throw std::runtime_error("the input zsparticles is not a tetrahedra mesh");

        if(!verts.hasProperty("is_inverted")){
            verts.append_channels(cudaPol,{{"is_inverted",1}});
        }     

        if(!use_append){
            TILEVEC_OPS::fill(cudaPol,verts,"is_inverted",(T)0.0);
        }

        cudaPol(zs::range(eles.size()),
            [verts = proxy<space>({},verts),
                    quads = proxy<space>({},eles)] ZS_LAMBDA(int ei) mutable {
                auto DmInv = quads.template pack<3,3>("IB",ei);
                auto inds = quads.template pack<4>("inds",ei).reinterpret_bits(int_c);
                vec3 x1[4] = {verts.template pack<3>("x", inds[0]),
                        verts.template pack<3>("x", inds[1]),
                        verts.template pack<3>("x", inds[2]),
                        verts.template pack<3>("x", inds[3])};   

                mat3 F{};
                {
                    auto x1x0 = x1[1] - x1[0];
                    auto x2x0 = x1[2] - x1[0];
                    auto x3x0 = x1[3] - x1[0];
                    auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                                    x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                    F = Ds * DmInv;
                } 
                if(zs::determinant(F) < 0.0){
                    for(int i = 0;i < 4;++i)
                        verts("is_inverted",inds[i]) = (T)1.0;   
                    // etemp("is_inverted",ei) = (T)1.0;   
                }           
        });    

        set_output("ZSParticles",zsparticles);    
    }
};

ZENDEFNODE(ZSMarkInvertedTet, {{{"ZSParticles"}},
                            {{"ZSParticles"}},
                            {
                                {"bool", "append", "0"},
                            },
                            {"ZSGeometry"}});


};