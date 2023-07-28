#include "kernel/bary_centric_weights.hpp"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "zensim/container/Bcht.hpp"
#include "kernel/tiled_vector_ops.hpp"
#include "kernel/geo_math.hpp"

#include <iostream>

namespace zeno {

struct ZSComputeSurfaceArea : zeno::INode {
    using T = float;
    virtual void apply() override {
        using namespace zs;
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec(); 
        constexpr auto exec_tag = wrapv<cuda_space>{}; 

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto& verts = zsparticles->getParticles();
        bool is_tet_volume_mesh = zsparticles->category == ZenoParticles::category_e::tet;
        auto &tris = is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints(); 

        auto attrName = get_param<std::string>("attrName");
        if(!verts.hasProperty(attrName)) {
            verts.append_channels(cudaPol,{{attrName,1}});
        }
        TILEVEC_OPS::fill(cudaPol,verts,attrName,(T)0.0);

        if(!tris.hasProperty(attrName)) {
            tris.append_channels(cudaPol,{{attrName,1}});
        }
        TILEVEC_OPS::fill(cudaPol,verts,attrName,(T)0.0);

        zs::Vector<int> nmIncidentTris{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(nmIncidentTris),[] ZS_LAMBDA(int& count) mutable {count = 0;});

        cudaPol(zs::range(tris.size()),[
            exec_tag,
            attrName = zs::SmallString(attrName),
            tris = proxy<cuda_space>({},tris),
            nmIncidentTris = proxy<cuda_space>(nmIncidentTris),
            verts = proxy<cuda_space>({},verts)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                zs::vec<T,3> tV[3] = {};
                for(int i = 0;i != 3;++i)
                    tV[i] = verts.pack(dim_c<3>,"x",tri[i]);
                auto A = LSL_GEO::area(tV[0],tV[1],tV[2]);
                tris(attrName,ti) = A;
                for(int i = 0;i != 3;++i) {
                    atomic_add(exec_tag,&verts(attrName,tri[i]),A);
                    atomic_add(exec_tag,&nmIncidentTris[0],(int)1);
                }
        });

        cudaPol(zs::range(verts.size()),[
            verts = proxy<cuda_space>({},verts),
            attrName = zs::SmallString(attrName),
            nmIncidentTris = proxy<cuda_space>(nmIncidentTris)] ZS_LAMBDA(int vi) mutable {
                if(nmIncidentTris[vi] > 0)
                    verts(attrName,vi) = verts(attrName,vi) / (T)nmIncidentTris[vi];
        });

        set_output("zsparticles",zsparticles);
    }
};


ZENDEFNODE(ZSComputeSurfaceArea, {{{"zsparticles"}},
                            {{"zsparticles"}},
                            {
                                {"string","attrName","area"}
                            },
                            {"ZSGeometry"}});

};