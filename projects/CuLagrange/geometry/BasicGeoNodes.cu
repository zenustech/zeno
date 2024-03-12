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

        // zs::Vector<int> nmIncidentTris{verts.get_allocator(),verts.size()};
        zs::Vector<T> nodal_area{verts.get_allocator(),verts.size()};

        // cudaPol(zs::range(nmIncidentTris),[] ZS_LAMBDA(auto& count) mutable {count = 0;});
        cudaPol(zs::range(nodal_area),[] ZS_LAMBDA(auto& A) mutable {A = 0;});

        cudaPol(zs::range(tris.size()),[
            exec_tag,
            attrName = zs::SmallString(attrName),
            tris = proxy<cuda_space>({},tris),
            nodal_area = proxy<cuda_space>(nodal_area),
            // nmIncidentTris = proxy<cuda_space>(nmIncidentTris),
            verts = proxy<cuda_space>({},verts)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                zs::vec<T,3> tV[3] = {};
                for(int i = 0;i != 3;++i)
                    tV[i] = verts.pack(dim_c<3>,"x",tri[i]);
                auto A = LSL_GEO::area(tV[0],tV[1],tV[2]);
                tris(attrName,ti) = A;
                for(int i = 0;i != 3;++i) {
                    atomic_add(exec_tag,&nodal_area[tri[i]],A / (T)3.0);
                    // atomic_add(exec_tag,&nmIncidentTris[0],(int)1);
                }
        }); 

        cudaPol(zs::range(verts.size()),[
            verts = proxy<cuda_space>({},verts),
            attrName = zs::SmallString(attrName),
            nodal_area = proxy<cuda_space>(nodal_area)] ZS_LAMBDA(int vi) mutable {
                // if(nmIncidentTris[vi] > 0)
                verts(attrName,vi) = nodal_area[vi];
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


struct ZSCalcSurfaceNormal : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec2 = zs::vec<T,2>;
    using vec3 = zs::vec<T, 3>;
    using vec3i = zs::vec<Ti,3>;
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

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto xtag = get_input2<std::string>("xtag");

        auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();


        if(!tris.hasProperty("nrm"))
            tris.append_channels(cudaPol,{{"nrm",3}});
        cudaPol(zs::range(tris.size()),
            [tris = proxy<space>({},tris),
            xtagOffset = verts.getPropertyOffset(xtag),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) {
            auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
            auto v0 = verts.template pack<3>(xtagOffset,tri[0]);
            auto v1 = verts.template pack<3>(xtagOffset,tri[1]);
            auto v2 = verts.template pack<3>(xtagOffset,tri[2]);

            auto e01 = v1 - v0;
            auto e02 = v2 - v0;

            auto nrm = e01.cross(e02);
            auto nrm_norm = nrm.norm();
            if(nrm_norm < 1e-8)
                nrm = zs::vec<T,3>::zeros();
            else
                nrm = nrm / nrm_norm;

            tris.tuple(dim_c<3>,"nrm",ti) = nrm;
        });
        if(!verts.hasProperty("nrm"))
            verts.append_channels(cudaPol,{{"nrm",3}});
        TILEVEC_OPS::fill(cudaPol,verts,"nrm",(T)0.0);     
        cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            bool is_active_tri = true;

            if(is_active_tri) {
                auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                for(int i = 0;i != 3;++i){
                    for(int d = 0;d != 3;++d)
                        atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
                }
            }
        });           
        cudaPol(zs::range(verts.size()),[verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
            auto nrm = verts.pack(dim_c<3>,"nrm",vi);
            nrm = nrm / (nrm.norm() + (T)1e-6);
            verts.tuple(dim_c<3>,"nrm",vi) = nrm;
        });   

        int nm_iterations = get_param<int>("nm_smooth_iters");
        for(int i = 0;i != nm_iterations;++i) {
            TILEVEC_OPS::fill(cudaPol,tris,"nrm",(T)0.0);  
            cudaPol(zs::range(tris.size()),
                        [tris = proxy<space>({},tris),
                            verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) {
                        auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                        auto nrm = vec3::zeros();
                        for(int i = 0;i != 3;++i)
                            nrm += verts.pack(dim_c<3>,"nrm",tri[i]);
                        nrm = nrm / (nrm.norm() + (T)1e-6);
                        tris.tuple(dim_c<3>,"nrm",ti) = nrm;
                    });
            TILEVEC_OPS::fill(cudaPol,verts,"nrm",(T)0.0);     
            cudaPol(zs::range(tris.size()),[
                    tris = proxy<space>({},tris),
                    verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                for(int i = 0;i != 3;++i)
                    for(int d = 0;d != 3;++d)
                        atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
            });           
            cudaPol(zs::range(verts.size()),[verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                auto nrm = verts.pack(dim_c<3>,"nrm",vi);
                nrm = nrm / (nrm.norm() + (T)1e-6);
                verts.tuple(dim_c<3>,"nrm",vi) = nrm;
            });               
        }


        set_output("zsparticles",zsparticles);
    }
        
};


ZENDEFNODE(ZSCalcSurfaceNormal, {{"zsparticles",{"string","xtag","x"}},
                                    {"zsparticles"},
                                    {
                                    {"int","nm_smooth_iters","0"}
                                    },
                                    {"ZSGeometry"}});
};