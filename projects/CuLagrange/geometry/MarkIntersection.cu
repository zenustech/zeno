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
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/topology.hpp"
#include "kernel/intersection.hpp"
#include "kernel/tiled_vector_ops.hpp"

namespace zeno {

struct ZSMarkSelfCollisionRegion : zeno::INode {
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
        auto &tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();   
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
            throw std::runtime_error("the input zsparticles has no surface lines");   

        auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];

        auto mark_edge = get_param<bool>("mark_edge");
        auto mark_tri = get_param<bool>("mark_tri");

        mark_edge_tri_intersection(cudaPol,verts,tris,lines,"x","inst",mark_edge,mark_tri);

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(ZSMarkSelfCollisionRegion, {{{"zsparticles"}},
							{{"zsparticles"}},
							{
                                {"bool","mark_edge","1"},
                                {"bool","mark_tri","1"}
                            },
							{"ZSGeometry"}});


struct ZSTriMeshSelfCollisionRegion : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto& verts = zsparticles->getParticles();
        auto &tris = zsparticles->getQuadraturePoints(); 

        if(tris.getPropertySize("inds") != 3)
            throw std::runtime_error("only trimesh is currently supported");

        zs::Vector<int> nodal_colors{verts.get_allocator(),verts.size()};
        zs::Vector<zs::vec<int,2>> instBuffer{tris.get_allocator(),tris.size() * 8};

        auto nm_insts = do_global_self_intersection_analysis_on_surface_mesh(cudaPol,
            verts,"x",tris,instBuffer,nodal_colors);

        std::cout << "nm_intersections : " << nm_insts << std::endl;

        auto paramName = get_param<std::string>("paramName");
        auto paramVal = get_input2<float>("paramVal");
        
        if(!verts.hasProperty(paramName))
            verts.append_channels(cudaPol,{{paramName,1}});
        TILEVEC_OPS::fill(cudaPol,verts,paramName,(T)0);
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            paramName = zs::SmallString(paramName),
            paramVal,
            nodal_colors = proxy<space>(nodal_colors)] ZS_LAMBDA(int vi) mutable {
                if(nodal_colors[vi] == 1)
                    verts(paramName,vi) = (T)paramVal;
        });

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(ZSTriMeshSelfCollisionRegion, {{{"zsparticles"},{"float","paramVal","1.0"}},
							{{"zsparticles"}},
							{
                                {"string","paramName","paramName"}
                            },
							{"ZSGeometry"}});



struct ZSMarkCollisionRegion : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();      

        auto zsparticles0 = get_input<ZenoParticles>("ZSParticles0");
        auto zsparticles1 = get_input<ZenoParticles>("ZSParticles1");

        auto& verts0 = zsparticles0->getParticles();
        auto& verts1 = zsparticles1->getParticles();

        auto& edges0 = (*zsparticles0)[ZenoParticles::s_surfEdgeTag];
        auto& edges1 = (*zsparticles1)[ZenoParticles::s_surfEdgeTag];

        auto &tris0 = zsparticles0->category == ZenoParticles::category_e::tet ? (*zsparticles0)[ZenoParticles::s_surfTriTag] : zsparticles0->getQuadraturePoints();   
        auto &tris1 = zsparticles1->category == ZenoParticles::category_e::tet ? (*zsparticles1)[ZenoParticles::s_surfTriTag] : zsparticles1->getQuadraturePoints();   

        dtiles_t verts(verts0.get_allocator(),
            {
                {"x",3}
            },verts0.size() + verts1.size());

        dtiles_t edges(edges0.get_allocator(),
            {
                {"inds",2}
            },edges0.size() + edges1.size());
        
        dtiles_t tris(tris0.get_allocator(),
            {
                {"inds",3}
            },tris0.size() + tris1.size());


        TILEVEC_OPS::copy(cudaPol,verts0,"x",verts,"x",0);
        TILEVEC_OPS::copy(cudaPol,verts1,"x",verts,"x",verts0.size());

        int offset = verts0.size();
        cudaPol(zs::range(edges.size()),[
            offset = verts0.size(),
            edges0 = proxy<space>({},edges0),
            edges1 = proxy<space>({},edges1),
            edges = proxy<space>({},edges),
            size0 = edges0.size(),
            size1 = edges1.size()] ZS_LAMBDA(int ei) mutable {
                if(ei < size0) {
                    edges.tuple(dim_c<2>,"inds",ei) = edges0.pack(dim_c<2>,"inds",ei);
                }else{
                    auto edge = edges1.pack(dim_c<2>,"inds",ei - size0).reinterpret_bits(int_c) + (int)offset;
                    edges.tuple(dim_c<2>,"inds",ei) = edge.template reinterpret_bits<T>();
                }
        });

        cudaPol(zs::range(tris.size()),[
            offset = verts0.size(),
            tris0 = proxy<space>({},tris0),
            tris1 = proxy<space>({},tris1),
            tris = proxy<space>({},tris),
            size0 = tris0.size(),
            size1 = tris1.size()] ZS_LAMBDA(int ti) mutable {
                if(ti < size0) {
                    tris.tuple(dim_c<3>,"inds",ti) = tris0.pack(dim_c<3>,"inds",ti);
                }else{
                    auto tri = tris1.pack(dim_c<3>,"inds",ti - size0).reinterpret_bits(int_c) + (int)offset;
                    tris.tuple(dim_c<3>,"inds",ti) = tri.template reinterpret_bits<T>();
                }
        });

        auto mark_vertex = get_param<bool>("mark_vertex");
        auto mark_edge = get_param<bool>("mark_edge");
        auto mark_tri = get_param<bool>("mark_tri");

        mark_edge_tri_intersection(cudaPol,verts,tris,edges,"x","inst",mark_edge,mark_tri);

        if(mark_vertex) {
            if(!verts0.hasProperty("inst"))
                verts0.append_channels(cudaPol,{{"inst",1}});
            if(!verts1.hasProperty("inst"))
                verts1.append_channels(cudaPol,{{"inst",1}});
            TILEVEC_OPS::fill(cudaPol,verts0,"inst",reinterpret_bits<T>((int)0));
            TILEVEC_OPS::fill(cudaPol,verts1,"inst",reinterpret_bits<T>((int)0));
            cudaPol(zs::range(edges.size()),[
                execTag = wrapv<space>{},
                edges = proxy<space>({},edges),
                verts0 = proxy<space>({},verts0),
                verts1 = proxy<space>({},verts1),
                size0 = edges0.size(),
                edges0 = proxy<space>({},edges0),
                edges1 = proxy<space>({},edges1)] ZS_LAMBDA(int ei) mutable {
                    auto inst = edges("inst",ei);
                    if(inst < (T)0.001)
                        return;
                    if(ei < size0){
                        auto edge = edges0.pack(dim_c<2>,"inds",ei).reinterpret_bits(int_c);
                        verts0("inst",edge[0]) = (T)1;
                        verts0("inst",edge[1]) = (T)1;
                    }else{
                        auto edge = edges1.pack(dim_c<2>,"inds",ei - size0).reinterpret_bits(int_c);
                        verts1("inst",edge[0]) = (T)1;
                        verts1("inst",edge[1]) = (T)1;
                    }
            });
        }

        if(mark_edge) {
            if(!edges0.hasProperty("inst"))
                edges0.append_channels(cudaPol,{{"inst",1}});
            if(!edges1.hasProperty("inst"))
                edges1.append_channels(cudaPol,{{"inst",1}});
            cudaPol(zs::range(edges.size()),[
                edges0 = proxy<space>({},edges0),
                edges1 = proxy<space>({},edges1),
                edges = proxy<space>({},edges),
                size0 = edges0.size(),
                size1 = edges1.size()] ZS_LAMBDA(int ei) mutable {
                    if(ei < size0) {
                        edges0("inst",ei) = edges("inst",ei);
                    }else{
                        edges1("inst",ei - size0) = edges("inst",ei);
                    }
            });           
        }

        if(mark_tri) {
            if(!tris0.hasProperty("inst"))
                tris0.append_channels(cudaPol,{{"inst",1}});
            if(!tris1.hasProperty("inst"))
                tris1.append_channels(cudaPol,{{"inst",1}});
            cudaPol(zs::range(tris.size()),[
                tris0 = proxy<space>({},tris0),
                tris1 = proxy<space>({},tris1),
                tris = proxy<space>({},tris),
                size0 = tris0.size(),
                size1 = tris1.size()] ZS_LAMBDA(int ti) mutable {
                    if(ti < size0) {
                        tris0("inst",ti) = tris("inst",ti);
                    }else{
                        tris1("inst",ti - size0) = tris("inst",ti);
                    }
            });           
        }

        set_output("ZSParticles0",zsparticles0);
        set_output("ZSParticles1",zsparticles1);
    }
};

ZENDEFNODE(ZSMarkCollisionRegion, {{"ZSParticles0",
                                "ZSParticles1"
                            },
							{"ZSParticles0","ZSParticles1"},
							{
                                {"bool","mark_vertex","1"},
                                {"bool","mark_edge","1"},
                                {"bool","mark_tri","1"}
                            },
							{"ZSGeometry"}
});





};