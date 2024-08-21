#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/tiled_vector_ops.hpp"
#include "zensim/container/Bcht.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "kernel/halfedge_structure.hpp"
#include "kernel/compute_characteristic_length.hpp"

namespace zeno {

    struct BuildSurfaceHalfEdgeStructure : zeno::INode {
        using T = float;
    
        virtual void apply() override {
            using namespace zs;
            using vec2i = zs::vec<int, 2>;
            using vec3i = zs::vec<int, 3>;
            constexpr auto space = zs::execspace_e::cuda;
    
            auto zsparticles = get_input<ZenoParticles>("zsparticles");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag) && zsparticles->category == ZenoParticles::category_e::tet)
                throw std::runtime_error("the input tet zsparticles has no surface tris");
            // if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
            //     throw std::runtime_error("the input zsparticles has no surface lines");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag) && zsparticles->category == ZenoParticles::category_e::tet)
                throw std::runtime_error("the input tet zsparticles has no surface points");
    
            auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();
            // auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
            auto& points = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfVertTag] : zsparticles->getParticles();
            
            auto& halfEdge = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
            halfEdge = typename ZenoParticles::particles_t({{"local_vertex_id",1},{"to_face",1},{"opposite_he",1},{"next_he",1}},
                tris.size() * 3,zs::memsrc_e::device);
    
            auto cudaPol = zs::cuda_exec();
    
            points.append_channels(cudaPol,{{"he_inds",1}});
            // lines.append_channels(cudaPol,{{"he_inds",1}});
            tris.append_channels(cudaPol,{{"he_inds",1}});
    
    #if 0
    
            // constexpr auto space = zs::execspace_e::cuda;
    
            TILEVEC_OPS::fill(cudaPol,halfEdge,"to_vertex",reinterpret_bits<T>((int)-1));
            TILEVEC_OPS::fill(cudaPol,halfEdge,"to_face",reinterpret_bits<T>((int)-1));
            TILEVEC_OPS::fill(cudaPol,halfEdge,"to_edge",reinterpret_bits<T>((int)-1));
            TILEVEC_OPS::fill(cudaPol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
            TILEVEC_OPS::fill(cudaPol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));
    
            // we might also need a space hash structure here, map from [i1,i2]->[ej]
            bcht<vec2i,int,true,universal_hash<vec2i>,32> de2fi{halfEdge.get_allocator(),halfEdge.size()};
    
            cudaPol(zs::range(tris.size()), [
                tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi),halfEdge = proxy<space>({},halfEdge)] ZS_LAMBDA(int ti) mutable {
                    auto fe_inds = tris.pack(dim_c<3>,"fe_inds",ti).reinterpret_bits(int_c);
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
    
                    vec3i nos{};
                    for(int i = 0;i != 3;++i) {
                        if(auto no = de2fi.insert(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
                            nos[i] = no;
                            halfEdge("to_vertex",no) = reinterpret_bits<T>(tri[i]);
                            halfEdge("to_face",no) = reinterpret_bits<T>(ti);
                            halfEdge("to_edge",no) = reinterpret_bits<T>(fe_inds[i]);
                            // halfEdge("next_he",no) = ti * 3 + (i+1) % 3;
                        } else {
                            // some error happen
    
                        }						
                    }
                    for(int i = 0;i != 3;++i)
                        halfEdge("next_he",nos[i]) = reinterpret_bits<T>(nos[(i+1) % 3]);
            });
            cudaPol(zs::range(halfEdge.size()),
                [halfEdge = proxy<space>({},halfEdge),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int hei) mutable {
                    auto idx0 = reinterpret_bits<int>(halfEdge("to_vertex",hei));
                    auto nexthei = reinterpret_bits<int>(halfEdge("next_he",hei));
                    auto idx1 = reinterpret_bits<int>(halfEdge("to_vertex",nexthei));
                    if(auto no = de2fi.query(vec2i{idx1,idx0});no >= 0)
                        halfEdge("opposite_he",hei) = reinterpret_bits<T>(no);
                    else	
                        halfEdge("opposite_he",hei) = reinterpret_bits<T>((int)-1);
            });
    
            points.append_channels(cudaPol,{{"he_inds",1}});
            lines.append_channels(cudaPol,{{"he_inds",1}});
            tris.append_channels(cudaPol,{{"he_inds",1}});
    
            cudaPol(zs::range(lines.size()),[
                lines = proxy<space>({},lines),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int li) mutable {
                    auto linds = lines.pack(dim_c<2>,"inds",li).reinterpret_bits(int_c);
                    if(auto no = de2fi.query(vec2i{linds[0],linds[1]});no >= 0){
                        lines("he_inds",li) = reinterpret_bits<T>((int)no);
                    }else {
                        // some algorithm bug
                    }
            });
    
            if(!tris.hasProperty("fp_inds") || tris.getPropertySize("fp_inds") != 3) {
                throw std::runtime_error("the tris has no fp_inds channel");
            }
    
            cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int ti) mutable {
    
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    if(auto no = de2fi.query(vec2i{tri[0],tri[1]});no >= 0){
                        tris("he_inds",ti) = reinterpret_bits<T>((int)no);
                    }else {
                        // some algorithm bug
                        printf("could not find half edge : %d %d\n",tri[0],tri[1]);
                    }
                    // auto tinds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);
                    // for(int i = 0;i != 3;++i) {
                    // 	if(auto no = de2fi.query(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
                    // 		points("he_inds",tinds[i]) = reinterpret_bits<T>((int)no);
                    // 	}else {
                    // 		// some algorithm bug
                    // 		printf("could not find half edge : %d %d\n",tri[i],tri[(i+1) % 3]);
                    // 	}						
                    // }
            });
    
            cudaPol(zs::range(tris.size()),[
                points = proxy<space>({},points),tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int ti) mutable {
                    auto tinds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    for(int i = 0;i != 3;++i) {
                        if(auto no = de2fi.query(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
                            points("he_inds",tinds[i]) = reinterpret_bits<T>((int)no);
                        }else {
                            // some algorithm bug
                            printf("could not find half edge : %d %d\n",tri[i],tri[(i+1) % 3]);
                        }						
                    }
            });
    #else
            auto accept_non_manifold = get_input2<bool>("accept_non_manifold");
            if(!accept_non_manifold) {
                if(!build_half_edge_structure_for_triangle_mesh(cudaPol,tris,points,halfEdge))
                    throw std::runtime_error("fail building surf half edge");
            }else {
                if(!build_half_edge_structure_for_triangle_mesh_robust(cudaPol,tris,halfEdge))
                    throw std::runtime_error("fail building surf half edge");
            }
    #endif
    
            zs::bht<int,1,int> edgeSet{tris.get_allocator(),tris.size() * 3};	
            zs::bht<int,1,int> boundaryHalfEdgeSet{tris.get_allocator(),tris.size() * 3};
            edgeSet.reset(cudaPol,true);
            boundaryHalfEdgeSet.reset(cudaPol,true);
            cudaPol(zs::range(halfEdge.size()),[
                halfedges = proxy<space>({},halfEdge),
                boundaryHalfEdgeSet = proxy<space>(boundaryHalfEdgeSet),
                edgeSet = proxy<space>(edgeSet),
                tris = proxy<space>({},tris)] ZS_LAMBDA(int hi) mutable {
                    auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    if(ti < 0) {
                        printf("oops!!! halfedge with no incident triangle!!!\n");
                        return;
                    }
                        
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    auto local_idx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                    zs::vec<int,2> edge{tri[local_idx],tri[(local_idx + 1) % 3]};
    
                    auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
                    if(ohi < 0)
                        boundaryHalfEdgeSet.insert(hi);
                    if(ohi >= 0 && edge[0] > edge[1])
                        return;
    
                    edgeSet.insert(hi);
            });
    
            auto &surfEdges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
            surfEdges = typename ZenoParticles::particles_t({{"inds", 2},{"he_inds",1}}, edgeSet.size(),zs::memsrc_e::device);
            cudaPol(zip(zs::range(edgeSet.size()),edgeSet._activeKeys),[
                halfedges = proxy<space>({},halfEdge),
                surfEdges = proxy<space>({},surfEdges),
                tris = proxy<space>({},tris)] ZS_LAMBDA(auto ei,const auto& hi_vec) mutable {
                    auto hi = hi_vec[0];
                    auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                    if(ti < 0) {
                        printf("oops!!! halfedge with no incident triangle!!!\n");
                        return;
                    }
    
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    auto local_idx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                    zs::vec<int,2> edge{tri[local_idx],tri[(local_idx + 1) % 3]};	
                    
                    surfEdges.tuple(dim_c<2>,"inds",ei) = edge.reinterpret_bits(float_c);
                    surfEdges("he_inds",ei) = reinterpret_bits<float>((int)hi);
            });
    
            auto& boundaryHalfEdges = (*zsparticles)[ZenoParticles::s_surfBoundaryEdgeTag];
            boundaryHalfEdges = typename ZenoParticles::particles_t({{"he_inds",1}},
                boundaryHalfEdgeSet.size(),zs::memsrc_e::device);
    
            cudaPol(zip(zs::range(boundaryHalfEdgeSet.size()),boundaryHalfEdgeSet._activeKeys),[
                boundaryHalfEdges = boundaryHalfEdges.begin("he_inds",dim_c<1>,int_c)] ZS_LAMBDA(int id,const auto& key) mutable {
                    boundaryHalfEdges[id] = key[0];
            });
    
            set_output("zsparticles",zsparticles);
        }
    
    };
    
    
    ZENDEFNODE(BuildSurfaceHalfEdgeStructure, {{
                                    {"zsparticles"},
                                    {gParamType_Bool,"accept_non_manifold","0"},
                                },
                                {{"zsparticles"}},
                                {},
                                {"ZSGeometry"}});
    
    struct BuildTetrahedraHalfFacet : zeno::INode {
        using T = float;
        virtual void apply() override {
            using namespace zs;
            auto cudaPol = zs::cuda_exec();
    
    
            auto zsparticles = get_input<ZenoParticles>("zsparticles");
    
            auto& tets = zsparticles->getQuadraturePoints();
            tets.append_channels(cudaPol,{{"hf_inds",1}});
    
            auto& halfFacet = (*zsparticles)[ZenoParticles::s_tetHalfFacetTag];
            halfFacet = typename ZenoParticles::particles_t({{"opposite_hf",1},{"next_hf",1},{"to_tet",1},{"local_idx",1}},
                    tets.size() * 4,zs::memsrc_e::device);
    
            build_tetrahedra_half_facet(cudaPol,tets,halfFacet);
    
            set_output("zsparticles",zsparticles);
        }
    };
    
    ZENDEFNODE(BuildTetrahedraHalfFacet, {{{"zsparticles"}},
                                {{"zsparticles"}},
                                {},
                                {"ZSGeometry"}});
    
    struct BuildSurfaceLinesStructure : zeno::INode {
        using T = float;
        virtual void apply() override {
            using namespace zs;
            using vec2i = zs::vec<int, 2>;
            using vec3i = zs::vec<int, 3>;
            auto cudaPol = zs::cuda_exec();
            constexpr auto space = zs::execspace_e::cuda;
    
            auto zsparticles = get_input<ZenoParticles>("zsparticles");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag) && zsparticles->category == ZenoParticles::category_e::tet)
                throw std::runtime_error("the input tet zsparticles has no surface tris");
    
            auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();
            zs::bht<int,2,int> edgeSet{tris.get_allocator(),tris.size() * 3};
            edgeSet.reset(cudaPol,true);
    
            cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                edgeSet = proxy<space>(edgeSet)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    for(int i = 0;i != 3;++i) {
                        auto idx0 = tri[(i + 0) % 3];
                        auto idx1 = tri[(i + 1) % 3];
                        if(idx0 < idx1)
                            edgeSet.insert(vec2i{idx0,idx1});
                    }
            });
            
            auto &surfEdges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
            surfEdges = typename ZenoParticles::particles_t({{"inds", 2}}, edgeSet.size(),zs::memsrc_e::device);	
            cudaPol(zip(zs::range(edgeSet.size()),edgeSet._activeKeys),[
                surfEdges = proxy<space>({},surfEdges)] ZS_LAMBDA(auto ei,const auto& pair) mutable {
                    surfEdges.tuple(dim_c<2>,"inds",ei) = pair.reinterpret_bits(float_c);
            });	
    
            set_output("zsparticles",zsparticles);
        }
    };
    
    ZENDEFNODE(BuildSurfaceLinesStructure, {{{"zsparticles"}},
                                {{"zsparticles"}},
                                {},
                                {"ZSGeometry"}});
    

    struct BuildSurfFacetTetraNeighboring : zeno::INode {
        virtual void apply() override {
            using namespace zs;
            using vec2i = zs::vec<int, 2>;
            using vec3i = zs::vec<int, 3>;
            using vec3 = zs::vec<float,3>;

            auto cudaPol = zs::cuda_exec();
            constexpr auto space = zs::execspace_e::cuda;

            auto zsparticles = get_input<ZenoParticles>("zsparticles");
            const auto& verts = zsparticles->getParticles();
            auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
            const auto& tets = zsparticles->getQuadraturePoints();

#if 0

            zs::bht<int,3,int> tris_htab{tris.get_allocator(),tris.size()};
            tris_htab.reset(cudaPol,true);
            zs::Vector<int> tris_id{tris.get_allocator(),tris.size()};
            cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                tris_id = proxy<space>(tris_id),
                tris_htab = proxy<space>(tris_htab)] ZS_LAMBDA(int ti) mutable {
                    auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
                    order_indices(tri);
                    for(int i = 0;i != 2;++i)
                        if(tri[i] >= tri[i + 1])
                            printf("invalid ordered tri : %d %d %d\n",tri[0],tri[1],tri[2]);
                    auto no = tris_htab.insert(tri);
                        tris_id[no] = ti;
            });

            cudaPol(zs::range(tris_id),[] ZS_LAMBDA(int& ti) {
                if(ti < 0)
                    printf("invalid ordered tri[%d]\n");
            });

            if(!tris.hasProperty("ft_inds")) {
                tris.append_channels(cudaPol,{{"ft_inds",1}});
            }
            TILEVEC_OPS::fill(cudaPol,tris,"ft_inds",zs::reinterpret_bits<float>((int)-1));

            cudaPol(zs::range(tets.size()),[
                tets = proxy<space>({},tets),
                tris = proxy<space>({},tris),
                tris_htab = proxy<space>(tris_htab),
                tris_id = proxy<space>(tris_id)] ZS_LAMBDA(int ei) mutable {
                    auto tet = tets.pack(dim_c<4>,"inds",ei,int_c);
                    order_indices(tet);
                    int tri_id = -1;
                    if(auto no = tris_htab.query(vec3i{tet[1],tet[2],tet[3]});no >= 0)
                        tri_id = tris_id[no];
                    if(auto no = tris_htab.query(vec3i{tet[0],tet[2],tet[3]});no >= 0)
                        tri_id = tris_id[no];
                    if(auto no = tris_htab.query(vec3i{tet[0],tet[1],tet[3]});no >= 0)
                        tri_id = tris_id[no];
                    if(auto no = tris_htab.query(vec3i{tet[0],tet[1],tet[2]});no >= 0)
                        tri_id = tris_id[no];
                    if(tri_id >= 0)
                        tris("ft_inds",tri_id) = zs::reinterpret_bits<float>(ei);
            });

            cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris)] ZS_LAMBDA(int ti) mutable {
                    auto ei = zs::reinterpret_bits<int>(tris("ft_inds",ti));
                    if(ei < 0) {
                        printf("dangling surface tri %d detected\n",ti);
                    }
            });

#else

            if(!tris.hasProperty("ft_inds")) {
                tris.append_channels(cudaPol,{{"ft_inds",1}});
            }
            TILEVEC_OPS::fill(cudaPol,tris,"ft_inds",zs::reinterpret_bits<float>((int)-1));

            // compute_ft_neigh_topo(cudaPol,verts,tris,tets,"ft_inds");

            
            auto bvh_thickness= compute_average_edge_length(cudaPol,verts,"x",tris);

            auto tetsBvh = LBvh<3,int,float>{};
            
            auto bvs = retrieve_bounding_volumes(cudaPol,verts,tets,wrapv<4>{},bvh_thickness,"x");
            tetsBvh.build(cudaPol,bvs);

            size_t nmTris = tris.size();
            cudaPol(zs::range(nmTris),
                [tets = proxy<space>({},tets),
                    verts = proxy<space>({},verts),
                    tris = proxy<space>({},tris),
                    tetsBvh = proxy<space>(tetsBvh)] ZS_LAMBDA(int ti) mutable {
                        auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                        tris("ft_inds",ti) = zs::reinterpret_bits<float>((int)-1);
                        int nm_found = 0;
                        auto cv = vec3::zeros();
                        for(int i = 0;i != 3;++i)
                            cv += verts.pack(dim_c<3>,"x",tri[i])/3.0f;
                        tetsBvh.iter_neighbors(cv,[&](int ntet) {
                            // if(ti == 0)
                            //     printf("test tet[%d] and tri[%d]\n",ntet,ti);
                            if(nm_found > 0)
                                return;
                            auto tet = tets.pack(dim_c<4>,"inds",ntet).reinterpret_bits(int_c);
                            for(int i = 0;i != 3;++i){
                                bool found_idx = false;
                                for(int j = 0;j != 4;++j)
                                    if(tet[j] == tri[i]){
                                        found_idx = true;
                                        break;
                                    }
                                if(!found_idx)
                                    return;
                            }

                            nm_found++;
                            tris("ft_inds",ti) = reinterpret_bits<float>(ntet);
                        });

                        if(nm_found == 0)
                            printf("found no neighbored tet for tri[%d]\n",ti);

            });
#endif

            set_output("zsparticles",zsparticles);
        }
};

ZENDEFNODE(BuildSurfFacetTetraNeighboring, {{{"zsparticles"}},
                            {
                                {"zsparticles"}
                            },
                            {
                                // {gParamType_String,"mark_tag","mark_tag"}
                            },
                            {"ZSGeometry"}});


};