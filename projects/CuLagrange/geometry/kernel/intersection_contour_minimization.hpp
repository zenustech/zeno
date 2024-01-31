#pragma once


#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "Utils.hpp"

#include <iostream>

#include "geo_math.hpp"

namespace zeno {


    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename HalfEdgeTileVec,
        typename HTHashMap,
        typename ICMGradTileVec,
        auto space = Pol::exec_tag::value,
        typename T = typename PosTileVec::value_type>
    void eval_intersection_contour_minimization_gradient(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xtag,
            const HalfEdgeTileVec& halfedges,
            const TriTileVec& tris,
            HTHashMap& csHT,
            ICMGradTileVec& icm_grad,
            const HalfEdgeTileVec& halfedges_host,
            const TriTileVec& tris_host,
            bool use_global_scheme = false,
            bool use_barycentric_interpolator = false) {
        using namespace zs;
        auto exec_tag = wrapv<space>{};
        using vec2i = zs::vec<int,2>; 
        using vec3 = zs::vec<T,3>;
   
        icm_grad.resize(csHT.size());
        // local scheme
        pol(zip(zs::range(csHT.size()),csHT._activeKeys),[
            icm_grad = proxy<space>({},icm_grad),
            use_barycentric_interpolator = use_barycentric_interpolator,
            verts = proxy<space>({},verts),xtag = zs::SmallString(xtag),
            halfedges = proxy<space>({},halfedges),
            tris = proxy<space>({},tris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto hi = pair[0];
                auto ti = pair[1];

                auto hedge = half_edge_get_edge(hi,halfedges,tris);

                auto hti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
                auto htri = tris.pack(dim_c<3>,"inds",hti,int_c);

                auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);

                for(int i = 0;i != 3;++i)
                    if(tri[i] == hedge[0] || tri[i] == hedge[1])
                        return;

                vec3 halfedge_vertices[2] = {};
                vec3 tri_vertices[3] = {};
                vec3 htri_vertices[3] = {};

                for(int i = 0;i != 2;++i)
                    halfedge_vertices[i] = verts.pack(dim_c<3>,xtag,hedge[i]);

                for(int i = 0;i != 3;++i) {
                    tri_vertices[i] = verts.pack(dim_c<3>,xtag,tri[i]);
                    htri_vertices[i] = verts.pack(dim_c<3>,xtag,htri[i]);
                }

                auto tri_normal = LSL_GEO::facet_normal(tri_vertices[0],tri_vertices[1],tri_vertices[2]);
                auto htri_normal = LSL_GEO::facet_normal(htri_vertices[0],htri_vertices[1],htri_vertices[2]);
                auto halfedge_orient = (halfedge_vertices[1] - halfedge_vertices[0]).normalized();

                auto R = tri_normal.cross(htri_normal).normalized();

                auto halfedge_local_vertex_id = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
                auto halfedge_opposite_vertex_id = htri[(halfedge_local_vertex_id + 2) % 3];
                auto halfedge_opposite_vertex = verts.pack(dim_c<3>,xtag,halfedge_opposite_vertex_id);

                auto in_plane_edge_normal = halfedge_opposite_vertex - halfedge_vertices[0];
                in_plane_edge_normal = in_plane_edge_normal - in_plane_edge_normal.dot(halfedge_orient) * halfedge_orient;
                in_plane_edge_normal = in_plane_edge_normal.normalized();

                if(R.dot(in_plane_edge_normal) < 0)
                    R = -R;
                
                const auto& E = halfedge_orient;
                const auto& N = tri_normal;

                auto ER = E.dot(R);
                auto EN = E.dot(N);

                auto G = R - static_cast<T>(2.0) * ER / EN * N;
                // https://diglib.eg.org/bitstream/handle/10.2312/SCA.SCA12.311-316/311-316.pdf?sequence=1
                // point it out the formula in the original paper by Nadia.etc is inaccurate, although the bad effect is limited
                // auto G = R - ER / EN * N;
                icm_grad.tuple(dim_c<3>,"grad",ci) = G;
                icm_grad.tuple(dim_c<2>,"inds",ci) = pair.reinterpret_bits(float_c);
        }); 
        // global scheme

        if(use_global_scheme) { // TODO: the nvcc compiler won't pass here, check later
            
        }
    }   


    template<typename VECTOR3d,typename SCALER>
    constexpr auto eval_HT_contour_minimization_gradient(const VECTOR3d& ht0,
        const VECTOR3d& ht1,
        const VECTOR3d& ht2,
        const VECTOR3d& htnrm,
        const VECTOR3d& tnrm,
        SCALER& normal_coeff) {
            // auto htri_normal = LSL_GEO::facet_normal(ht0,ht1,ht2);  
            // auto tri_normal = LSL_GEO::facet_normal(t0,t1,t2); 
            const auto eps = 1e-6;

            const auto& htri_normal = htnrm;
            const auto& tri_normal = tnrm;
            auto R = tri_normal.cross(htri_normal).normalized();

            auto E = (ht1 - ht0).normalized();
            
            auto in_plane_edge_normal = ht2 - ht0;
            in_plane_edge_normal = in_plane_edge_normal - in_plane_edge_normal.dot(E) * E;
            // in_plane_edge_normal = in_plane_edge_normal.normalized();
    
            if(R.dot(in_plane_edge_normal) < 0)
                R = -R;

            const auto& N = tri_normal;

            auto ER = E.dot(R);
            auto EN = E.dot(N);

        // https://diglib.eg.org/bitstream/handle/10.2312/SCA.SCA12.311-316/311-316.pdf?sequence=1
        // point it out the formula in the original paper by Nadia.etc is inaccurate, although the bad effect is limited
        // auto G = R - ER / EN * N;
            // auto G = R - static_cast<T>(2.0) * ER / EN * N;
            // G = R;

            if(zs::abs(EN) > eps)
                normal_coeff = static_cast<T>(2.0) * ER / EN;
            else
                normal_coeff = 0;
            // normal_coeff = ER / EN;

            // if(isnan(normal_coeff)) {
            //     printf("nan normalcoeff : %f %f %f\n",R.norm(),E.norm(),EN);
            // }

            return R;
    }

    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename EdgeTileVec,
        typename HalfEdgeTileVec,
        typename HTHashMap,
        typename ICMGradTileVec,
        auto space = Pol::exec_tag::value,
        typename T = typename PosTileVec::value_type>
    void eval_self_intersection_contour_minimization_gradient(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xtag,
            const EdgeTileVec& edges,
            const HalfEdgeTileVec& halfedges,
            const TriTileVec& tris,
            const T& maximum_correction,
            const T& progressive_slope,
            const HTHashMap& csET,
            ICMGradTileVec& icm_grad,
            bool enforce_self_intersection_normal = false,
            bool recheck_intersection = false) {
        using namespace zs;
        auto exec_tag = wrapv<space>{};
        using vec2i = zs::vec<int,2>; 
        using vec3 = zs::vec<T,3>;
   
        constexpr auto eps = 1e-6;
        // icm_grad.resize(csET.size());
        // local scheme
        pol(zip(zs::range(csET.size()),csET._activeKeys),[
            recheck_intersection = recheck_intersection,
            icmGradOffset = icm_grad.getPropertyOffset("grad"),
            icm_grad = proxy<space>({},icm_grad),
            edges = proxy<space>({},edges),
            edgeOffset = edges.getPropertyOffset("inds"),
            edgeHeIndsOffset = edges.getPropertyOffset("he_inds"), 
            h0 = maximum_correction,
            eps = eps,
            enforce_self_intersection_normal = enforce_self_intersection_normal,
            g02 = progressive_slope * progressive_slope,
            verts = proxy<space>({},verts),
            xtagOffset = verts.getPropertyOffset(xtag),
            halfedges = proxy<space>({},halfedges),
            hfToFaceOffset = halfedges.getPropertyOffset("to_face"),
            hfVIDOffset = halfedges.getPropertyOffset("local_vertex_id"),
            hfOppoHeOffset = halfedges.getPropertyOffset("opposite_he"),
            tris = proxy<space>({},tris),
            triOffset = tris.getPropertyOffset("inds"),
            triNrmOffset = tris.getPropertyOffset("nrm"),
            triDOffset = tris.getPropertyOffset("d")] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto ei = pair[0];
                auto ti = pair[1];

                auto edge = edges.pack(dim_c<2>,edgeOffset,ei,int_c);
                auto tri = tris.pack(dim_c<3>,triOffset,ti,int_c);

                vec3 edge_vertices[2] = {};
                vec3 tri_vertices[3] = {};
                for(int i = 0;i != 2;++i)
                    edge_vertices[i] = verts.pack(dim_c<3>,xtagOffset,edge[i]);
                for(int i = 0;i != 3;++i)
                    tri_vertices[i] = verts.pack(dim_c<3>,xtagOffset,tri[i]);

                auto G = vec3::zeros();
                icm_grad.tuple(dim_c<3>,icmGradOffset,ci) = G;

                auto nrm = tris.pack(dim_c<3>,triNrmOffset,ti);

                if(recheck_intersection) {
                    if(nrm.norm() < eps)
                        return;
                    // printf("do the recheck intersection\n");

                    auto d = tris(triDOffset,ti);
                    const auto& ve = edge_vertices[1];
                    const auto& vs = edge_vertices[0];
                    const auto& tvs = tri_vertices;

                    auto en = (ve - vs).norm();
                    if(en < eps) {
                        // printf("reject due to too short the edge\n");
                        return;   
                    }
                        
                    auto is_parallel = zs::abs(nrm.dot(ve - vs) / en) < 1e-3;
                    if(is_parallel) {
                        // printf("reject due to parallel the edge\n");
                        return;    
                    }
                        
                    auto ts = vs.dot(nrm) + d;
                    auto te = ve.dot(nrm) + d;  
                    
                    if(ts * te > 0) {
                        // printf("reject due to same side\n");
                        return;
                    }

                    auto e0 = tvs[1] - tvs[0];
                    auto e1 = tvs[2] - tvs[0];
                    auto e2 = tvs[1] - tvs[2];

                    auto t = ts / (ts - te);
                    auto p = vs + t * (ve - vs);
                    auto q = p - tvs[0];

                    auto n0 = e0.cross(q);
                    auto n1 = e1.cross(q);
                    if(n0.dot(n1) > 0) {
                        return;
                    } 

                    q = p - tvs[2];
                    n1 = e1.cross(q);
                    auto n2 = e2.cross(q);

                    if(n1.dot(n2) < 0) {
                        return;
                    }

                    q = p - tvs[1];
                    n0 = e0.cross(q);
                    n2 = e2.cross(q);

                    if(n0.dot(n2) > 0) {
                        return;
                    }                    

                }

                int his[2] = {};
                his[0] = zs::reinterpret_bits<int>(edges(edgeHeIndsOffset,ei));
                his[1] = zs::reinterpret_bits<int>(halfedges(hfOppoHeOffset,his[0]));

                T normal_coeff = 0;
                for(auto hi : his){

                    if(hi < 0)
                        break;

                    auto hti = zs::reinterpret_bits<int>(halfedges(hfToFaceOffset,hi));
                    auto htri = tris.pack(dim_c<3>,triOffset,hti,int_c);
                    auto halfedge_local_vertex_id = zs::reinterpret_bits<int>(halfedges(hfVIDOffset,hi));
                    auto halfedge_opposite_vertex_id = htri[(halfedge_local_vertex_id + 2) % 3];
                    auto halfedge_opposite_vertex = verts.pack(dim_c<3>,xtagOffset,halfedge_opposite_vertex_id);

                    auto hnrm = tris.pack(dim_c<3>,triNrmOffset,hti);
                    if(hnrm.norm() < eps)
                        continue;

                    T dnc{};
                    G += eval_HT_contour_minimization_gradient(edge_vertices[0],
                        edge_vertices[1],   
                        halfedge_opposite_vertex,
                        hnrm,nrm,dnc);

                    if(enforce_self_intersection_normal && dnc > 0)
                        dnc = -dnc;

                    normal_coeff += dnc;
                }
                
                G -= normal_coeff * nrm;

                auto Gn = G.norm();
                auto Gn2 = G.l2NormSqr();
                G = h0 * G / zs::sqrt(Gn2 + g02);

                icm_grad.tuple(dim_c<3>,icmGradOffset,ci) = G;
        }); 
    }


    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename EdgeTileVec,
        typename HalfEdgeTileVec,
        typename HTHashMap,
        typename ICMGradTileVec,
        auto space = Pol::exec_tag::value,
        typename T = typename PosTileVec::value_type>
    void eval_intersection_contour_minimization_gradient_of_edgeA_with_triB(Pol& pol,
            const PosTileVec& verts,const zs::SmallString& xtag,
            const EdgeTileVec& edges,
            const HalfEdgeTileVec& halfedges,
            const TriTileVec& tris,
            const PosTileVec& kverts,const zs::SmallString& kxtag,
            const TriTileVec& ktris,
            const T& maximum_correction,
            const T& progressive_slope,
            HTHashMap& csET,
            ICMGradTileVec& icm_grad,
            bool enforce_triangle_normal = false,
            bool recheck_intersection = false) {
        using namespace zs;
        auto exec_tag = wrapv<space>{};
        using vec2i = zs::vec<int,2>; 
        using vec3 = zs::vec<T,3>;

        constexpr auto eps = 1e-6;
   
        // icm_grad.resize(csET.size());
        // local scheme
        pol(zip(zs::range(csET.size()),csET._activeKeys),[
            enforce_triangle_normal = enforce_triangle_normal,
            icm_grad = proxy<space>({},icm_grad),
            icmGradOffset = icm_grad.getPropertyOffset("grad"),
            verts = proxy<space>({},verts),
            xtagOffset = verts.getPropertyOffset(xtag),
            edges = proxy<space>({},edges),
            edgeOffset = edges.getPropertyOffset("inds"),
            edgeHeOffest = edges.getPropertyOffset("he_inds"),
            halfedges = proxy<space>({},halfedges),
            recheck_intersection = recheck_intersection,
            oppoHeOffset = halfedges.getPropertyOffset("opposite_he"),
            toFaceOffset = halfedges.getPropertyOffset("to_face"),
            localVIDOffset = halfedges.getPropertyOffset("local_vertex_id"),
            tris = proxy<space>({},tris),
            triOffset = tris.getPropertyOffset("inds"),
            triNrmOffset = tris.getPropertyOffset("nrm"),
            h0 = maximum_correction,
            eps = eps,
            g02 = progressive_slope * progressive_slope,
            kverts = proxy<space>({},kverts),
            kxtagOffset = kverts.getPropertyOffset(kxtag),
            ktris = proxy<space>({},ktris),
            ktriOffset = ktris.getPropertyOffset("inds"),
            ktriNrmOffset = ktris.getPropertyOffset("nrm"),
            ktriDOffset = ktris.getPropertyOffset("d")] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto ei = pair[0];
                auto kti = pair[1];

                auto edge = edges.pack(dim_c<2>,edgeOffset,ei,int_c);
                auto ktri = ktris.pack(dim_c<3>,ktriOffset,kti,int_c);

                vec3 edge_vertices[2] = {};
                vec3 ktri_vertices[3] = {};
                for(int i = 0;i != 2;++i)
                    edge_vertices[i] = verts.pack(dim_c<3>,xtagOffset,edge[i]);
                for(int i = 0;i != 3;++i)
                    ktri_vertices[i] = kverts.pack(dim_c<3>,kxtagOffset,ktri[i]);

                icm_grad.tuple(dim_c<3>,icmGradOffset,ci) = vec3::zeros();

                auto knrm = ktris.pack(dim_c<3>,ktriNrmOffset,kti);


                if(recheck_intersection) {
                    if(knrm.norm() < eps)
                        return;

                    // auto ktnrm = ktris.pack(dim_c<3>,ktriNrmOffset,kti);
                    auto d = ktris(ktriDOffset,kti);

                    const auto& ve = edge_vertices[1];
                    const auto& vs = edge_vertices[0];
                    const auto& ktvs = ktri_vertices;

                    auto en = (ve - vs).norm();
                    if(en < eps)
                        return;                        

                    auto is_parallel = zs::abs(knrm.dot(ve - vs) / en) < 1e-3;
                    if(is_parallel)
                        return;      
                        
                    auto ts = vs.dot(knrm) + d;
                    auto te = ve.dot(knrm) + d;  
                    
                    if(ts * te > 0) 
                        return;

                    auto e0 = ktvs[1] - ktvs[0];
                    auto e1 = ktvs[2] - ktvs[0];
                    auto e2 = ktvs[1] - ktvs[2];

                    auto t = ts / (ts - te);
                    auto p = vs + t * (ve - vs);
                    auto q = p - ktvs[0];

                    auto n0 = e0.cross(q);
                    auto n1 = e1.cross(q);
                    if(n0.dot(n1) > 0) {
                        // if(is_intersect)
                        //     printf("should intersect but n0 * n1 = %f > 0\n",(float)(n0.dot(n1)));
                        return;
                    }


                    q = p - ktvs[2];
                    n1 = e1.cross(q);
                    auto n2 = e2.cross(q);

                    if(n1.dot(n2) < 0) {
                        return;
                    }

                    q = p - ktvs[1];
                    n0 = e0.cross(q);
                    n2 = e2.cross(q);

                    if(n0.dot(n2) > 0) {
                        return;
                    }
                }


                auto G = vec3::zeros();

                int his[2] = {};
                his[0] = zs::reinterpret_bits<int>(edges(edgeHeOffest,ei));
                his[1] = zs::reinterpret_bits<int>(halfedges(oppoHeOffset,his[0]));

                T normal_coeff = 0;

                for(auto hi : his){
                    if(hi < 0)
                        break;

                    auto hti = zs::reinterpret_bits<int>(halfedges(toFaceOffset,hi));
                    auto htri = tris.pack(dim_c<3>,triOffset,hti,int_c);
                    auto halfedge_local_vertex_id = zs::reinterpret_bits<int>(halfedges(localVIDOffset,hi));
                    auto halfedge_opposite_vertex_id = htri[(halfedge_local_vertex_id + 2) % 3];
                    auto halfedge_opposite_vertex = verts.pack(dim_c<3>,xtagOffset,halfedge_opposite_vertex_id);

                    auto hnrm = tris.pack(dim_c<3>,triNrmOffset,hti);
                    if(hnrm.norm() < eps)
                        continue;

                    if(hnrm.cross(knrm).norm() < eps)
                        continue;

                    T dnc{};

                    G += eval_HT_contour_minimization_gradient(edge_vertices[0],
                        edge_vertices[1],
                        halfedge_opposite_vertex,
                        hnrm,knrm,dnc);

                    if(enforce_triangle_normal && dnc > 0) {
                        dnc = -dnc;
                    }

                    normal_coeff += dnc;
                }

                G -= normal_coeff * knrm;

                // auto Gn = G.norm();
                // auto Gn2 = G.l2NormSqr();
                // G = h0 * G / zs::sqrt(Gn2 + g02);

                icm_grad.tuple(dim_c<3>,icmGradOffset,ci) = G;
        }); 
    }

};