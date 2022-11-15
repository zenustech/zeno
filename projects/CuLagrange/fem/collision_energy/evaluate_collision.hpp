#pragma once

#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>


#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "../../geometry/kernel/calculate_facet_normal.hpp"
#include "../../geometry/kernel/topology.hpp"
#include "../../geometry/kernel/compute_characteristic_length.hpp"
#include "../../geometry/kernel/calculate_bisector_normal.hpp"

#include "../../geometry/kernel/tiled_vector_ops.hpp"
#include "../../geometry/kernel/geo_math.hpp"


#include "../../geometry/kernel/calculate_edge_normal.hpp"

#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"

#include "vertex_face_sqrt_collision.hpp"
#include "vertex_face_collision.hpp"
#include "edge_edge_sqrt_collision.hpp"
#include "edge_edge_collision.hpp"

namespace zeno { namespace COLLISION_UTILS {

using T = float;
using bvh_t = zs::LBvh<3,int,T>;
using bv_t = zs::AABBBox<3, T>;
using vec3 = zs::vec<T, 3>;


template<int MAX_FP_COLLISION_PAIRS,typename Pol,
            typename PosTileVec,
            typename SurfPointTileVec,
            typename SurfLineTileVec,
            typename SurfTriTileVec,
            typename SurfTriNrmVec,
            typename SurfLineNrmVec,
            typename FPCollisionBuffer> 
void do_facet_point_collision_detection(Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const SurfPointTileVec& points,
    const SurfLineTileVec& lines,
    const SurfTriTileVec& tris,
    SurfTriNrmVec& sttemp,
    SurfLineNrmVec& setemp,
    FPCollisionBuffer& cptemp,
    // const bvh_t& stBvh,
    T in_collisionEps,T out_collisionEps) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto stBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,verts,tris,wrapv<3>{},(T)0.0,xtag);
        stBvh.build(cudaPol,bvs);

        auto avgl = compute_average_edge_length(cudaPol,verts,xtag,tris);
        auto bvh_thickness = 5 * avgl;

        if(!calculate_facet_normal(cudaPol,verts,xtag,tris,sttemp,"nrm")){
            throw std::runtime_error("fail updating facet normal");
        }       
        if(!COLLISION_UTILS::calculate_cell_bisector_normal(cudaPol,
            verts,xtag,
            lines,
            tris,
            sttemp,"nrm",
            setemp,"nrm")){
                throw std::runtime_error("fail calculate cell bisector normal");
        }       
        TILEVEC_OPS::fill<4>(cudaPol,cptemp,"inds",zs::vec<int,4>::uniform(-1).template reinterpret_bits<T>());
        TILEVEC_OPS::fill(cudaPol,cptemp,"inverted",reinterpret_bits<T>((int)0));
        cudaPol(zs::range(points.size()),[in_collisionEps = in_collisionEps,
                        out_collisionEps = out_collisionEps,
                        verts = proxy<space>({},verts),xtag,
                        sttemp = proxy<space>({},sttemp),
                        setemp = proxy<space>({},setemp),
                        points = proxy<space>({},points),
                        lines = proxy<space>({},lines),
                        tris = proxy<space>({},tris),
                        cptemp = proxy<space>({},cptemp),
                        stbvh = proxy<space>(stBvh),thickness = bvh_thickness] ZS_LAMBDA(int svi) mutable {
            auto vi = reinterpret_bits<int>(points("inds",svi));
            auto active = verts("active",vi);
            if(active < 1e-6)
                return;

            auto p = verts.template pack<3>(xtag,vi);
            auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};

            int nm_collision_pairs = 0;
            auto process_vertex_face_collision_pairs = [&](int stI) {
                if(nm_collision_pairs >= MAX_FP_COLLISION_PAIRS) 
                    return;

                auto tri = tris.pack(dim_c<3>, "inds",stI).reinterpret_bits(int_c);
                if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
                    return;

                bool is_active_tri = true;
                for(int i = 0;i != 3;++i)
                    if(verts("active",tri[i]) < 1e-6)
                        is_active_tri = false;
                if(!is_active_tri)
                    return;

                T dist = (T)0.0;

                // we should also neglect over deformed facet
                auto triRestArea = tris("area",stI);

                if(triRestArea < 1e-8)
                    return;

                auto triDeformedArea = LSL_GEO::area(
                    verts.template pack<3>(xtag,tri[0]),
                    verts.template pack<3>(xtag,tri[1]),
                    verts.template pack<3>(xtag,tri[2]));


                auto areaDeform = triDeformedArea / triRestArea;
                if(areaDeform < 1e-1)
                    return;

                // if(COLLISION_UTILS::is_inside_the_cell(verts,xtag,
                //         lines,tris,
                //         sttemp,"nrm",
                //         setemp,"nrm",
                //         stI,p,in_collisionEps,out_collisionEps,dist)) {
                // //     cptemp.template tuple<4>("inds",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = zs::vec<int,4>(vi,tri[0],tri[1],tri[2]).template reinterpret_bits<T>();
                // //     auto vertexFaceCollisionAreas = tris("area",stI) + points("area",svi); 
                // //     cptemp("area",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = vertexFaceCollisionAreas;   
                // //     if(vertexFaceCollisionAreas < 0)
                // //         printf("negative face area detected\n");  
                // //     int is_inverted = dist > (T)0.0 ? 1 : 0;  
                // //     cptemp("inverted",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = reinterpret_bits<T>(is_inverted);            
                // //     nm_collision_pairs++;  

                // }

                auto nrm = sttemp.template pack<3>("nrm",stI);
                
                auto seg = p - verts.template pack<3>(xtag,tri[0]);    

                // evaluate the avg edge length
                auto t0 = verts.template pack<3>(xtag,tri[0]);
                auto t1 = verts.template pack<3>(xtag,tri[1]);
                auto t2 = verts.template pack<3>(xtag,tri[2]);

                auto e01 = (t0 - t1).norm();
                auto e02 = (t0 - t2).norm();
                auto e12 = (t1 - t2).norm();

                // auto avge = (e01 + e02 + e12)/(T)3.0;

                T barySum = (T)1.0;
                T distance = COLLISION_UTILS::pointTriangleDistance(t0,t1,t2,p,barySum);
                // auto max_ratio = inset_ratio > outset_ratio ? inset_ratio : outset_ratio;
                // collisionEps = avge * max_ratio;
                auto collisionEps = seg.dot(nrm) > 0 ? out_collisionEps : in_collisionEps;

                if(barySum > 2)
                    return;

                if(distance > collisionEps)
                    return;
                dist = seg.dot(nrm);
                // if(dist < -(avge * inset_ratio + 1e-6) || dist > (outset_ratio * avge + 1e-6))
                //     return;

                // if the triangle cell is too degenerate
                if(!pointProjectsInsideTriangle(t0,t1,t2,p))
                    for(int i = 0;i != 3;++i) {
                            auto bisector_normal = get_bisector_orient(lines,tris,setemp,"nrm",stI,i);
                            // auto test = bisector_normal.cross(nrm).norm() < 1e-2;

                            seg = p - verts.template pack<3>(xtag,tri[i]);
                            if(bisector_normal.dot(seg) < 0)
                                return;

                        }
        
                // now the points is inside the cell

                cptemp.template tuple<4>("inds",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = zs::vec<int,4>(vi,tri[0],tri[1],tri[2]).template reinterpret_bits<T>();
                auto vertexFaceCollisionAreas = tris("area",stI) + points("area",svi); 
                cptemp("area",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = vertexFaceCollisionAreas;   
                if(vertexFaceCollisionAreas < 0)
                    printf("negative face area detected\n");  
                int is_inverted = dist > (T)0.0 ? 1 : 0;  
                cptemp("inverted",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = reinterpret_bits<T>(is_inverted);            
                nm_collision_pairs++;  

            };
            stbvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
        });
}

// template<int MAX_EE_COLLISION_PAIRS,typename Pol,
//     typename PosTileVec,
//     typename SurfPointTileVec,
//     typename SurfLineTileVec,
//     typename SurfTriTileVec,
//     typename SurfTriNrmVec,
//     typename SurfLineNrmVec,
//     typename PointNeighHash,
//     typename EECollisionBuffer>
// void do_edge_edge_collision_detection(Pol& cudaPol,
//     const PosTileVec& verts,const zs::SmallString& xtag,
//     const SurfPointTileVec& points,
//     const SurfLineTileVec& lines,
//     const SurfTriTileVec& tris,
//     SurfTriNrmVec& sttemp,
//     SurfLineNrmVec& setemp,
//     EECollisionBuffer& eetemp,
//     const PointNeighHash& pphash,
//     T in_collisionEps,T out_collisionEps) {
//         using namespace zs;
//         constexpr auto space = execspace_e::cuda;

//         auto seBvh = bvh_t{};
//         auto bvs = retrieve_bounding_volumes(cudaPol,verts,lines,wrapv<2>{},(T)0.0,xtag);

//         auto avgl = compute_average_edge_length(cudaPol,verts,xtag,lines);
//         auto bvh_thickness = 5 * avgl;

//         if(!calculate_facet_normal(cudaPol,verts,xtag,sttemp,"nrm"))
//             throw std::runtime_error("fail updating facet normal");

        
// }


template<int MAX_FP_COLLISION_PAIRS,
    typename Pol,
    typename PosTileVec,
    typename FPCollisionBuffer>
void evaluate_collision_grad_and_hessian(Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    FPCollisionBuffer& cptemp,
    T in_collisionEps,T out_collisionEps,
    T collisionStiffness,
    T mu,T lambda) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        TILEVEC_OPS::fill<12*12>(cudaPol,cptemp,"H",zs::vec<T,12*12>::zeros());
        TILEVEC_OPS::fill<12>(cudaPol,cptemp,"grad",zs::vec<T,12>::zeros());  
        // TILEVEC_OPS::fill(cudaPol,cptemp,"area",(T)0.0);

#if 0
        int nm_points = cptemp.size() / MAX_FP_COLLISION_PAIRS;
        cudaPol(zs::range(nm_points),
            [verts = proxy<space>({},verts),xtag,
                cptemp = proxy<space>({},cptemp),
                in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                stiffness = collisionStiffness,
                mu = mu,lam = lambda] ZS_LAMBDA(int pi) mutable {
            for(int i = 0;i != MAX_FP_COLLISION_PAIRS;++i)  {
                auto inds = cptemp.template pack<4>("inds",pi * MAX_FP_COLLISION_PAIRS + i).reinterpret_bits(int_c);
                for(int j = 0;j != 4;++j)
                    if(inds[j] < 0)
                        return;

                for(int j = 0;j != 4;++j){
                    auto active = verts("active",inds[j]);
                    if(active < 1e-6)
                        return;
                }
                vec3 cv[4] = {};
                for(int j = 0;j != 4;++j)
                    cv[j] = verts.template pack<3>(xtag,inds[j]);
            
                

                auto is_inverted = reinterpret_bits<int>(cptemp("inverted",pi * MAX_FP_COLLISION_PAIRS + i));
                auto ceps = is_inverted ? in_collisionEps : out_collisionEps;

                auto alpha = stiffness;
                auto beta = cptemp("area",pi * MAX_FP_COLLISION_PAIRS + i);
                cptemp.template tuple<12>("grad",pi * MAX_FP_COLLISION_PAIRS + i) = alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                cptemp.template tuple<12*12>("H",pi * MAX_FP_COLLISION_PAIRS + i) = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps);
            }
        });
#else
        cudaPol(zs::range(cptemp.size()),
            [verts = proxy<space>({},verts),xtag,
                cptemp = proxy<space>({},cptemp),
                in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                stiffness = collisionStiffness,
                mu = mu,lam = lambda] ZS_LAMBDA(int cpi) mutable {
                auto inds = cptemp.template pack<4>("inds",cpi).reinterpret_bits(int_c);
                for(int j = 0;j != 4;++j)
                    if(inds[j] < 0)
                        return;
                vec3 cv[4] = {};
                for(int j = 0;j != 4;++j)
                    cv[j] = verts.template pack<3>(xtag,inds[j]);
            
                auto is_inverted = reinterpret_bits<int>(cptemp("inverted",cpi));
                // auto ceps = is_inverted ? in_collisionEps : out_collisionEps;

                auto ceps = out_collisionEps;
                // ceps += (T)1e-2 * ceps;

                auto alpha = stiffness;
                auto beta = cptemp("area",cpi);
          
#if 0
                cptemp.template tuple<12>("grad",cpi) = alpha * beta * VERTEX_FACE_COLLISION::gradient(cv,mu,lam,out_collisionEps);
                cptemp.template tuple<12*12>("H",cpi) = alpha * beta * VERTEX_FACE_COLLISION::hessian(cv,mu,lam,out_collisionEps);
#else
                cptemp.template tuple<12>("grad",cpi) = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                cptemp.template tuple<12*12>("H",cpi) = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps); 
#endif

                // printf("cpi[%d] : %f %f %f\n",cpi,(float)alpha,(float)beta,(float)cptemp.template pack<12>("grad",cpi).norm());   

        });

#endif

    }


// template<int MAX_FP_COLLISION_PAIRS,
//             typename Pol,
//             typename SurfPointTileVec,
//             typename SurfLineTileVec,
//             typename SurfTriTileVec,
//             typename PosTileVec,
//             typename CellPointTileVec,
//             typename CellBisectorTileVec,
//             typename CellTriTileVec,
//             typename FPCollisionBuffer>
// void evaluate_collision_grad_and_hessian(Pol& cudaPol,
//     const PosTileVec& verts,
//     const zs::SmallString& xtag,
//     const SurfPointTileVec& points,
//     const SurfLineTileVec& lines,
//     const SurfTriTileVec& tris,
//     CellPointTileVec& sptemp,
//     CellBisectorTileVec& setemp,
//     CellTriTileVec& sttemp,
//     FPCollisionBuffer& cptemp,
//     T cellBvhThickness,
//     T collisionEps,
//     T collisionStiffness,
//     T mu,T lambda) {
//         using namespace zs;
//         constexpr auto space = execspace_e::cuda;
//         TILEVEC_OPS::fill<12*12>(cudaPol,cptemp,"H",zs::vec<T,12*12>::zeros());
//         TILEVEC_OPS::fill<3>(cudaPol,sttemp,"grad",zs::vec<T,3>::zeros());
//         TILEVEC_OPS::fill<3>(cudaPol,sptemp,"grad",zs::vec<T,3>::zeros());

//         cudaPol(zs::range(points.size()),
//             [   collisionEps = collisionEps,
//                 cellBvhThickness = cellBvhThickness,
//                 verts = proxy<space>({},verts),
//                 sttemp = proxy<space>({},sttemp),
//                 setemp = proxy<space>({},setemp),
//                 sptemp = proxy<space>({},sptemp),
//                 cptemp = proxy<space>({},cptemp),
//                 points = proxy<space>({},points),
//                 lines = proxy<space>({},lines),
//                 tris = proxy<space>({},tris),
//                 stbvh = proxy<space>(stbvh),xtag,
//                 collisionStiffness = collisionStiffness,
//                 mu = mu,lambda = lambda] ZS_LAMBDA(int pi) mutable {

//             auto vi = reinterpret_bits<int>(points("inds",pi));
//             auto p = verts.template pack<3>(xtag,vi);
//             auto bv = bv_t{get_bounding_box(p - cellBvhThickness, p + cellBvhThickness)};

//             vec3 collision_verts[4] = {};
//             collision_verts[0] = p;

//             int nm_collision_pairs = 0;
//             auto process_vertex_face_collision_pairs = [&](int stI) {
//                 if(nm_collision_pairs >= MAX_FP_COLLISION_PAIRS)     
//                     return;   

//                 auto tri = tris.pack(dim_c<3>, "inds",stI).reinterpret_bits(int_c);
//                 if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
//                     return;

//                 collision_verts[1] = verts.template pack<3>(xtag,tri[0]);
//                 collision_verts[2] = verts.template pack<3>(xtag,tri[1]);
//                 collision_verts[3] = verts.template pack<3>(xtag,tri[2]);

//                 // check whether the triangle is degenerate
//                 auto restArea = tris("area",stI);

//                 const auto e10 = collision_verts[2] - collision_verts[1];
//                 const auto e20 = collision_verts[3] - collision_verts[1];
//                 auto deformedArea = (T)0.5 * e10.cross(e20).norm();
//                 const T degeneracyEps = 1e-4;
//                 // skip the degenerate triangles
//                 const T relativeArea = deformedArea / (restArea + (T)1e-6);
//                 if(relativeArea < degeneracyEps)
//                     return;

//                 bool collide = false;

//                 if(COLLISION_UTILS::is_inside_the_cell(verts,xtag,
//                         lines,tris,
//                         sttemp,"nrm",
//                         setemp,"nrm",
//                         stI,p,collisionEps)) {
//                     collide = true;
//                 }

//                 if(!collide)
//                     return;

//                 auto vertexFaceCollisionAreas = tris("area",stI) + points("area",pi);

//                 auto grad = collisionStiffness * VERTEX_FACE_SQRT_COLLISION::gradient(collision_verts,mu,lambda,collisionEps) * vertexFaceCollisionAreas;
//                 auto hessian = collisionStiffness * VERTEX_FACE_SQRT_COLLISION::hessian(collision_verts,mu,lambda,collisionEps) * vertexFaceCollisionAreas;
                

//                 cptemp.template tuple<4>("inds",pi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = zs::vec<int,4>(vi,tri[0],tri[1],tri[2]).template reinterpret_bits<T>();      
//                 cptemp.template tuple<12*12>("H",pi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = hessian;
//                 // auto pf = zs::vec<T,3>{grad[0],grad[1],grad[2]};    
//                 zs::vec<T,3> tf[3] = {};
//                 for(int j = 0;j != 3;++j)
//                     tf[j] = zs::vec<T,3>{grad[j * 3 + 3 + 0],grad[j * 3 + 3 + 1],grad[j * 3 + 3 + 2]};     

//                 // auto avgtf = (tf[0] + tf[1] + tf[2])/(T)3.0;
//                 auto avgtf = (tf[0] + tf[1] + tf[2]);
//                 for(int j = 0;j != 3;++j)
//                     atomic_add(exec_cuda,&sttemp("grad",j,stI),avgtf[j]);


//                 auto fp_inds = tris.template pack<3>("fp_inds",stI).reinterpret_bits(int_c);
//                 for(int j = 0;j != 3;++j){
//                     atomic_add(exec_cuda,&sptemp("grad",j,pi),grad[j]);
//                     for(int k = 0;k != 3;++k)   {
//                         auto fp_idx = fp_inds[k];
//                         atomic_add(exec_cuda,&sptemp("grad",j,fp_idx),tf[k][j]);
//                     }
//                 }   

//                 nm_collision_pairs++;                   
//             };
//             stbvh.iter_neighbors(bv,process_vertex_face_collision_pairs);                
//         });
//     }

};

};