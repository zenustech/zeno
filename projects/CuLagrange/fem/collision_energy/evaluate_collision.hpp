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
// #include "edge_edge_sqrt_collision.hpp"
// #include "edge_edge_collision.hpp"

namespace zeno { namespace COLLISION_UTILS {

using T = float;
using bvh_t = zs::LBvh<3,int,T>;
using bv_t = zs::AABBBox<3, T>;
using vec3 = zs::vec<T, 3>;



template<typename Pol,
            typename PosTileVec,
            typename SurfPointTileVec,
            typename SurfTriTileVec,
            typename SurfTriNrmTileVec> 
void do_facet_point_collsion_detection_and_compute_surface_normal(Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const SurfPointTileVec& points,
    const SurfTriTileVec& tris,
    SurfTriNrmTileVec& triNrmBuffer,
    zs::Vector<zs::vec<int,4>>& csPT,
    int& nm_collisions,T in_collisionEps,T out_collisionEps) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto avgl = compute_average_edge_length(cudaPol,verts,xtag,tris);
        auto bvh_thickness = 3 * avgl;
        if(!calculate_facet_normal(cudaPol,verts,xtag,tris,triNrmBuffer,"nrm")){
            throw std::runtime_error("fail updating facet normal");
        } 

        auto spBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,verts,points,wrapv<1>{},(T)bvh_thickness,xtag);
        spBvh.build(cudaPol,bvs);

        zs::Vector<int> nm_csPT{points.get_allocator(),1};
        nm_csPT.setVal(0);

        cudaPol(zs::range(tris.size()),[in_collisionEps = in_collisionEps,
            out_collisionEps = out_collisionEps,
            verts = proxy<space>({},verts),
            points = proxy<space>({},points),
            tris = proxy<space>({},tris),
            triNrmBuffer = proxy<space>({},triNrmBuffer),
            nm_csPT = proxy<space>(nm_csPT),
            xtag = xtag,
            csPT = proxy<space>(csPT),
            spBvh = proxy<space>(spBvh),thickness = bvh_thickness] ZS_LAMBDA(int stI) {
                auto tri = tris.pack(dim_c<3>,"inds",stI,int_c);
                if(verts.hasProperty("active"))
                    for(int i = 0;i != 3;++i)
                        if(verts("active",tri[i]) < 1e-6)
                            return;
                
                auto cp = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    cp += verts.pack(dim_c<3>,xtag,tri[i]) / (T)3.0;
                auto bv = bv_t{get_bounding_box(cp - thickness,cp + thickness)};
            
                auto tnrm = triNrmBuffer.pack(dim_c<3>,"nrm",stI);
                vec3 tvs[3] = {};
                for(int i = 0;i != 3;++i)
                    tvs[i] = verts.pack(dim_c<3>,xtag,tri[i]);

                auto ntris = tris.pack(dim_c<3>,"ff_inds",stI,int_c);
                vec3 bnrms[3] = {};
                for(int i = 0;i != 3;++i){
                    auto nti = ntris[i];
                    auto edge_normal = vec3::zeros();
                    if(nti < 0)
                        edge_normal = tnrm;
                    else{
                        edge_normal = tnrm + triNrmBuffer.pack(dim_c<3>,"nrm",nti);
                        edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                    }
                    auto e01 = tvs[(i + 1) % 3] - tvs[(i + 0) % 3];
                    bnrms[i] = edge_normal.cross(e01).normalized();
                }  

                auto process_vertex_face_collision_pairs = [&](int spI) {
                    auto vi = reinterpret_bits<int>(points("inds",spI));
                    if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
                        return;
                    if(verts.hasProperty("active"))
                        if(verts("active",vi) < 1e-6)
                            return;

                    auto p = verts.pack(dim_c<3>,xtag,vi);
                    auto seg = p - tvs[0];
                    auto dist = seg.dot(tnrm);
                    if(dist < 0 && verts.hasProperty("ring_mask")) {
                        int RING_MASK = zs::reinterpret_bits<int>(verts("ring_mask",vi));
                        if(RING_MASK == 0)
                            return;
                        // bool is_same_ring = false;
                        int TRING_MASK = 0;
                        for(int i = 0;i != 3;++i) {
                            // auto TGIA_TAG = reinterpret_bits<int>(verts("ring_mask",tri[i]));
                            TRING_MASK |= zs::reinterpret_bits<int>(verts("ring_mask",tri[i]));
                            // if((TGIA_TAG | GIA_TAG) > 0)
                            //     is_same_ring = true;
                        }
                        RING_MASK = RING_MASK & TRING_MASK;
                        // the point and the tri should belong to the same ring
                        if(RING_MASK == 0)
                            return;

                        // now the two pair belong to the same ring, check whether they belong black-white loop, and have different colors 

                        // {
#if 1
                        auto COLOR_MASK = reinterpret_bits<int>(verts("color_mask",vi));
                        auto TYPE_MASK = reinterpret_bits<int>(verts("type_mask",vi));

                        // only check the common type-1(white-black loop) rings
                        int TTYPE_MASK = 0;
                        for(int i = 0;i != 3;++i)
                            TTYPE_MASK |= reinterpret_bits<int>(verts("type_mask",tri[i]));
                        
                        RING_MASK &= (TYPE_MASK & TTYPE_MASK);
                        // int nm_common_rings = 0;
                        // while()
                        // as long as there is one ring in which the pair have different colors, neglect the pair
                        int curr_ri_mask = 1;
                        for(;RING_MASK > 0;RING_MASK = RING_MASK >> 1,curr_ri_mask = curr_ri_mask << 1) {
                            if(RING_MASK & 1) {
                                for(int i = 0;i != 3;++i) {
                                    auto TCOLOR_MASK = reinterpret_bits<int>(verts("color_mask",tri[i])) & curr_ri_mask;
                                    auto VCOLOR_MASK = reinterpret_bits<int>(verts("color_mask",vi)) & curr_ri_mask;
                                    if(TCOLOR_MASK == VCOLOR_MASK)
                                        return;
                                }
                            }
                        }
#endif
                        // }
                    }

                    auto collisionEps = dist > 0 ? out_collisionEps : in_collisionEps;

                    auto barySum = (T)1.0;
                    T distance = LSL_GEO::pointTriangleDistance(tvs[0],tvs[1],tvs[2],p,barySum);

                    if(distance > collisionEps)
                        return;

                    if(barySum > (T)(1.0 + 1e-6)) {
                        for(int i = 0;i != 3;++i){
                            seg = p - tvs[i];
                            if(bnrms[i].dot(seg) < 0)
                                return;
                        }
                    }

                    csPT[atomic_add(exec_cuda,&nm_csPT[0],(int)1)] = zs::vec<int,4>{vi,tri[0],tri[1],tri[2]};
                };
                spBvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
        });

        nm_collisions = nm_csPT.getVal(0);
}

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
    FPCollisionBuffer& fp_collision_buffer,
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
        // if(!COLLISION_UTILS::calculate_cell_bisector_normal(cudaPol,
        //     verts,xtag,
        //     lines,
        //     tris,
        //     sttemp,"nrm",
        //     setemp,"nrm")){
        //         throw std::runtime_error("fail calculate cell bisector normal");
        // }       
        TILEVEC_OPS::fill<4>(cudaPol,fp_collision_buffer,"inds",zs::vec<int,4>::uniform(-1).template reinterpret_bits<T>());
        TILEVEC_OPS::fill(cudaPol,fp_collision_buffer,"inverted",reinterpret_bits<T>((int)0));
        cudaPol(zs::range(points.size()),[in_collisionEps = in_collisionEps,
                        out_collisionEps = out_collisionEps,
                        verts = proxy<space>({},verts),xtag,
                        sttemp = proxy<space>({},sttemp),
                        setemp = proxy<space>({},setemp),
                        points = proxy<space>({},points),
                        lines = proxy<space>({},lines),
                        tris = proxy<space>({},tris),
                        fp_collision_buffer = proxy<space>({},fp_collision_buffer),
                        stbvh = proxy<space>(stBvh),thickness = bvh_thickness] ZS_LAMBDA(int svi) mutable {
            auto vi = reinterpret_bits<int>(points("inds",svi));
            auto active = verts("active",vi);
            bool is_active_vert = true;
            if(active < 1e-6){
                is_active_vert = false;
                return;
            }

// EDIT CANCEL INVERSION
            // if(verts.hasProperty("is_verted")) {
            //     auto is_inverted =reinterpret_bits<int>(verts("is_inverted",vi));
            //     if(is_inverted)
            //         return;
            // }

            // if(verts.hasProperty("gia_tag")) {
            //     auto gia_tag = verts("gia_tag",vi);
            //     if(gia_tag < (T)0.5)
            //         return;
            // }

            auto p = verts.template pack<3>(xtag,vi);
            auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};

            int nm_collision_pairs = 0;
            auto process_vertex_face_collision_pairs = [&](int stI) {
                if(nm_collision_pairs >= MAX_FP_COLLISION_PAIRS) 
                    return;

                auto tri = tris.pack(dim_c<3>, "inds",stI).reinterpret_bits(int_c);
                if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
                    return;

// EDIT CANCEL INVERSION
                // if(verts.hasProperty("is_verted")) {

                //     for(int i = 0;i != 3;++i)
                //         if(reinterpret_bits<int>(verts("is_inverted",tri[i])))
                //             return;

                // }

                // if(verts.hasProperty("gia_tag")) {
                //     for(int i = 0;i != 3;++i)
                //         if(verts("gia_tag",tri[i]) < (T)0.5)
                //             return;
                //     // auto gia_tag = verts("gia_tag",vi);
                //     // if(gia_tag < (T)0.5)
                //     //     return;
                // }


                bool is_active_tri = true;
                for(int i = 0;i != 3;++i)
                    if(verts("active",tri[i]) < 1e-6)
                        return;

                T dist = (T)0.0;

                // we should also neglect over deformed facet
                // auto triRestArea = tris("area",stI);

                // if(triRestArea < 1e-8)
                //     return;

                // auto triDeformedArea = LSL_GEO::area(
                //     verts.template pack<3>(xtag,tri[0]),
                //     verts.template pack<3>(xtag,tri[1]),
                //     verts.template pack<3>(xtag,tri[2]));


                // auto areaDeform = triDeformedArea / triRestArea;
                // if(areaDeform < 1e-1)
                //     return;

                auto nrm = sttemp.template pack<3>("nrm",stI);
                
                auto seg = p - verts.template pack<3>(xtag,tri[0]);    

                // evaluate the avg edge length
                // auto t0 = verts.template pack<3>(xtag,tri[0]);
                // auto t1 = verts.template pack<3>(xtag,tri[1]);
                // auto t2 = verts.template pack<3>(xtag,tri[2]);
                vec3 tvs[3] = {};
                for(int i = 0;i != 3;++i)
                    tvs[i] = verts.template pack<3>(xtag,tri[i]);

                auto e01 = (tvs[0] - tvs[1]).norm();
                auto e02 = (tvs[0] - tvs[2]).norm();
                auto e12 = (tvs[1] - tvs[2]).norm();

                // auto avge = (e01 + e02 + e12)/(T)3.0;

                T barySum = (T)1.0;
                T distance = LSL_GEO::pointTriangleDistance(tvs[0],tvs[1],tvs[2],p,barySum);
                // auto max_ratio = inset_ratio > outset_ratio ? inset_ratio : outset_ratio;
                // collisionEps = avge * max_ratio;
                dist = seg.dot(nrm);
                auto collisionEps = dist > 0 ? out_collisionEps : in_collisionEps;
                if(distance > collisionEps)
                    return;

                if(dist < 0 && verts.hasProperty("gia_tag")) {  
                    auto GIA_TAG = reinterpret_bits<int>(verts("gia_tag",vi));
                    if(GIA_TAG == 0)
                        return;
                    
                    // for(int i = 0;i != 3;++i)
                    //     if(verts("gia_tag",tri[i]) < (T)0.5)
                    //         return;

                    bool is_same_ring = false;
                    for(int i = 0;i != 3;++i){
                        auto TGIA_TAG = reinterpret_bits<int>(verts("gia_tag",tri[i]));
                        if((TGIA_TAG | GIA_TAG) > 0)
                            is_same_ring = true;
                    }
                    if(!is_same_ring)
                        return;
                }


                // if()

#if 0

                if(barySum > 1.01){
                //     return;
                // else {


                    // if(dist < -(avge * inset_ratio + 1e-6) || dist > (outset_ratio * avge + 1e-6))
                    //     return;

                    // if the triangle cell is too degenerate
                    // if(!LSL_GEO::pointProjectsInsideTriangle(t0,t1,t2,p))
                        for(int i = 0;i != 3;++i) {
                                auto bisector_normal = get_bisector_orient(lines,tris,setemp,"nrm",stI,i);
                                // auto test = bisector_normal.cross(nrm).norm() < 1e-2;

                                seg = p - verts.template pack<3>(xtag,tri[i]);
                                if(bisector_normal.dot(seg) < 0)
                                    return;

                            }
                }
#else
                if(barySum > 1.01) {
                    auto ntris = tris.pack(dim_c<3>,"ff_inds",stI,int_c);
                    vec3 bnrms[3] = {};
                    for(int i = 0;i != 3;++i){
                        auto nti = ntris[i];
                        auto edge_normal = vec3::zeros();
                        if(nti < 0)
                            edge_normal = nrm;
                        else{
                            edge_normal = nrm + sttemp.pack(dim_c<3>,"nrm",nti);
                            edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                        }
                        auto e01 = tvs[(i + 1) % 3] - tvs[(i + 0) % 3];
                        bnrms[i] = edge_normal.cross(e01).normalized();
                    }  

                    for(int i = 0;i != 3;++i) {
                        // auto bisector_normal = get_bisector_orient(lines,tris,setemp,"nrm",stI,i);
                        // auto test = bisector_normal.cross(nrm).norm() < 1e-2;

                        seg = p - tvs[i];
                        if(bnrms[i].dot(seg) < 0)
                            return;

                    }
                }
#endif
        
                // now the points is inside the cell

                fp_collision_buffer.template tuple<4>("inds",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = zs::vec<int,4>(vi,tri[0],tri[1],tri[2]).template reinterpret_bits<T>();
                auto vertexFaceCollisionAreas = tris("area",stI) + points("area",svi); 
                fp_collision_buffer("area",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = vertexFaceCollisionAreas;   
                if(vertexFaceCollisionAreas < 0)
                    printf("negative face area detected\n");  
                int is_inverted = dist > (T)0.0 ? 1 : 0;  
                fp_collision_buffer("inverted",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = reinterpret_bits<T>(is_inverted);            
                nm_collision_pairs++;  

            };
            stbvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
        });
}


template<int MAX_KINEMATIC_COLLISION_PAIRS,
    typename Pol,
    typename PosTileVec,
    typename SurfPointTileVec,
    typename SurfLineTileVec,
    typename SurfTriTileVec,
    typename SurfLineNrmTileVec,
    typename SurfTriNrmTileVec,
    typename KPosTileVec,
    typename KCollisionBuffer>
void do_kinematic_point_collision_detection(Pol& cudaPol,
    PosTileVec& verts,const zs::SmallString& xtag,
    const SurfPointTileVec& points,
    SurfLineTileVec& lines,
    SurfTriTileVec& tris,
    SurfLineNrmTileVec& nrmLines,
    SurfTriNrmTileVec& nrmTris,
    const KPosTileVec& kverts,
    KCollisionBuffer& kc_buffer,
    T in_collisionEps,T out_collisionEps,bool update_normal = true) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto stBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,verts,tris,wrapv<3>{},(T)0.0,xtag);
        stBvh.build(cudaPol,bvs);

        auto avgl = compute_average_edge_length(cudaPol,verts,xtag,tris);
        auto bvh_thickness = 5 * avgl;    

        if(update_normal) {
            if(!calculate_facet_normal(cudaPol,verts,xtag,tris,nrmTris,"nrm")){
                throw std::runtime_error("fail updating kinematic facet normal");
            }       
            if(!COLLISION_UTILS::calculate_cell_bisector_normal(cudaPol,
                verts,xtag,
                lines,
                tris,
                nrmTris,"nrm",
                nrmLines,"nrm")){
                    throw std::runtime_error("fail calculate cell bisector normal");
            }    
        }

        TILEVEC_OPS::fill<2>(cudaPol,kc_buffer,"inds",zs::vec<int,2>::uniform(-1).template reinterpret_bits<T>());
        TILEVEC_OPS::fill(cudaPol,kc_buffer,"inverted",reinterpret_bits<T>((int)0));

        cudaPol(zs::range(kverts.size()),[in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                verts = proxy<space>({},verts),xtag,
                lines = proxy<space>({},lines),
                tris = proxy<space>({},tris),
                nrmTris = proxy<space>({},nrmTris),
                nrmLines = proxy<space>({},nrmLines),
                kverts = proxy<space>({},kverts),
                kc_buffer = proxy<space>({},kc_buffer),
                stBvh = proxy<space>(stBvh),thickness = bvh_thickness] ZS_LAMBDA(int kvi) mutable {

                    auto kp = kverts.pack(dim_c<3>,"x",kvi);
                    auto bv = bv_t{get_bounding_box(kp - thickness,kp + thickness)};

                    int nm_collision_pairs = 0;
                    auto process_kinematic_vertex_face_collision_pairs = [&](int stI) {
                        if(nm_collision_pairs >= MAX_KINEMATIC_COLLISION_PAIRS)
                            return;
                        auto tri = tris.pack(dim_c<3>,"inds",stI).reinterpret_bits(int_c);
                        for(int i = 0;i != 3;++i)
                            if(verts("k_active",tri[i]) < 1e-6)
                                return;

                        auto average_thickness = (T)0.0;
                        if(verts.hasProperty("k_thickness")){
                            // average_thickness = (T)0.0;
                            for(int i = 0;i != 3;++i)
                                average_thickness += verts("k_thickness",tri[i])/(T)3.0;
                        }



                        if(verts.hasProperty("is_verted")) {

                            for(int i = 0;i != 3;++i)
                                if(reinterpret_bits<int>(verts("is_inverted",tri[i])))
                                    return;

                        }

                        T dist = (T)0.0;

                        // if(tri[0] > 5326 || tri[1] > 5326 || tri[2] > 5326){
                        //     printf("invalid tri detected : %d %d %d\n",tri[0],tri[1],tri[2]);
                        //     return;
                        // }

                        auto nrm = nrmTris.pack(dim_c<3>,"nrm",stI);
                        auto seg = kp - verts.pack(dim_c<3>,xtag,tri[0]);


                        auto t0 = verts.pack(dim_c<3>,xtag,tri[0]);
                        auto t1 = verts.pack(dim_c<3>,xtag,tri[1]);
                        auto t2 = verts.pack(dim_c<3>,xtag,tri[2]);

                        auto e01 = (t0 - t1).norm();
                        auto e02 = (t0 - t2).norm();
                        auto e12 = (t1 - t2).norm();

                        T barySum = (T)1.0;
                        T distance = LSL_GEO::pointTriangleDistance(t0,t1,t2,kp,barySum);

                        dist = seg.dot(nrm);
                        // increase the stability, the tri must already in collided in the previous frame before been penerated in the current frame
                        // if(dist > 0 && tris("collide",stI) < 0.5)
                        //     return;

                        auto collisionEps = dist < 0 ? out_collisionEps * ((T)1.0 + average_thickness) : in_collisionEps;

                        if(barySum > 1.1)
                            return;

                        if(distance > collisionEps)
                            return;

                        // if(dist < -(avge * inset_ratio + 1e-6) || dist > (outset_ratio * avge + 1e-6))
                        //     return;

                        // if the triangle cell is too degenerate
                        if(!LSL_GEO::pointProjectsInsideTriangle(t0,t1,t2,kp))
                            for(int i = 0;i != 3;++i) {
                                auto bisector_normal = get_bisector_orient(lines,tris,nrmLines,"nrm",stI,i);
                                // auto test = bisector_normal.cross(nrm).norm() < 1e-2;
                                seg = kp - verts.pack(dim_c<3>,xtag,tri[i]);
                                if(bisector_normal.dot(seg) < 0)
                                    return;
                            }

                        kc_buffer.template tuple<2>("inds",kvi * MAX_KINEMATIC_COLLISION_PAIRS + nm_collision_pairs) = zs::vec<int,2>(kvi,stI).template reinterpret_bits<T>();
                        auto vertexFaceCollisionAreas = /*tris("area",stI) + */kverts("area",kvi); 
                        kc_buffer("area",kvi * MAX_KINEMATIC_COLLISION_PAIRS + nm_collision_pairs) = vertexFaceCollisionAreas;   
                        // if(vertexFaceCollisionAreas < 0)
                        //     printf("negative face area detected\n");  
                        int is_inverted = dist > (T)0.0 ? 1 : 0;  
                        kc_buffer("inverted",kvi * MAX_KINEMATIC_COLLISION_PAIRS + nm_collision_pairs) = reinterpret_bits<T>(is_inverted);            
                        nm_collision_pairs++;  
                    };
                    stBvh.iter_neighbors(bv,process_kinematic_vertex_face_collision_pairs);
            });
}



// template<typename Pol,
//     typename PosTileVec,
//     typename SurfPointTileVec,
//     typename SurfLineTileVec,
//     typename SurfTriTileVec,
//     typename SurfTriNrmVec,
//     typename SurfLineNrmVec,
//     typename EECollisionBuffer>
// void do_edge_edge_collision_detection(Pol& cudaPol,
//     const PosTileVec& verts,const zs::SmallString& xtag,
//     const SurfPointTileVec& points,
//     const SurfLineTileVec& lines,
//     const SurfTriTileVec& tris,
//     SurfTriNrmVec& sttemp,SurfLineNrmVec& setemp,
//     EECollisionBuffer& ee_collision_buffer,
//     // const PointNeighHash& pphash,// we might need an one-ring neighbor removal tech
//     T in_collisionEps,T out_collisionEps) {
//         using namespace zs;
//         constexpr auto space = execspace_e::cuda;

//         auto seBvh = bvh_t{};
//         auto bvs = retrieve_bounding_volumes(cudaPol,verts,lines,wrapv<2>{},(T)0.0,xtag);
//         seBvh.build(cudaPol,bvs);

//         auto avgl = compute_average_edge_length(cudaPol,verts,xtag,lines);
//         auto bvh_thickness = 5 * avgl;


//         if(!sttemp.hasProperty("nrm") || sttemp.getChannelSize("nrm") != 3)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid sttemp's \"nrm\" channel");

//         if(!setemp.hasProperty("nrm") || setemp.getChannelSize("nrm") != 3)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid setemp's \"nrm\" channel");

//         if(setemp.size() != lines.size())
//             throw std::runtime_error("setemp.size() != lines.size()");
//         if(sttemp.size() != tris.size())
//             throw std::runtime_error("sttemp.size() != tris.size()");

//         // std::cout << "do edge edge collision detection" << std::endl;
//         if(!calculate_facet_normal(cudaPol,verts,xtag,tris,sttemp,"nrm"))
//             throw std::runtime_error("do_edge_edge_collision_detection::fail updating facet normal");


//         // std::cout << "calculate edge normal" << std::endl;

//         if(!calculate_edge_normal_from_facet_normal(cudaPol,sttemp,"nrm",setemp,"nrm",lines))
//             throw std::runtime_error("do_edge_edge_collision_detection::fail updating edge normal");

//         if(ee_collision_buffer.size() != lines.size())
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid ee_colliision_buffer size");

//         if(!ee_collision_buffer.hasProperty("inds") || ee_collision_buffer.getChannelSize("inds") != 4)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid ee_colliision_buffer's \"inds\" channel");

//         if(!ee_collision_buffer.hasProperty("inverted") || ee_collision_buffer.getChannelSize("inverted") != 1)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid ee_colliision_buffer's \"inverted\" channel");

//         if(!ee_collision_buffer.hasProperty("abary") || ee_collision_buffer.getChannelSize("abary") != 2)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid ee_colliision_buffer's \"abary\" channel");

//         if(!ee_collision_buffer.hasProperty("bbary") || ee_collision_buffer.getChannelSize("bbary") != 2)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid ee_colliision_buffer's \"bbary\" channel");

//         if(!ee_collision_buffer.hasProperty("area") || ee_collision_buffer.getChannelSize("area") != 1)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid ee_colliision_buffer's \"area\" channel");

//         if(!lines.hasProperty("area") || lines.getChannelSize("area") != 1)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid lines's \"area\" channel");

//         TILEVEC_OPS::fill<4>(cudaPol,ee_collision_buffer,"inds",zs::vec<int,4>::uniform(-1).template reinterpret_bits<T>());
//         TILEVEC_OPS::fill(cudaPol,ee_collision_buffer,"inverted",reinterpret_bits<T>((int)0));
//         // TILEVEC_OPS::fill(cudaPol,ee_collision_buffer,"abary",(T)0.0);
//         // TILEVEC_OPS::fill(cudaPol,ee_collision_buffer,"bbary",(T)0.0);

//         if(!verts.hasProperty("active") || verts.getChannelSize("active") != 1)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid verts' \"active\" channel");
//         if(!verts.hasProperty(xtag) || verts.getChannelSize(xtag) != 3)
//             throw std::runtime_error("do_edge_edge_collision_detection::invalid verts' \"xtag\" channel");

//         cudaPol(zs::range(lines.size()),[in_collisionEps = in_collisionEps,
//                 out_collisionEps = out_collisionEps,
//                 verts = proxy<space>({},verts),xtag = xtag,
//                 points = proxy<space>({},points),
//                 lines = proxy<space>({},lines),
//                 tris = proxy<space>({},tris),
//                 sttemp = proxy<space>({},sttemp),
//                 setemp = proxy<space>({},setemp),
//                 ee_collision_buffer = proxy<space>({},ee_collision_buffer),
//                 seBvh = proxy<space>(seBvh),
//                 thickness = bvh_thickness] ZS_LAMBDA(int sei) mutable {
//                     auto einds = lines.template pack<2>("inds",sei).reinterpret_bits(int_c);
//                     auto id0 = einds[0];
//                     auto id1 = einds[1];
//                     auto is_active0 = verts("active",id0) > 1e-6;
//                     auto is_active1 = verts("active",id1) > 1e-6;

//                     if(!is_active0 || !is_active1){
//                         // printf("skip inactive edge %d\n",sei);
//                         return;
//                     }

//                     auto a0 = verts.template pack<3>(xtag,id0);
//                     auto a1 = verts.template pack<3>(xtag,id1);

//                     auto ac = (a0 + a1) / (T)2.0;
//                     auto bv = bv_t{get_bounding_box(ac - thickness,ac + thickness)};

//                     // int nm_collision_pairs = 0;
//                     int closestEdge = -1;
//                     T closestDistance = 1e8;

//                     zs::vec<T,2> aClosest{};
//                     zs::vec<T,2> bClosest{};
//                     zs::vec<T,3> aClosestPoint{};
//                     zs::vec<T,3> bClosestPoint{};

//                     auto aNrm = setemp.template pack<3>("nrm",sei);

//                     auto process_edge_edge_collision_pairs = [&](int nseI) {
//                         // printf("check edge pairs : %d %d\n",sei,nseI);

//                         zs::vec<T,3> aPoint{};
//                         zs::vec<T,3> bPoint{};
//                         // zs::vec<T,3> bNrm{};
//                         zs::vec<T,2> a{},b{};

//                         auto nedge = lines.pack(dim_c<2>,"inds",nseI).reinterpret_bits(int_c);

//                         if(nedge[0] == id0 || nedge[1] == id0 || nedge[0] == id1 || nedge[1] == id1){
//                             // printf("skip neighbor pairs : %d %d\n",sei,nseI);
//                             return;
//                         }

//                         auto is_active0_nei = verts("active",nedge[0]) > 1e-6;
//                         auto is_active1_nei = verts("active",nedge[1]) > 1e-6;

//                         if(!is_active0_nei || !is_active1_nei){
//                             // printf("skip inactive nedge %d\n",sei);
//                             return;
//                         }


//                         // // the two edges should orient in different directions
//                         auto bNrm = setemp.template pack<3>("nrm",nseI);
//                         auto orient = bNrm.dot(aNrm);
//                         if(orient > 0.2){
//                             // printf("skip pairs : %d %d due to orient problem %f %f %f\n",sei,nseI,(float)orient,(float)bNrm.norm(),(float)aNrm.norm());
//                             return;
//                         }

//                         auto nid0 = nedge[0];
//                         auto nid1 = nedge[1];

//                         auto b0 = verts.template pack<3>(xtag,nid0);
//                         auto b1 = verts.template pack<3>(xtag,nid1);

//                         COLLISION_UTILS::IntersectLineSegments(a0,a1,b0,b1,aPoint,bPoint);
//                         auto distance = (aPoint - bPoint).norm();

//                         if(distance > closestDistance){  
//                             // printf("skip pairs : %d %d due to distance %f %f\n",sei,nseI,(float)distance,(float)closestDistance);
//                             return;
//                         }

//                         zs::vec<T,3> ea = a1 - a0;
//                         zs::vec<T,3> eb = b1 - b0;

//                         a[1] = (aPoint - a0).norm() / ea.norm();
//                         a[0] = (T)1.0 - a[1];

//                         b[1] = (bPoint - b0).norm() / eb.norm();
//                         b[0] = (T)1.0 - b[1];

//                         T skipEps = 1e-4;
//                         if ((a[0] < skipEps) || (a[0] > 1.0 - skipEps)) return;
//                         if ((a[1] < skipEps) || (a[1] > 1.0 - skipEps)) return;
//                         if ((b[0] < skipEps) || (b[0] > 1.0 - skipEps)) return;
//                         if ((b[1] < skipEps) || (b[1] > 1.0 - skipEps)) return;

//                         closestDistance = distance;
//                         closestEdge = nseI;

//                         aClosest = a;
//                         bClosest = b;      
//                         aClosestPoint = aPoint;
//                         bClosestPoint = bPoint;                    
//                 };
//                 seBvh.iter_neighbors(bv,process_edge_edge_collision_pairs);



//                 if(closestEdge == -1) return;

//                 // printf("find closest pairs : %d -> %d\n",sei,closestEdge);


//                 if(closestEdge >= lines.size()){
//                     printf("closestEdge bigger than lines size\n");
//                     return;
//                 }

//                 if(lines.size() != setemp.size()){
//                     printf("lines size and setemp size not match\n");
//                     return;
//                 }
//                 if(!setemp.hasProperty("nrm")){
//                     printf("setemp has no nrm channel");
//                     return;
//                 }

//                 auto innerEdge = lines.pack(dim_c<2>,"inds",closestEdge).reinterpret_bits(int_c);

//                 // return;

//                 // // skip the one-ring neighbor_check
//                 // bool insideOneRing = false;

//                 // for (int j = 0; j < 2; j++)
//                 // {
//                 // pair<int, int> lookup;
//                 // lookup.first = outerEdge[j];
//                 // for (int i = 0; i < 2; i++)
//                 // {
//                 //     lookup.second = innerEdge[i];
//                 //     if (_insideSurfaceVertexOneRing.find(lookup) != _insideSurfaceVertexOneRing.end())
//                 //     insideOneRing = true;
//                 // }
//                 // }
//                 // if (insideOneRing) return;
//                 auto a2b = bClosestPoint - aClosestPoint;
//                 auto bNrm = setemp.template pack<3>("nrm",closestEdge);


//                 // auto avgNrm = (bNrm - aNrm).normalized();
//                 bool is_penertrating = a2b.dot(aNrm) < 0 && a2b.dot(bNrm) > 0;

//                 auto collisionEps = is_penertrating ? in_collisionEps : out_collisionEps;

//                 // then there is edge edge collision
//                 if(closestDistance > collisionEps)  
//                     return;

//                 // if(is_penertrating)
//                 //     printf("find penertrating pair %d %d %d %d\n",einds[0],einds[1],innerEdge[0],innerEdge[1]);


//                 ee_collision_buffer.template tuple<4>("inds",sei) = zs::vec<int,4>(einds[0],einds[1],innerEdge[0],innerEdge[1]).template reinterpret_bits<T>();
//                 auto edgeEdgeCollsionAreas = lines("area",sei) + lines("area",closestEdge);
//                 ee_collision_buffer("area",sei) = edgeEdgeCollsionAreas;

//                 int is_inverted = is_penertrating ? 1 : 0;  
//                 ee_collision_buffer("inverted",sei) = reinterpret_bits<T>(is_inverted);   

//                 ee_collision_buffer.template tuple<4>("bary",sei) = zs::vec<T,4>(aClosest[0],aClosest[1],bClosest[0],bClosest[1]);
                
//                                 // return;

//                 ee_collision_buffer.template tuple<2>("abary",sei) = aClosest;

//                 // ee_collision_buffer("abary",0,sei) = (T)0.0;
//                 // ee_collision_buffer("abary",1,sei) = (T)0.0;
//                 // // return;
//                 ee_collision_buffer.template tuple<2>("bbary",sei) = bClosest;
//                 // ee_collision_buffer("bbary",0,sei) = (T)0.0;
//                 // ee_collision_buffer("bbary",1,sei) = (T)0.0;
//         });
        
// }


template<typename Pol,
    typename PosTileVec,
    typename GradHessianTileVec>
void evaluate_fp_collision_grad_and_hessian(
    Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,
    const zs::Vector<zs::vec<int,4>>& csPT,
    int nm_csPT,
    GradHessianTileVec& gh_buffer,
    T in_collisionEps,T out_collisionEps,
    T collisionStiffness,
    T mu,T lambda) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        gh_buffer.resize(nm_csPT);
        cudaPol(zs::range(nm_csPT),[
            gh_buffer = proxy<space>({},gh_buffer),
            in_collisionEps = in_collisionEps,
            out_collisionEps = out_collisionEps,
            verts = proxy<space>({},verts),
            csPT = proxy<space>(csPT),
            mu = mu,lam = lambda,
            stiffness = collisionStiffness,
            xtag = xtag] ZS_LAMBDA(int ci) mutable {
                auto inds = csPT[ci];
                gh_buffer.tuple(dim_c<4>,"inds",ci) = inds.reinterpret_bits(float_c);
                vec3 cv[4] = {};
                for(int i = 0;i != 4;++i)
                    cv[i] = verts.pack(dim_c<3>,xtag,inds[i]);

                auto ceps = out_collisionEps;
                auto alpha = stiffness;
                auto beta = (T)1.0;

                auto cforce = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                auto K = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps);

                gh_buffer.tuple(dim_c<12>,"grad",ci) = cforce/* + dforce*/;
                gh_buffer.tuple(dim_c<12 * 12>,"H",ci) = K/* + C/dt*/;
                if(isnan(K.norm())){
                    printf("nan cK detected : %d\n",ci);
                }
        });
}

template<typename Pol,
    typename PosTileVec,
    typename FPCollisionBuffer,
    typename GradHessianTileVec>
void evaluate_fp_collision_grad_and_hessian(
    Pol& cudaPol,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,T dt,
    const FPCollisionBuffer& fp_collision_buffer,// recording all the fp collision pairs
    GradHessianTileVec& gh_buffer,int offset,
    T in_collisionEps,T out_collisionEps,
    T collisionStiffness,
    T mu,T lambda,T kd_theta) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
 
        int start = offset;
        int fp_size = fp_collision_buffer.size(); 

        TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"H",(T)0.0,start,fp_size);
        TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"grad",(T)0.0,start,fp_size); 

        // std::cout << "inds size compair : " << fp_collision_buffer.getChannelSize("inds") << "\t" << gh_buffer.getChannelSize("inds") << std::endl;

        TILEVEC_OPS::copy(cudaPol,fp_collision_buffer,"inds",gh_buffer,"inds",start); 

        cudaPol(zs::range(fp_size),
            [verts = proxy<space>({},verts),xtag,vtag,dt,kd_theta,
                fp_collision_buffer = proxy<space>({},fp_collision_buffer),
                gh_buffer = proxy<space>({},gh_buffer),
                in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                stiffness = collisionStiffness,
                mu = mu,lam = lambda,start = start] ZS_LAMBDA(int cpi) mutable {
                auto inds = fp_collision_buffer.template pack<4>("inds",cpi).reinterpret_bits(int_c);
                for(int j = 0;j != 4;++j)
                    if(inds[j] < 0)
                        return;
                vec3 cv[4] = {};
                for(int j = 0;j != 4;++j)
                    cv[j] = verts.template pack<3>(xtag,inds[j]);
            
                // auto is_inverted = reinterpret_bits<int>(fp_collision_buffer("inverted",cpi));
                // auto ceps = is_inverted ? in_collisionEps : out_collisionEps;

                auto ceps = out_collisionEps;
                // ceps += (T)1e-2 * ceps;

                auto alpha = stiffness;
                auto beta = fp_collision_buffer("area",cpi);
          
                auto cforce = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                auto K = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps);

                // gh_buffer.template tuple<12>("grad",cpi + start) = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps);
                // gh_buffer.template tuple<12*12>("H",cpi + start) =  alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps); 
                
                
                // adding rayleigh damping term
                // vec3 v0[4] = {verts.pack(dim_c<3>,vtag, inds[0]),
                // verts.pack(dim_c<3>,vtag, inds[1]),
                // verts.pack(dim_c<3>,vtag, inds[2]),
                // verts.pack(dim_c<3>,vtag, inds[3])}; 
                // auto vel = COLLISION_UTILS::flatten(v0); 

                // auto C = K * kd_theta;
                // auto dforce = -C * vel;
                gh_buffer.template tuple<12>("grad",cpi + start) = cforce/* + dforce*/;
                gh_buffer.template tuple<12*12>("H",cpi + start) = K/* + C/dt*/;
                if(isnan(K.norm())){
                    printf("nan cK detected : %d\n",cpi);
                }
        });
}

// TODO: add damping collision term
template<typename Pol,
    typename TetTileVec,
    typename PosTileVec,
    typename SurfTriTileVec,
    typename FPCollisionBuffer,
    typename GradHessianTileVec>
void evaluate_kinematic_fp_collision_grad_and_hessian(
    Pol& cudaPol,
    const TetTileVec& eles,
    const PosTileVec& verts,const zs::SmallString& xtag,const zs::SmallString& vtag,T dt,
    const SurfTriTileVec& tris,
    const PosTileVec& kverts,
    const FPCollisionBuffer& kc_buffer,
    GradHessianTileVec& gh_buffer,int offset,
    T in_collisionEps,T out_collisionEps,
    T collisionStiffness,
    T mu,T lambda,T kd_theta) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        int start = offset;
        int fp_size = kc_buffer.size();

        // TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"H",(T)0.0,start,fp_size);
        // TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"grad",(T)0.0,start,fp_size);

        // get only the dynamic object's dofs
        // TILEVEC_OPS::copy(cudaPol,kc_buffer,"inds",gh_buffer,"inds",start);
        // cudaPol(zs::range(fp_size),
        //     [gh_buffer = proxy<space>({},gh_buffer),start = start] ZS_LAMBDA(int fpi) mutable {
        //         gh_buffer("inds",0,start + fpi) = gh_buffer("inds",1,start + fpi);
        //         auto tmp = gh_buffer("inds",2,start + fpi);
        //         gh_buffer("inds",2,start + fpi) = gh_buffer("inds",3,start + fpi);
        //         gh_buffer("inds",3,start + fpi) = tmp;
        // });


        cudaPol(zs::range(fp_size),
            [verts = proxy<space>({},verts),xtag,vtag,dt,kd_theta,
                eles = proxy<space>({},eles),
                tris = proxy<space>({},tris),
                kverts = proxy<space>({},kverts),
                kc_buffer = proxy<space>({},kc_buffer),
                gh_buffer = proxy<space>({},gh_buffer),start,
                in_collisionEps = in_collisionEps,
                out_collisionEps = out_collisionEps,
                stiffness = collisionStiffness,
                mu = mu,lam = lambda] ZS_LAMBDA(int cpi) mutable {
                auto inds = kc_buffer.pack(dim_c<2>,"inds",cpi).reinterpret_bits(int_c);
                // auto oinds = kc_buffer.pack(dim_c<4>,"inds",cpi).reinterpret_bits(int_c);
                for(int i = 0;i != 2;++i)
                    if(inds[i] < 0)
                        return;
                vec3 cv[4] = {};
                cv[0] = kverts.pack(dim_c<3>,"x",inds[0]);
                auto tri = tris.pack(dim_c<3>,"inds",inds[1]).reinterpret_bits(int_c);
                for(int j = 1;j != 4;++j)
                    cv[j] = verts.template pack<3>(xtag,tri[j-1]);
                
                // vec3 cvel[4] = {};
                // cvel[0] = vec3::zeros();
                // for(int j = 1;j != 4;++j)
                //     cvel[j] = verts.template pack<3>(vel_tag,inds[j]);

                // auto is_inverted = reinterpret_bits<int>(kc_buffer("inverted",cpi));
                auto average_thickness = (T)0.0;
                if(verts.hasProperty("k_thickness")){
                    // average_thickness = (T)0.0;
                    for(int i = 0;i != 3;++i)
                        average_thickness += verts("k_thickness",tri[i])/(T)3.0;
                }


                auto ceps = out_collisionEps * ((T)1.0 + average_thickness);
                auto alpha = stiffness;
                auto beta = kc_buffer("area",cpi);

                // change the 

                auto cgrad = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps,true);
                auto cH = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps,true);

                auto ei = reinterpret_bits<int>(tris("ft_inds",inds[1]));
                // auto cp = gh_buffer.pack(dim_c<2>,"inds",ei).reinterpret_bits(int_c);
                // auto pidx = cp[0];
                // auto tri = tris.pack(dim_c<3>,"inds",cp[1]).reinterpret_bits(int_c);
                auto tet = eles.pack(dim_c<4>,"inds",ei).reinterpret_bits(int_c);
                auto inds_reorder = zs::vec<int,3>::zeros();
                for(int i = 0;i != 3;++i){
                    auto idx = tri[i];
                    for(int j = 0;j != 4;++j)
                        if(idx == tet[j])
                            inds_reorder[i] = j;
                }

                vec3 v0[4] = {zs::vec<T,3>::zeros(),
                verts.pack(dim_c<3>,vtag, tri[0]),
                verts.pack(dim_c<3>,vtag, tri[1]),
                verts.pack(dim_c<3>,vtag, tri[2])}; 
                auto vel = COLLISION_UTILS::flatten(v0);

                auto C = cH * kd_theta;
                auto dforce = -C * vel;

                cgrad += dforce;
                cH += C/dt;

                // gh_buffer.template tuple<12>("grad",cpi + start) = cforce + dforce;
                // gh_buffer.template tuple<12*12>("H",cpi + start) = K + C/dt;

                for(int i = 3;i != 12;++i){
                    int d0 = i % 3;
                    int row = inds_reorder[i/3 - 1]*3 + d0;
                    atomic_add(exec_cuda,&gh_buffer("grad",row,ei),cgrad[i]);
                    for(int j = 3;j != 12;++j){
                        int d1 = j % 3;
                        int col = inds_reorder[j/3 - 1]*3 + d1;
                        if(row >= 12 || col >= 12){
                            printf("invalid row = %d and col = %d %d %d detected %d %d %d\n",row,col,i/3,j/3,
                                inds_reorder[0],
                                inds_reorder[1],
                                inds_reorder[2]);
                        }
                        atomic_add(exec_cuda,&gh_buffer("H",row*12 + col,ei),cH(i,j));
                    }                    
                }
                // for(int i = 1;i != 4;++i){ 
                //     auto idx = inds[i];
                //     for(int j = 0;j != 4;++j){
                //         if(idx == tet[j]) {
                //             for(int d = 0;d != 3;++d)
                //                 atomic_add(exec_cuda,&gh_buffer("grad",j*3 + d,ei),cgrad[i * 3 + d]);
                //         }
                //     }
                    
                //     gh_buffer("grad",i,cpi + start) = cgrad[i];

                // }
                // for(int i = 3;i != 12;++i)
                //     for(int j = 3;j != 12;++j)
                //         gh_buffer("H",i * 12 + j,cpi + start) = cH(i,j);
                // auto test_ind = gh_buffer.pack(dim_c<4>,"inds",start + cpi).reinterpret_bits(int_c);
                // auto cgrad_norm = cgrad.norm();
                // auto cH_norm = cH.norm();
                // printf("find_kinematic_collision[%d %d %d %d] : %f %f\n",inds[0],inds[1],inds[2],inds[3],(float)alpha,(float)beta);
        });
}

// template<typename Pol,
//     typename PosTileVec,
//     typename EECollisionBuffer,
//     typename GradHessianTileVec>
// void evaluate_ee_collision_grad_and_hessian(Pol& cudaPol,
//     const PosTileVec& verts,const zs::SmallString& xtag,
//     const EECollisionBuffer& ee_collision_buffer,
//     GradHessianTileVec& gh_buffer,int offset,
//     T in_collisionEps,T out_collisionEps,
//     T collisionStiffness,
//     T mu,T lambda) {
//         using namespace zs;
//         constexpr auto space = execspace_e::cuda;

//         int start = offset;
//         int ee_size = ee_collision_buffer.size();

//         TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"H",(T)0.0,start,ee_size);
//         TILEVEC_OPS::fill_range(cudaPol,gh_buffer,"grad",(T)0.0,start,ee_size);
//         TILEVEC_OPS::copy(cudaPol,ee_collision_buffer,"inds",gh_buffer,"inds",start);

//         cudaPol(zs::range(ee_size),[
//             verts = proxy<space>({},verts),xtag,
//             in_collisionEps,out_collisionEps,
//             ee_collision_buffer = proxy<space>({},ee_collision_buffer),
//             gh_buffer = proxy<space>({},gh_buffer),
//             start = start,
//             stiffness = collisionStiffness,mu = mu,lam = lambda] ZS_LAMBDA(int eei) mutable {
//                 auto inds = ee_collision_buffer.template pack<4>("inds",eei).reinterpret_bits(int_c);
//                 for(int i = 0;i != 4;++i)
//                     if(inds[i] < 0)
//                         return;
//                 for(int j = 0;j != 4;++j){
//                     auto active = verts("active",inds[j]);
//                     if(active < 1e-6)
//                         return;
//                 }  
//                 vec3 cv[4] = {};
//                 for(int j = 0;j != 4;++j)
//                     cv[j] = verts.template pack<3>(xtag,inds[j]);       

//                 auto is_inverted = reinterpret_bits<int>(ee_collision_buffer("inverted",eei));
//                 auto ceps = is_inverted ? in_collisionEps : out_collisionEps;

//                 auto alpha = stiffness;
//                 auto beta = ee_collision_buffer("area",eei);

//                 auto a = ee_collision_buffer.template pack<2>("abary",eei);
//                 auto b = ee_collision_buffer.template pack<2>("bbary",eei);

//                 const T tooSmall = (T)1e-6;

//                 if(is_inverted) {
//                     gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_SQRT_COLLISION::gradientNegated(cv,a,b,mu,lam,ceps,tooSmall);
//                     gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_SQRT_COLLISION::hessianNegated(cv,a,b,mu,lam,ceps,tooSmall);
//                     // gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_COLLISION::gradientNegated(cv,a,b,mu,lam,ceps);
//                     // gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_COLLISION::hessianNegated(cv,a,b,mu,lam,ceps);
//                 }else {
//                     gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_SQRT_COLLISION::gradient(cv,a,b,mu,lam,ceps,tooSmall);
//                     gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_SQRT_COLLISION::hessian(cv,a,b,mu,lam,ceps,tooSmall);  
//                     // gh_buffer.template tuple<12>("grad",eei + start) = -alpha * beta * EDGE_EDGE_COLLISION::gradient(cv,a,b,mu,lam,ceps);
//                     // gh_buffer.template tuple<12*12>("H",eei + start) = alpha * beta * EDGE_EDGE_COLLISION::hessian(cv,a,b,mu,lam,ceps);                  
//                 }
//         });
//     }


};

};