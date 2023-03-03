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
// #include "collision_energy/edge_edge_sqrt_collision.hpp"
// #include "collision_energy/edge_edge_collision.hpp"

#include "collision_energy/evaluate_collision.hpp"

namespace zeno {

#define MAX_FP_COLLISION_PAIRS 4

struct FleshDynamicStepping : INode {

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

    // currently only backward euler integrator is supported
    // topology evaluation should be called before applying this node
    struct FEMDynamicSteppingSystem {
        template <typename Model>
        void computeCollisionEnergy(zs::CudaExecutionPolicy& cudaPol,const Model& model,
                dtiles_t& vtemp,
                dtiles_t& etemp,
                dtiles_t& sttemp,
                dtiles_t& setemp,
                dtiles_t& ee_buffer,
                dtiles_t& fe_buffer) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            T lambda = model.lam;
            T mu = model.mu;
        }


        void findInversion(zs::CudaExecutionPolicy& cudaPol,dtiles_t& vtemp,dtiles_t& etemp) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            TILEVEC_OPS::fill(cudaPol,vtemp,"is_inverted",reinterpret_bits<T>((int)0));  
            TILEVEC_OPS::fill(cudaPol,etemp,"is_inverted",reinterpret_bits<T>((int)0));  
            cudaPol(zs::range(eles.size()),
                [vtemp = proxy<space>({},vtemp),
                        quads = proxy<space>({},eles),
                        etemp = proxy<space>({},etemp)] ZS_LAMBDA(int ei) mutable {
                    auto DmInv = quads.template pack<3,3>("IB",ei);
                    auto inds = quads.template pack<4>("inds",ei).reinterpret_bits(int_c);
                    vec3 x1[4] = {vtemp.template pack<3>("xn", inds[0]),
                            vtemp.template pack<3>("xn", inds[1]),
                            vtemp.template pack<3>("xn", inds[2]),
                            vtemp.template pack<3>("xn", inds[3])};   

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
                        // for(int i = 0;i < 4;++i)
                        //     vtemp("is_inverted",inds[i]) = reinterpret_bits<T>((int)1);   
                        etemp("is_inverted",ei) = reinterpret_bits<T>((int)1);   
                    }else {
                        etemp("is_inverted",ei) = reinterpret_bits<T>((int)0);   
                    }               
            });
            cudaPol(zs::range(eles.size()),
                [vtemp = proxy<space>({},vtemp),
                        quads = proxy<space>({},eles),
                        etemp = proxy<space>({},etemp)] ZS_LAMBDA(int ei) mutable {
                auto inds = quads.template pack<4>("inds",ei).reinterpret_bits(int_c);
                auto is_inverted = reinterpret_bits<int>(etemp("is_inverted",ei));  
                if(is_inverted)
                    for(int i = 0;i != 4;++i){
                        vtemp("is_inverted",inds[i]) = reinterpret_bits<T>((int)1);     
                    }       
            });
        }

        // template <typename Model>
        // void computeKinematicCollisionGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,const Model& model,
        //     dtiles_t& vtemp,
        //     dtiles_t& sptemp,
        //     dtiles_t& sttemp,
        //     const dtiles_t& kvtemp,
        //     const dtiles_t& kltemp,
        //     const dtiles_t& kttemp,
        //     dtiles_t& kc_buffer,
        //     dtiles_t& gh_buffer,
        //     bool neglect_inverted = true) {
        //         using namespace zs;
        //         constexpr auto space = execspace_e::cuda;

        //         int offset = eles.size() + b_verts.size() + points.size() * MAX_FP_COLLISION_PAIRS;
        //         T lambda = model.lam;
        //         T mu = model.mu;

        //         // COLLISION_UTILS::do_kinematic_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
        //         //     vtemp,"xn",
        //         //     points,
        //         //     lines,
        //         //     tris,
        //         //     kvtemp,
        //         //     kltemp,
        //         //     kttemp,
        //         //     kc_buffer,
        //         //     in_collisionEps,out_collisionEps);
                
        //         // COLLISION_UTILS::evaluate_kinematic_fp_collision_grad_and_hessian(cudaPol,
        //         //     vtemp,"xn",
        //         //     kvtemp,
        //         //     kc_buffer,
        //         //     gh_buffer,offset,
        //         //     in_collisionEps,out_collisionEps,
        //         //     (T)collisionStiffness,
        //         //     (T)mu,(T)lambda);    

        //         if(neglect_inverted) {
        //             cudaPol(zs::range(kc_buffer.size()),
        //                 [gh_buffer = proxy<space>({},gh_buffer),
        //                         vtemp = proxy<space>({},vtemp),
        //                         kc_buffer = proxy<space>({},kc_buffer),
        //                         offset] ZS_LAMBDA(int cpi) {
        //                     auto inds = gh_buffer.template pack<4>("inds",cpi + offset).reinterpret_bits(int_c);
        //                     for(int i = 0;i != 4;++i)
        //                         if(inds[i] < 0)
        //                             return;
                            
        //                     bool is_inverted = false;
        //                     int is_fp = reinterpret_bits<int>(kc_buffer("is_fp",cpi));
        //                     int check_len = is_fp > 0 ? 3 : 1;
        //                     for(int i = 0;i != check_len;++i){
        //                         auto vi = inds[i];
        //                         auto is_vertex_inverted = reinterpret_bits<int>(vtemp("is_inverted",vi));
        //                         if(is_vertex_inverted)
        //                             is_inverted = true;
        //                     }
 
        //                     if(is_inverted){
        //                         gh_buffer.template tuple<12*12>("H",cpi + offset) = zs::vec<T,12,12>::zeros();
        //                         gh_buffer.template tuple<12>("grad",cpi + offset) = zs::vec<T,12>::zeros();
        //                     }
        //             });    
        //         }                            
        // }

        template <typename Model>
        void computeCollisionGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,const Model& model,
                            dtiles_t& vtemp,
                            dtiles_t& etemp,
                            dtiles_t& sttemp,
                            dtiles_t& setemp,
                            // dtiles_t& ee_buffer,
                            dtiles_t& fp_buffer,
                            dtiles_t& kverts,
                            dtiles_t& kc_buffer,
                            dtiles_t& gh_buffer,
                            T kd_theta = (T)0.0,
                            bool explicit_collision = false,
                            bool neglect_inverted = true) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            int offset = eles.size();

            T lambda = model.lam;
            T mu = model.mu; 

            // auto stBvh = bvh_t{};
            // auto bvs = retrieve_bounding_volumes(cudaPol,vtemp,tris,wrapv<3>{},(T)0.0,"xn");
            // stBvh.build(cudaPol,bvs);
            // auto avgl = compute_average_edge_length(cudaPol,vtemp,"xn",tris);
            // auto bvh_thickness = 5 * avgl;            
            // if(!calculate_facet_normal(cudaPol,vtemp,"xn",tris,sttemp,"nrm")){
            //     throw std::runtime_error("fail updating facet normal");
            // }       
            // if(!COLLISION_UTILS::calculate_cell_bisector_normal(cudaPol,
            //     vtemp,"xn",
            //     lines,
            //     tris,
            //     sttemp,"nrm",
            //     setemp,"nrm")){
            //         throw std::runtime_error("fail calculate cell bisector normal");
            // }    


            COLLISION_UTILS::do_facet_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
                vtemp,"xn",
                points,
                lines,
                tris,
                sttemp,
                setemp,
                fp_buffer,
                in_collisionEps,out_collisionEps);

            COLLISION_UTILS::evaluate_fp_collision_grad_and_hessian(cudaPol,
                vtemp,"xn","vn",dt,
                fp_buffer,
                gh_buffer,offset,
                in_collisionEps,out_collisionEps,
                (T)collisionStiffness,
                (T)mu,(T)lambda,(T)kd_theta);
            


            COLLISION_UTILS::do_kinematic_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
                vtemp,"xn",
                points,
                lines,
                tris,
                setemp,
                sttemp,
                kverts,
                kc_buffer,
                (T)kine_in_collisionEps,(T)kine_out_collisionEps,false);

            offset = 0;

            COLLISION_UTILS::evaluate_kinematic_fp_collision_grad_and_hessian(cudaPol,
                eles,
                vtemp,"xn","vn",dt,
                tris,
                kverts,
                kc_buffer,
                gh_buffer,offset,
                (T)kine_in_collisionEps,(T)kine_out_collisionEps,
                (T)kineCollisionStiffness,
                (T)mu,(T)lambda,(T)kd_theta);


            // adding collision damping on self collision
            // int offset = eles.size() + b_verts.size();
            // cudaPol(zs::range(fp_buffer.size() + kc_buffer.size()),
            //     [vtemp = proxy<space>({},vtemp),
            //         gh_buffer = proxy<space>({},gh_buffer),offset,kd_theta] ZS_LAMBDA(int ci) mutable {
            //     auto inds = gh_buffer.pack(dim_c<4>,"inds",ci).reinterpret_bits(int_c);
            //     for(int i = 0;i != 4;++i)
            //         if(inds[i] < 0)
            //             return;
            //     vec3 vs[4] = {};
            //     for(int i = 0;i = 4;++i)
            //         vs[i] = vtemp.pack(dim_c<3>,"vn",inds[i]);
            //     auto H = gh_buffer.pack(dim_c<12*12>,"H",ci);
            //     gh_buffer.tuple(dim_c<12*12>,"H",ci) = H;
            // });
        

        }


        template <typename ElasticModel,typename AnisoElasticModel>
        void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                            const ElasticModel& model,
                            const AnisoElasticModel& amodel,
                            const dtiles_t& vtemp,
                            const dtiles_t& etemp,
                            dtiles_t& gh_buffer,
                            T kd_alpha = (T)0.0,
                            T kd_beta = (T)0.0) {        
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            int offset = 0;
            TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",gh_buffer,"inds",offset);   
            // eval the inertia term gradient
            cudaPol(zs::range(eles.size()),[dt2 = dt2,
                        verts = proxy<space>({},verts),
                        eles = proxy<space>({},eles),
                        vtemp = proxy<space>({},vtemp),
                        gh_buffer = proxy<space>({},gh_buffer),
                        dt = dt,offset = offset] ZS_LAMBDA(int ei) mutable {
                auto m = eles("m",ei)/(T)4.0;
                auto inds = eles.pack(dim_c<4>,"inds",ei).reinterpret_bits(int_c);
                auto pgrad = zs::vec<T,12>::zeros();
                // auto H  = zs::vec<T,12,12>::zeros();
                // if(eles.hasProperty("dt")) {
                //     dt2 = eles("dt",ei) * eles("dt",ei);
                // }

                auto inertia = (T)1.0;
                if(eles.hasProperty("inertia"))
                    inertia = eles("inertia",ei);
                for(int i = 0;i != 4;++i){
                    auto x1 = vtemp.pack(dim_c<3>,"xn",inds[i]);
                    auto x0 = vtemp.pack(dim_c<3>,"xp",inds[i]);
                    auto v0 = vtemp.pack(dim_c<3>,"vp",inds[i]);

                    auto alpha = inertia * m/dt2;
                    auto nodal_pgrad = -alpha * (x1 - x0 - v0 * dt);
                    for(int d = 0;d != 3;++d){
                        auto idx = i * 3 + d;
                        gh_buffer("grad",idx,ei) = nodal_pgrad[d];
                        gh_buffer("H",idx*12 + idx,ei + offset) = alpha;
                    }
                    
                }
                // gh_buffer.tuple(dim_c<12>,"grad",ei + offset) = pgrad;
                // gh_buffer.template tuple<12*12>("H",ei + offset) = H;
            });


            cudaPol(zs::range(eles.size()), [dt = dt,dt2 = dt2,aniso_strength = aniso_strength,
                            verts = proxy<space>({},verts),
                            vtemp = proxy<space>({}, vtemp),
                            etemp = proxy<space>({}, etemp),
                            gh_buffer = proxy<space>({},gh_buffer),
                            eles = proxy<space>({}, eles),
                            kd_alpha = kd_alpha,kd_beta = kd_beta,
                            model = model,amodel = amodel, volf = volf,offset = offset] ZS_LAMBDA (int ei) mutable {
                auto DmInv = eles.pack(dim_c<3,3>,"IB",ei);
                auto dFdX = dFdXMatrix(DmInv);
                auto inds = eles.pack(dim_c<4>,"inds",ei).reinterpret_bits(int_c);
                vec3 x1[4] = {vtemp.pack(dim_c<3>,"xn", inds[0]),
                                vtemp.pack(dim_c<3>,"xn", inds[1]),
                                vtemp.pack(dim_c<3>,"xn", inds[2]),
                                vtemp.pack(dim_c<3>,"xn", inds[3])};


                mat3 FAct{};
                {
                    auto x1x0 = x1[1] - x1[0];
                    auto x2x0 = x1[2] - x1[0];
                    auto x3x0 = x1[3] - x1[0];
                    auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                                    x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                    FAct = Ds * DmInv;
                    FAct = FAct * etemp.template pack<3,3>("ActInv",ei);
                } 
                auto dFActdF = dFAdF(etemp.template pack<3,3>("ActInv",ei));

                // add the force term in gradient
                if(eles.hasProperty("mu") && eles.hasProperty("lam")) {
                    model.mu = eles("mu",ei);
                    model.lam = eles("lam",ei);
                }
                auto P = model.first_piola(FAct);
                auto vole = eles("vol", ei);
                auto vecP = flatten(P);
                vecP = dFActdF.transpose() * vecP;
                auto dFdXT = dFdX.transpose();
                auto vf = -vole * (dFdXT * vecP);     

                auto mg = volf * vole / (T)4.0;
                for(int i = 0;i != 4;++i)
                    for(int d = 0;d !=3 ;++d){
                        vf[i*3 + d] += mg[d];
                    }


                // assemble element-wise hessian matrix
                auto Hq = model.first_piola_derivative(FAct, true_c);
                auto dFdAct_dFdX = dFActdF * dFdX; 
                // add inertia hessian term
                auto H = dFdAct_dFdX.transpose() * Hq * dFdAct_dFdX * vole;

                if(eles.hasProperty("Muscle_ID") && (int)eles("Muscle_ID",ei) >= 0) {
                    auto fiber = eles.pack(dim_c<3>,"fiber",ei);
                    if(zs::abs(fiber.norm() - 1.0) < 1e-3) {
                        fiber /= fiber.norm();
                        // if(eles.hasProperty("mu")) {
                        //     amodel.mu = eles("mu",ei);
                        //     // amodel.lam = eles("lam",ei);
                            
                        // }
                        auto aP = amodel.do_first_piola(FAct,fiber);
                        auto vecAP = flatten(P);
                        vecAP = dFActdF.transpose() * vecP;
                        vf -= vole  * dFdXT * vecAP *aniso_strength;

                        auto aHq = amodel.do_first_piola_derivative(FAct,fiber);
                        H += dFdAct_dFdX.transpose() * aHq * dFdAct_dFdX * vole * aniso_strength;
                        // if((int)eles("Muscle_ID",ei) == 0){
                        //     printf("fiber : %f %f %f,Fa = %f,aP = %f,aHq = %f,H = %f\n",fiber[0],fiber[1],fiber[2],(float)FAct.norm(),(float)aP.norm(),(float)aHq.norm(),(float)H.norm());
                        // }
                    }
                }


                // adding rayleigh damping term
                vec3 v0[4] = {vtemp.pack(dim_c<3>,"vn", inds[0]),
                vtemp.pack(dim_c<3>,"vn", inds[1]),
                vtemp.pack(dim_c<3>,"vn", inds[2]),
                vtemp.pack(dim_c<3>,"vn", inds[3])}; 

                auto inertia = (T)1.0;
                if(eles.hasProperty("inertia"))
                    inertia = eles("inertia",ei);

                auto vel = COLLISION_UTILS::flatten(v0); 
                auto m = eles("m",ei)/(T)4.0;
                auto C = kd_beta * H + kd_alpha * inertia * m * zs::vec<T,12,12>::identity();
                auto rdamping = C * vel;  

                gh_buffer.tuple(dim_c<12>,"grad",ei + offset) = gh_buffer.pack(dim_c<12>,"grad",ei + offset) + vf - rdamping; 
                // gh_buffer.tuple(dim_c<12>,"grad",ei + offset) = gh_buffer.pack(dim_c<12>,"grad",ei + offset) - rdamping; 
                // H += kd_beta*H/dt;

                gh_buffer.template tuple<12*12>("H",ei + offset) = gh_buffer.template pack<12,12>("H",ei + offset) + H + C/dt;
            });
        // Bone Driven Potential Energy
            // T lambda = model.lam;
            // T mu = model.mu;

            auto nmEmbedVerts = b_verts.size();

            // TILEVEC_OPS::fill_range<4>(cudaPol,gh_buffer,"inds",zs::vec<int,4>::uniform(-1).reinterpret_bits(float_c),eles.size() + offset,b_verts.size());
            // TILEVEC_OPS::fill_range<3>(cudaPol,gh_buffer,"grad",zs::vec<T,3>::zeros(),eles.size() + offset,b_verts.size());
            // TILEVEC_OPS::fill_range<144>(cudaPol,gh_buffer,"H",zs::vec<T,144>::zeros(),eles.size() + offset,b_verts.size());

            // we should neglect the inverted element
            // std::cout << "nmEmbedVerts : " << nmEmbedVerts << std::endl;
            // std::cout << "bcwsize :  " << b_bcws.size() << std::endl;
            // return;
            cudaPol(zs::range(nmEmbedVerts), [
                    gh_buffer = proxy<space>({},gh_buffer),model = model,
                    bcws = proxy<space>({},b_bcws),b_verts = proxy<space>(b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    eles = proxy<space>({},eles),bone_driven_weight = bone_driven_weight,offset = offset] ZS_LAMBDA(int vi) mutable {
                        auto ei = reinterpret_bits<int>(bcws("inds",vi));
 
                        if(ei < 0){

                            return;
                        }
                        // if(ei >= etemp.size()){
                        //     printf("ei too big for etemp\n");
                        //     return;
                        // }
                        // auto is_inverted = reinterpret_bits<int>(etemp("is_inverted",ei));
                        // if(is_inverted){
                        //     if(vi == 0)
                        //         printf("inverted tet\n");
                        //     return;
                        // }

                        // auto FatID = eles("FatID",ei);
                        // if(FatID > 0)
                        //     return;

                        auto lambda = model.lam;
                        auto mu = model.mu;
                        // if(eles.hasProperty("mu") && eles.hasProperty("lam")) {
                        //     mu = eles("mu",ei);
                        //     lambda = eles("lam",ei);
                        // }

                        auto inds = eles.pack(dim_c<4>,"inds",ei).reinterpret_bits(int_c);
                        // gh_buffer.tuple(dim_c<4>,"inds",vi + offset + eles.size()) = eles.pack(dim_c<4>,"inds",ei);
                        auto w = bcws.pack(dim_c<4>,"w",vi);
                        if(w[0] < 1e-4 || w[1] < 1e-4 || w[2] < 1e-4 || w[3] < 1e-4){
                            if(vi == 0)
                                printf("boundary tet\n");
                            return;
                        }
                        auto tpos = vec3::zeros();
                        for(int i = 0;i != 4;++i)
                            tpos += w[i] * vtemp.pack(dim_c<3>,"xn",inds[i]);
                        // auto pdiff = tpos - b_verts.pack<3>("x",vi);
                        auto pdiff = tpos - b_verts[vi];

                        T stiffness = 2.0066 * mu + 1.0122 * lambda;

                        zs::vec<T,12> elm_grad{};
                        // auto elm_H = zs::vec<T,12,12>::zeros();

                        for(size_t i = 0;i != 4;++i){
                            auto tmp = pdiff * (-stiffness *  bcws("strength",vi) * bcws("cnorm",vi) * bone_driven_weight * w[i] * eles("vol",ei)) * eles("bdw",ei); 
                            // if(vi == 0 && i == 0) {
                            //     printf("check: %f %f %f\n",(float)tmp[0],(float)tmp[1],(float)tmp[2]);
                            // }
                            for(size_t d = 0;d != 3;++d){
                                atomic_add(exec_cuda,&gh_buffer("grad",i*3 + d,ei),tmp[d]);
                                // elm_grad[i*3 + d] = tmp[d];
                                // atomic_add(exec_cuda,&gh_buffer("grad",i * 3 + d,ei),tmp[d]);
                            }
                        }
                        for(int i = 0;i != 4;++i)
                            for(int j = 0;j != 4;++j){
                                T alpha = stiffness * bone_driven_weight * w[i] * w[j] * bcws("strength",vi) * bcws("cnorm",vi) * eles("vol",ei) * eles("bdw",ei);
                                for(int d = 0;d != 3;++d){
                                    // elm_H(i*3 + d,j*3 + d) = alpha;
                                    atomic_add(exec_cuda,&gh_buffer("H",(i*3 + d)*12 + j*3 + d,ei),alpha);
                                }
                            }
                        
                        // for(int i = 0;i != 12;++i){
                            // atomic_add(exec_cuda,&gh_buffer("grad",i,ei),elm_grad[i]);
                            // for(int j = 0;j != 12;++j)
                            //     atomic_add(exec_cuda,&gh_buffer("H",i*12 + j,ei),elm_H(i,j));
                        // }
                        // gh_buffer.tuple(dim_c<12>,"grad",vi + eles.size() + offset) = elm_grad;
                        // gh_buffer.tuple(dim_c<12*12>,"H",vi + eles.size() + offset) = elm_H;
            });

            // cudaPol(zs::range(eles.size()), [gh_buffer = proxy<space>({},gh_buffer)] ZS_LAMBDA (int ei) mutable {
            //     auto H = gh_buffer.template pack<12,12>("H",ei);
            //     make_pd(H);
            //     gh_buffer.template tuple<12*12>("H",ei) = H;
            // });

        }

        FEMDynamicSteppingSystem(const tiles_t &verts, const tiles_t &eles,
                const tiles_t& points,const tiles_t& lines,const tiles_t& tris,
                T in_collisionEps,T out_collisionEps,
                const tiles_t &b_bcws, const zs::Vector<zs::vec<T,3>>& b_verts,T bone_driven_weight,
                const vec3& volf,const T& _dt,const T& collisionStiffness,
                const T& kine_in_collisionEps,const T& kine_out_collisionEps,
                const T& kineCollisionStiffness,const T& aniso_strength)
            : verts{verts}, eles{eles},points{points}, lines{lines}, tris{tris},
                    in_collisionEps{in_collisionEps},out_collisionEps{out_collisionEps},
                    b_bcws{b_bcws}, b_verts{b_verts}, bone_driven_weight{bone_driven_weight},
                    volf{volf},
                    kine_in_collisionEps{kine_in_collisionEps},kine_out_collisionEps{kine_out_collisionEps},
                    kineCollisionStiffness{kineCollisionStiffness},aniso_strength{aniso_strength},
                    dt{_dt}, dt2{_dt * _dt},collisionStiffness{collisionStiffness},use_edge_edge_collision{true}, use_vertex_facet_collision{true} {}

        const tiles_t &verts;
        const tiles_t &eles;
        const tiles_t &points;
        const tiles_t &lines;
        const tiles_t &tris;
        const tiles_t &b_bcws;  // the barycentric interpolation of embeded bones 
        const zs::Vector<zs::vec<T,3>> &b_verts; // the position of embeded bones

        T bone_driven_weight;
        vec3 volf;
        T dt;
        T dt2;
        T in_collisionEps;
        T out_collisionEps;

        T collisionStiffness;

        bool bvh_initialized;
        bool use_edge_edge_collision;
        bool use_vertex_facet_collision;

        T kine_in_collisionEps;
        T kine_out_collisionEps;
        T kineCollisionStiffness;

        T aniso_strength;

        // int default_muscle_id;
        // zs::vec<T,3> default_muscle_dir;
        // T default_act;

        // T inset;
        // T outset;
    };




    void apply() override {
        using namespace zs;
        auto zsparticles = get_input<ZenoParticles>("ZSParticles");
        auto gravity = zeno::vec<3,T>(0);
        if(has_input("gravity"))
            gravity = get_input2<zeno::vec<3,T>>("gravity");
        T armijo = (T)1e-4;
        T wolfe = (T)0.9;
        // T cg_res = (T)0.01;
        // T cg_res = (T)0.0001;
        T cg_res = get_param<float>("cg_res");
        T btl_res = (T)0.1;
        auto models = zsparticles->getModel();
        auto& verts = zsparticles->getParticles();
        auto& eles = zsparticles->getQuadraturePoints();

        // zs::Vector<vec3>(MAX_VERTS)
        // TileVec("pos","tag","deleted","")

        if(eles.getChannelSize("inds") != 4)
            throw std::runtime_error("the input zsparticles is not a tetrahedra mesh");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
            throw std::runtime_error("the input zsparticles has no surface tris");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
            throw std::runtime_error("the input zsparticles has no surface lines");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) 
            throw std::runtime_error("the input zsparticles has no surface points");

        auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
        auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

        auto zsbones = get_input<PrimitiveObject>("driven_boudary");
        auto driven_tag = get_input2<std::string>("driven_tag");
        auto bone_driven_weight = get_input2<float>("driven_weight");
        auto muscle_id_tag = get_input2<std::string>("muscle_id_tag");



        // auto bone_driven_weight = (T)0.02;

        auto newton_res = get_input2<float>("newton_res");

        auto dt = get_input2<float>("dt");

        auto volf = vec3::from_array(gravity * models.density);

        std::vector<zeno::vec2f> act_;    
        std::size_t nm_acts = 0;

        if(has_input("Acts")) {
            act_ = get_input<zeno::ListObject>("Acts")->getLiterial<zeno::vec2f>();
            nm_acts = act_.size();
        }

        constexpr auto host_space = zs::execspace_e::openmp;
        auto ompExec = zs::omp_exec();
        auto act_buffer = dtiles_t{{{"act",2}},nm_acts,zs::memsrc_e::host};
        ompExec(zs::range(act_buffer.size()),
            [act_buffer = proxy<host_space>({},act_buffer),act_] (int i) mutable {
                act_buffer.tuple(dim_c<2>,"act",i) = vec2(act_[i][0],act_[i][1]);
        });

        act_buffer = act_buffer.clone({zs::memsrc_e::device, 0});

        const auto& zsbones_verts = zsbones->verts;
        zs::Vector<zs::vec<T,3>> bverts{zsbones_verts.size()};
        ompExec(zs::range(zsbones_verts.size()),
            [bverts = proxy<host_space>(bverts),&zsbones_verts] (int i) mutable {
                auto v = zsbones_verts[i];
                bverts[i] = zs::vec<T,3>{v[0],v[1],v[2]};
        });
        bverts = bverts.clone({zs::memsrc_e::device,0});


        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto kverts = typename ZenoParticles::particles_t({
                {"x",3},
                {"xp",3},
                {"area",1}},0,zs::memsrc_e::device,0);
        if(has_input<ZenoParticles>("kinematic_boundary")){
            auto kinematic_boundary = get_input<ZenoParticles>("kinematic_boundary");
            // if (kinematic_boundary.empty())

            // const auto& prim_kverts = kinematic_boundary.verts;
            // auto& prim_kverts_area = kinematic_boundary.attr<float>("area");
            auto& kb_verts = kinematic_boundary->getParticles();

            // auto& kb_tris = kinematic_boundary->getQuadraturePoints();
            // if(kb_tris.getPropertySize("inds") != 3){
            //     fmt::print(fg(fmt::color::red),"the kinematic boundary is not a surface triangulate mesh\n");
            //     throw std::runtime_error("the kinematic boundary is not a surface triangulate mesh");
            // }
            // if(!kb_tris.hasProperty("area")){
            //     fmt::print(fg(fmt::color::red),"the kinematic boundary has no 'area' channel\n");
            //     throw std::runtime_error("the kinematic boundary has no 'area' channel");
            // }     
            kverts.resize(kb_verts.size());
            TILEVEC_OPS::copy<3>(cudaPol,kb_verts,"x",kverts,"x");
            TILEVEC_OPS::copy<3>(cudaPol,kb_verts,"x",kverts,"xp");
            TILEVEC_OPS::fill(cudaPol,kverts,"area",(T)1.0);
        }
        // std::cout << "nm_kb_tris : " << kb_tris.size() << " nm_kb_verts : " << kb_verts.size() << std::endl;
        // cudaPol(zs::range(kb_tris.size()),
        //     [kb_verts = proxy<space>({},kb_verts),kb_tris = proxy<space>({},kb_tris),kverts = proxy<space>({},kverts)] ZS_LAMBDA(int ti) mutable {
        //         auto tri = kb_tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
        //         for(int i = 0;i != 3;++i)
        //             atomic_add(exec_cuda,&kverts("area",tri[i]),(T)kb_tris("area",ti)/(T)3.0);
        //         if(ti == 0)
        //             printf("tri[0] area : %f\n",(float)kb_tris("area",ti));
        // });

        const auto& bbw = (*zsparticles)[driven_tag];
        // the temp buffer only store the data that will change every iterations or every frame
        static dtiles_t vtemp{verts.get_allocator(),
                            {
                                {"grad", 3},
                                {"P", 9},
                                {"bou_tag",1},
                                {"dir", 3},
                                {"xn", 3},
                                {"xp",3},
                                {"vn",3},
                                {"vp",3},
                                {"is_inverted",1},
                                {"active",1},
                                {"k_active",1},
                                // {"inertia",1},
                                {"k_thickness",1},
                            },verts.size()};

        // auto max_collision_pairs = tris.size() / 10; 
        static dtiles_t etemp(eles.get_allocator(), {
                // {"H", 12 * 12},
                    {"ActInv",3*3},
                // {"muscle_ID",1},
                    {"is_inverted",1}
                }, eles.size()
        );

                // {{tags}, cnt, memsrc_e::um, 0}
        static dtiles_t sttemp(tris.get_allocator(),
            {
                {"nrm",3}
            },tris.size()
        );
        static dtiles_t setemp(lines.get_allocator(),
            {
                {"nrm",3}
            },lines.size()
        );

        // std::cout << "sttemp.size() << " << sttemp.size() << std::endl;
        // std::cout << "setemp.size() << " << setemp.size() << std::endl;

        int fp_buffer_size = points.size() * MAX_FP_COLLISION_PAIRS;
        // int fp_buffer_size = 0;

        static dtiles_t fp_buffer(points.get_allocator(),{
            {"inds",4},
            {"area",1},
            {"inverted",1},
        },fp_buffer_size);

        // static dtiles_t ee_buffer(lines.get_allocator(),{
        //     {"inds",4},
        //     {"area",1},
        //     {"inverted",1},
        //     {"abary",2},
        //     {"bbary",2},
        //     {"bary",4}
        // },lines.size());

        // int ee_buffer_size = ee_buffer.size();
        int ee_buffer_size = 0;


        int kc_buffer_size = kverts.size() * MAX_FP_COLLISION_PAIRS;
        // int kc_buffer_size = 0;

        static dtiles_t kc_buffer(points.get_allocator(),{
            {"inds",2},
            {"area",1},
            {"inverted",1},
        },kc_buffer_size);

        // int kc_buffer_size = kc_buffer.size();
        // int kc_buffer_size = 0;

// change
        // static dtiles_t gh_buffer(eles.get_allocator(),{
        //     {"inds",4},
        //     {"H",12*12},
        //     {"grad",12}
        // },eles.size() + bbw.size() + fp_buffer.size() + kc_buffer_size);

        static dtiles_t gh_buffer(eles.get_allocator(),{
            {"inds",4},
            {"H",12*12},
            {"grad",12}
        },eles.size() + fp_buffer.size());



        // TILEVEC_OPS::fill<4>(cudaPol,etemp,"inds",zs::vec<int,4>::uniform(-1).template reinterpret_bits<T>())
        // TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",etemp,"inds");
        TILEVEC_OPS::fill<9>(cudaPol,etemp,"ActInv",zs::vec<T,9>{1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0});
        // TILEVEC_OPS::fill(cudaPol,vtemp,"inertia",(T)1.0);
        // if(verts.hasProperty("inertia"))
        //     TILEVEC_OPS::copy(cudaPol,verts,"inertia",vtemp,"inertia");
        if(verts.hasProperty("k_thickness"))
            TILEVEC_OPS::copy(cudaPol,verts,"k_thickness",vtemp,"k_thickness");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"k_thickness",(T)1.0);
        // apply muscle activation

        if(!eles.hasProperty("Act"))
            eles.append_channels(cudaPol,{{"Act",1}});

        if(!eles.hasProperty(muscle_id_tag) || !eles.hasProperty("fiber"))
            fmt::print(fg(fmt::color::red),"the quadrature has no \"{}\" muscle_id_tag\n",muscle_id_tag);
        if(nm_acts == 0)
            fmt::print(fg(fmt::color::red),"no activation input\n");

        cudaPol(zs::range(eles.size()),
            [etemp = proxy<space>({},etemp),eles = proxy<space>({},eles),
                act_buffer = proxy<space>({},act_buffer),muscle_id_tag = SmallString(muscle_id_tag),nm_acts] ZS_LAMBDA(int ei) mutable {
                // auto act = eles.template pack<3>("act",ei);
                // auto fiber = etemp.template pack<3>("fiber",ei);

                vec3 act{1.0,1.0,1.0};
                vec3 fiber{};
                // float a = 1.0f;
                if(eles.hasProperty("fiber") && eles.hasProperty(muscle_id_tag) && nm_acts > 0 && (int)eles(muscle_id_tag,ei) >= 0 && fabs(eles.template pack<3>("fiber",ei).norm() - 1.0) < 0.001 && (int)eles(muscle_id_tag,ei) < act_buffer.size()){
                    fiber = eles.template pack<3>("fiber",ei);
                    auto ID = (int)eles(muscle_id_tag,ei);
                    auto a = 1. - act_buffer("act",0,ID);
                    auto b = 1. - act_buffer("act",1,ID);
                    // act = vec3{zs::sqrt(a),zs::sqrt(1./a),zs::sqrt(1./a)};
                    // auto aclamp = 
                    // act = vec3{a < 0.7 ? 0.7 : a,zs::sqrt(1./a),zs::sqrt(1./a)};
                    act = vec3{a,zs::sqrt(1./b),zs::sqrt(1./b)};
                    eles("Act",ei) = act_buffer("act",0,ID) + 1e-6;
                }else{
                    fiber = zs::vec<T,3>(1.0,0.0,0.0);
                    act = vec3{1,1,1};
                    eles("Act",ei) = (T)0.0;
                }
                if(fabs(fiber.norm() - 1.0) > 0.1) {
                    printf("invalid fiber[%d] detected : %f %f %f\n",(int)ei,
                        (float)fiber[0],(float)fiber[1],(float)fiber[2]);
                }

                vec3 dir[3];
                dir[0] = fiber;
                auto tmp = vec3{0.0,1.0,0.0};
                dir[1] = dir[0].cross(tmp);
                if(dir[1].length() < 1e-3) {
                    tmp = vec3{0.0,0.0,1.0};
                    dir[1] = dir[0].cross(tmp);
                }

                dir[1] = dir[1] / dir[1].length();
                dir[2] = dir[0].cross(dir[1]);
                dir[2] = dir[2] / dir[2].length();

                auto R = mat3{};
                for(int i = 0;i < 3;++i)
                    for(int j = 0;j < 3;++j)
                        R(i,j) = dir[j][i];

                auto Act = mat3::zeros();
                Act(0,0) = act[0];
                Act(1,1) = act[1];
                Act(2,2) = act[2];

                Act = R * Act * R.transpose();
                etemp.template tuple<9>("ActInv",ei) = zs::inverse(Act);
                // if(a < 1.0f) {
                //     auto ActInv = etemp.template pack<3,3>("ActInv",ei);
                //     printf("ActInv[%d] : \n%f %f %f\n%f %f %f\n%f %f %f\n",ei,
                //         (float)ActInv(0,0),(float)ActInv(0,1),(float)ActInv(0,2),
                //         (float)ActInv(1,0),(float)ActInv(1,1),(float)ActInv(1,2),
                //         (float)ActInv(2,0),(float)ActInv(2,1),(float)ActInv(2,2));
                // }
        });
        auto collisionStiffness = get_input2<float>("cstiffness");
        auto kineCollisionStiffness = get_input2<float>("kineCstiffness");


        // auto inset_ratio = get_input2<float>("collision_inset");
        // auto outset_ratio = get_input2<float>("collision_outset");    

        auto in_collisionEps = get_input2<float>("in_collisionEps");
        auto out_collisionEps = get_input2<float>("out_collisionEps");

        auto kine_in_collisionEps = get_input2<float>("kine_inCollisionEps");
        auto kine_out_collisionEps = get_input2<float>("kine_outCollisionEps");

        auto aniso_strength = get_input2<float>("aniso_strength");

        FEMDynamicSteppingSystem A{
            verts,eles,
            points,lines,tris,
            (T)in_collisionEps,(T)out_collisionEps,
            bbw,bverts,bone_driven_weight,
            volf,dt,collisionStiffness,
            (T)kine_in_collisionEps,(T)kine_out_collisionEps,
            (T)kineCollisionStiffness,(T)aniso_strength};

        // std::cout << "set initial guess" << std::endl;
        // setup initial guess
        // if(verts.hasProperty("dt")) {
        //     std::cout << "verts has property 'dt'" << std::endl;
        // }

        TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"xp");
        TILEVEC_OPS::copy<3>(cudaPol,verts,"v",vtemp,"vp");
        if(verts.hasProperty("active"))
            TILEVEC_OPS::copy(cudaPol,verts,"active",vtemp,"active");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"active",(T)1.0);

        if(verts.hasProperty("k_active"))
            TILEVEC_OPS::copy(cudaPol,verts,"k_active",vtemp,"k_active");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"k_active",(T)1.0);

        // if there is no init_x as guess, then use the baraff witkin approach
        // if(verts.hasProperty("init_x"))
        //     TILEVEC_OPS::copy<3>(cudaPol,verts,"init_x",vtemp,"xn");   
        // else {
            // TILEVEC_OPS::add<3>(cudaPol,vtemp,"xp",1.0,"vp",dt,"xn");
        TILEVEC_OPS::copy(cudaPol,verts,"v",vtemp,"vn");  
        TILEVEC_OPS::copy(cudaPol,verts,"x",vtemp,"xn");
            // TILEVEC_OPS::add<3>(cudaPol,verts,"x",1.0,"vp",(T)0.0,"xn");  
        // }
        if(verts.hasProperty("bou_tag") && verts.getPropertySize("bou_tag") == 1)
            TILEVEC_OPS::copy(cudaPol,verts,"bou_tag",vtemp,"bou_tag");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"bou_tag",(T)0.0);

        int max_newton_iterations = get_param<int>("max_newton_iters");
        int nm_iters = 0;
        // make sure, at least one baraf simi-implicit step will be taken
        auto res0 = 1e10;

        auto kd_alpha = get_input2<float>("kd_alpha");
        auto kd_beta = get_input2<float>("kd_beta");
        auto kd_theta = get_input2<float>("kd_theta");

        auto max_cg_iters = get_param<int>("max_cg_iters");

        while(nm_iters < max_newton_iterations) {
            // break;

            TILEVEC_OPS::fill(cudaPol,gh_buffer,"grad",(T)0.0);
            TILEVEC_OPS::fill(cudaPol,gh_buffer,"H",(T)0.0);  
            TILEVEC_OPS::fill<4>(cudaPol,gh_buffer,"inds",zs::vec<int,4>::uniform(-1).reinterpret_bits(float_c)); 
            A.findInversion(cudaPol,vtemp,etemp);  
            // match([&](auto &elasticModel,auto &anisoModel) -> std::enable_if_t<zs::is_same_v<RM_CVREF_T(anisoModel),zs::AnisotropicArap<float>>> {...},[](...) {
            //     A.computeGradientAndHessian(cudaPol, elasticModel,anisoModel,vtemp,etemp,gh_buffer,kd_alpha,kd_beta);
            // })(models.getElasticModel(),models.getAnisoElasticModel());
 
            match([&](auto &elasticModel,zs::AnisotropicArap<float> &anisoModel){
                A.computeGradientAndHessian(cudaPol, elasticModel,anisoModel,vtemp,etemp,gh_buffer,kd_alpha,kd_beta);
            },[](...) {
                throw std::runtime_error("unsupported anisotropic elasticity model");
            })(models.getElasticModel(),models.getAnisoElasticModel());

            match([&](auto &elasticModel) {
                A.computeCollisionGradientAndHessian(cudaPol,elasticModel,
                    vtemp,
                    etemp,
                    sttemp,
                    setemp,
                    // ee_buffer,
                    fp_buffer,
                    kverts,
                    kc_buffer,
                    gh_buffer,kd_theta);
            })(models.getElasticModel());

            TILEVEC_OPS::fill(cudaPol,vtemp,"grad",(T)0.0); 
            TILEVEC_OPS::assemble(cudaPol,gh_buffer,"grad","inds",vtemp,"grad");
            // break;

            PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"H",gh_buffer,"P",vtemp);
            // PCG::precondition<3>(cudaPol,vtemp,"P","grad","q");
            // T res = TILEVEC_OPS::inf_norm<3>(cudaPol, vtemp, "q");
            // if(res < newton_res){
            //     fmt::print(fg(fmt::color::cyan),"reach desire newton res {} : {}\n",newton_res,res);
            //     break;
            // }
            // auto nP = TILEVEC_OPS::inf_norm<9>(cudaPol,vtemp,"P");
            // std::cout << "nP : " << nP << std::endl;
            // PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"H",etemp,"P",vtemp);
            // if the grad is too small, return the result
            // Solve equation using PCG
            TILEVEC_OPS::fill(cudaPol,vtemp,"dir",(T)0.0);
            // std::cout << "solve using pcg" << std::endl;
            auto nm_CG_iters = PCG::pcg_with_fixed_sol_solve<3,4>(cudaPol,vtemp,gh_buffer,"dir","bou_tag","grad","P","inds","H",cg_res,max_cg_iters,100);
            fmt::print(fg(fmt::color::cyan),"nm_cg_iters : {}\n",nm_CG_iters);
            T alpha = 1.;

            auto nxn = TILEVEC_OPS::inf_norm<3>(cudaPol,vtemp,"xn");
            auto ndir = TILEVEC_OPS::dot<3>(cudaPol,vtemp,"dir","dir");
            auto nP = TILEVEC_OPS::dot<9>(cudaPol,vtemp,"P","P");

            // std::cout << "vtemp's xn : " << nxn << std::endl;
            // std::cout << "vtemp's dir : " << ndir << std::endl;
            // std::cout << "vtemp's P : " << nP << std::endl;

            cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),alpha,dt] __device__(int i) mutable {
                vtemp.template tuple<3>("xn", i) =
                    vtemp.template pack<3>("xn", i) + alpha * vtemp.template pack<3>("dir", i);
                vtemp.template tuple<3>("vn",i) = 
                    (vtemp.template pack<3>("xn",i) - vtemp.template pack<3>("xp",i))/dt; 
            });

            // nxn = TILEVEC_OPS::inf_norm<3>(cudaPol,vtemp,"xn");
            // std::cout << "new vtemp's xn : " << nxn << std::endl;


            // res = TILEVEC_OPS::inf_norm<3>(cudaPol, vtemp, "dir");// this norm is independent of descriterization
            // std::cout << "res[" << nm_iters << "] : " << res << std::endl;
            // if(res < newton_res){
            //     fmt::print(fg(fmt::color::cyan),"reach desire newton res {} : {}\n",newton_res,res);
            //     break;
            // }
            nm_iters++;
        }


        cudaPol(zs::range(verts.size()),
                [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),dt = dt] __device__(int vi) mutable {
                    // auto newX = vtemp.pack(dim_c<3>,"xn", vi);
                    verts.tuple<3>("x", vi) = vtemp.pack(dim_c<3>,"xn", vi);
                    // if(verts.hasProperty("dt"))
                    //     dt = verts("dt",vi);
                    verts.tuple<3>("v",vi) = vtemp.pack<3>("vn",vi);
                });

        set_output("ZSParticles", zsparticles);
    }
};

ZENDEFNODE(FleshDynamicStepping, {{"ZSParticles","kinematic_boundary",
                                    "gravity","Acts",
                                    "driven_boudary",
                                    {"string","driven_tag","bone_bw"},
                                    {"float","driven_weight","0.02"},
                                    {"string","muscle_id_tag","ms_id_tag"},
                                    {"float","cstiffness","0.0"},
                                    {"float","in_collisionEps","0.01"},
                                    {"float","out_collisionEps","0.01"},
                                    {"float","kineCstiffness","1"},
                                    {"float","kine_inCollisionEps","0.01"},
                                    {"float","kine_outCollisionEps","0.02"},
                                    {"float","dt","0.5"},
                                    {"float","newton_res","0.001"},
                                    {"float","kd_alpha","0.01"},
                                    {"float","kd_beta","0.01"},
                                    {"float","kd_theta","0.01"},
                                    {"float","aniso_strength","1.0"},
                                    },
                                  {"ZSParticles"},
                                  {
                                    {"int","max_cg_iters","1000"},
                                    {"int","max_newton_iters","5"},
                                    {"float","cg_res","0.0001"}
                                  },
                                  {"FEM"}});
};