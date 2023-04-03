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

struct TendonDynamicStepping : INode {

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

    struct TendonDynamicSteppingSystem {
        // the function won't reset gradient and hessian
        void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                            const dtiles_t& vtemp,
                            const dtiles_t& ttemp,
                            dtiles_t& gh_buffer) {        
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            TILEVEC_OPS::copy<3>(cudaPol,tris,"inds",gh_buffer,"inds");   
            // eval the inertia term gradient, 
            // cancel out the inertia term
            // cudaPol(zs::range(tris.size()),[dt2 = dt2,density = density,
            //             verts = proxy<space>({},verts),
            //             tris = proxy<space>({},tris),
            //             vtemp = proxy<space>({},vtemp),
            //             gh_buffer = proxy<space>({},gh_buffer),
            //             dt = dt,volf = volf] ZS_LAMBDA(int ti) mutable {
            //     auto m = density * tris("vol",ti)/(T)3.0;
            //     auto inds = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);

            //     auto inertia = (T)1.0;
            //     if(tris.hasProperty("inertia"))
            //         inertia = tris("inertia",ti);
            //     for(int i = 0;i != 3;++i){
            //         auto x1 = vtemp.pack(dim_c<3>,"xn",inds[i]);
            //         auto x0 = vtemp.pack(dim_c<3>,"xp",inds[i]);
            //         auto v0 = vtemp.pack(dim_c<3>,"vp",inds[i]);

            //         auto alpha = inertia * m;
            //         auto nodal_pgrad = -alpha * (x1 - x0 - v0 * dt) + dt2 * volf * tris("vol",ti)/(T)3.0;
            //         for(int d = 0;d != 3;++d){
            //             auto idx = i * 3 + d;
            //             gh_buffer("grad",idx,ti) += nodal_pgrad[d];
            //             gh_buffer("H",idx*9 + idx,ti) += alpha;
            //         }
                    
            //     }
            // });

            if(!tris.hasProperty("vol")) {
                std::cout << "tris has no vole" << std::endl;
                throw std::runtime_error("tris has no vole");
            }

            if(!tris.hasProperty("mu")) {
                std::cout << "tris has no mu" << std::endl;
                throw std::runtime_error("tris has no mu");
            }

            // eval the elasticity term
            cudaPol(zs::range(tris.size()), [dt = dt,dt2 = dt2,
                            verts = proxy<space>({},verts),
                            vtemp = proxy<space>({}, vtemp),
                            ttemp = proxy<space>({},ttemp),
                            gh_buffer = proxy<space>({},gh_buffer),
                            tris = proxy<space>({}, tris),
                            volf = volf] ZS_LAMBDA (int ti) mutable {
                auto IB = tris.template pack<2, 2>("IB", ti);
                auto inds = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                auto vole = tris("vol",ti);
                vec3 xs[3] = {vtemp.pack(dim_c<3>, "xn", inds[0]), vtemp.pack(dim_c<3>, "xn", inds[1]),
                                vtemp.pack(dim_c<3>, "xn", inds[2])};
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];  

                zs::vec<T, 3, 2> Ds{x1x0[0], x2x0[0], x1x0[1], x2x0[1], x1x0[2], x2x0[2]};
                auto F = Ds * IB;
                zs::vec<T,2,2> AInv = zs::vec<T,2,2>::identity();
                if(reinterpret_bits<int>(tris("nm_binders",ti)) > 1)
                    for(int i = 0;i != 2;++i)
                        AInv(i,i) = (T)1.0/ttemp("shrink",ti);
                auto dFActdF = dFAdF(AInv);
                F = F * AInv;

                auto dFdX = dFdXMatrix(IB,wrapv<3>{});
                auto dFdXT = dFdX.transpose();

                auto f0 = col(F, 0);
                auto f1 = col(F, 1);
                auto f0Norm = zs::sqrt(f0.l2NormSqr() + (T)1e-6);
                auto f1Norm = zs::sqrt(f1.l2NormSqr() + (T)1e-6);
                auto f0Tf1 = f0.dot(f1);
                zs::vec<T, 3, 2> Pstretch, Pshear;
                for (int d = 0; d != 3; ++d) {
                    Pstretch(d, 0) = 2 * (1 - 1 / f0Norm) * F(d, 0);
                    Pstretch(d, 1) = 2 * (1 - 1 / f1Norm) * F(d, 1);
                    Pshear(d, 0) = 2 * f0Tf1 * f1(d);
                    Pshear(d, 1) = 2 * f0Tf1 * f0(d);
                }

                auto mu = tris("mu",ti);

                auto vecP = dFActdF.transpose() * flatten(mu * Pstretch + (mu * 0.3) * Pshear);
                auto vfdt2 = -vole * (dFdXT * vecP) * (dt * dt);

                gh_buffer.tuple(dim_c<9>,"grad",ti) = gh_buffer.pack(dim_c<9>,"grad",ti) + vfdt2;
                // printf("f[%d] : %f\n",ti,vfdt2.norm());

                auto stretchHessian = [&F,&mu]() {
                    auto H = zs::vec<T, 6, 6>::zeros();
                    const zs::vec<T, 2> u{1, 0};
                    const zs::vec<T, 2> v{0, 1};
                    const T I5u = (F * u).l2NormSqr();
                    const T I5v = (F * v).l2NormSqr();
                    const T invSqrtI5u = (T)1 / zs::sqrt(I5u + (T)1e-6);
                    const T invSqrtI5v = (T)1 / zs::sqrt(I5v + (T)1e-6);

                    H(0, 0) = H(1, 1) = H(2, 2) = zs::max(1 - invSqrtI5u, (T)0);
                    H(3, 3) = H(4, 4) = H(5, 5) = zs::max(1 - invSqrtI5v, (T)0);

                    const auto fu = col(F, 0).normalized();
                    const T uCoeff = (1 - invSqrtI5u >= 0) ? invSqrtI5u : (T)1;
                    for (int i = 0; i != 3; ++i)
                        for (int j = 0; j != 3; ++j)
                            H(i, j) += uCoeff * fu(i) * fu(j);

                    const auto fv = col(F, 1).normalized();
                    const T vCoeff = (1 - invSqrtI5v >= 0) ? invSqrtI5v : (T)1;
                    for (int i = 0; i != 3; ++i)
                        for (int j = 0; j != 3; ++j)
                            H(3 + i, 3 + j) += vCoeff * fv(i) * fv(j);

                    H *= mu;
                    return H;
                };


                auto shearHessian = [&F,&mu]() {
                    using mat6 = zs::vec<T, 6, 6>;
                    auto H = mat6::zeros();
                    const zs::vec<T, 2> u{1, 0};
                    const zs::vec<T, 2> v{0, 1};
                    const T I6 = (F * u).dot(F * v);
                    const T signI6 = I6 >= 0 ? 1 : -1;

                    H(3, 0) = H(4, 1) = H(5, 2) = H(0, 3) = H(1, 4) = H(2, 5) = (T)1;

                    const auto g_ = F * (dyadic_prod(u, v) + dyadic_prod(v, u));
                    zs::vec<T, 6> g{};
                    for (int j = 0, offset = 0; j != 2; ++j) {
                        for (int i = 0; i != 3; ++i)
                            g(offset++) = g_(i, j);
                    }

                    const T I2 = F.l2NormSqr();
                    const T lambda0 = (T)0.5 * (I2 + zs::sqrt(I2 * I2 + (T)12 * I6 * I6 + (T)1e-6));

                    const zs::vec<T, 6> q0 = (I6 * H * g + lambda0 * g).normalized();

                    auto t = mat6::identity();
                    t = 0.5 * (t + signI6 * H);

                    const zs::vec<T, 6> Tq = t * q0;
                    const auto normTq = Tq.l2NormSqr() + (T)1e-6;

                    mat6 dPdF =
                        zs::abs(I6) * (t - (dyadic_prod(Tq, Tq) / normTq)) + lambda0 * (dyadic_prod(q0, q0));
                    dPdF *= (mu * 0.3);
                    return dPdF;
                };

                auto stH = stretchHessian();
                auto shH = shearHessian();
                auto He = stH + shH;
                auto dFdAct_dFdX = dFActdF * dFdX;
                auto H = dFdAct_dFdX.transpose() * He * dFdAct_dFdX * (dt * dt * vole);

                auto nH = H.norm();
                if(isnan(nH)){
                    auto nstH = stH.norm();
                    auto nshH = shH.norm();
                    auto nF = F.norm();
                    printf("nan H[%d] detected %f %f %f\n",ti,nstH,nshH,nF);
                }


                gh_buffer.template tuple<9*9>("H",ti) = gh_buffer.template pack<9,9>("H",ti) + H;

                // printf("H[%d] : %f \n IB : \n%f %f\n%f %f\n",ti,H.norm(),
                //     IB(0,0),IB(0,1),IB(1,0),IB(1,1));
            });
        }

        void computePositionConstraintGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
            dtiles_t& vtemp,
            const std::string& binderTag,
            const std::string& thicknessTag,
            const std::string& inversionTag,
            const dtiles_t& tris,
            const dtiles_t& kverts,
            dtiles_t& gh_buffer) {
                using namespace zs;
                constexpr auto space = execspace_e::cuda;
                int max_nm_binders = tris.getChannelSize(binderTag);
                // printf("max_nm_binders = %d\n",max_nm_binders);

                cudaPol(zs::range(tris.size()),
                    [vtemp = proxy<space>({},vtemp),
                        binderTag = zs::SmallString(binderTag),
                        thicknessTag = zs::SmallString(thicknessTag),
                        inversionTag = zs::SmallString(inversionTag),
                        tris = proxy<space>({},tris),
                        kverts = proxy<space>({},kverts),
                        // binderThickness = binderThickness,
                        binderStiffness = binderStiffness,
                        max_nm_binders = max_nm_binders,
                        gh_buffer = proxy<space>({},gh_buffer)] ZS_LAMBDA(int ti) mutable {
                    int nm_binders = 0;
                    // printf("binder_ids[%d] : %d %d %d %d\n",ti,
                    //     reinterpret_bits<int>(tris(binderTag,0,ti)),
                    //     reinterpret_bits<int>(tris(binderTag,1,ti)),
                    //     reinterpret_bits<int>(tris(binderTag,2,ti)),
                    //     reinterpret_bits<int>(tris(binderTag,3,ti)));
                    for(int i = 0;i != max_nm_binders;++i){
                        auto idx = reinterpret_bits<int>(tris(binderTag,i,ti));
                        if(idx < 0)
                            break;
                        ++nm_binders;
                    }

                    // printf("nm_binders : %d\n",nm_binders);

                    if(nm_binders == 0)
                        return;
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    auto mu = tris("mu",ti);
                    auto lam = tris("lam",ti);
                    // auto vole = tris("vol",ti);
                    vec3 cp[4] = {};

                    cp[1] = vtemp.pack(dim_c<3>,"xn",tri[0]);
                    cp[2] = vtemp.pack(dim_c<3>,"xn",tri[1]);
                    cp[3] = vtemp.pack(dim_c<3>,"xn",tri[2]);

                    vec3 cv[4] = {};
                    cv[0] = vec3::zeros();
                    cv[1] = vtemp.pack(dim_c<3>,"vn",tri[0]);
                    cv[2] = vtemp.pack(dim_c<3>,"vn",tri[1]);
                    cv[3] = vtemp.pack(dim_c<3>,"vn",tri[2]);

                    for(int i = 0;i != nm_binders;++i) {
                        auto idx = reinterpret_bits<int>(tris(binderTag,i,ti));
                        if(kverts.hasProperty("b_fail"))
                            if(kverts("b_fail",idx) > (T)0.5)
                                continue;
                        auto ceps = tris(thicknessTag,i,ti);
                        auto from_inside = tris(inversionTag,i,ti) > (T)0.0;
                        cp[0] = kverts.pack(dim_c<3>,"x",idx);
                        auto kstiffness = (T)1.0;
                        if(kverts.hasProperty("binderStiffness"))
                            kstiffness = kverts("binderStiffness",idx);

                        auto alpha = binderStiffness * kstiffness;
                        auto beta = (T)1.0/(T)nm_binders;
                        auto cgrad = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cp,mu,lam,ceps,from_inside);
                        auto cH = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cp,mu,lam,ceps,from_inside);

                        
                        for(int j = 3;j != 12;++j){
                            int row = j - 3;
                            atomic_add(exec_cuda,&gh_buffer("grad",row,ti),cgrad[j]);
                            for(int k = 3;k != 12;++k){
                                int col = k - 3;
                                atomic_add(exec_cuda,&gh_buffer("H",row*9 + col,ti),cH(j,k));
                            }                    
                        }
                    }
                });
        }

        void computeKinematicCollisionGradientAndHessian2(zs::CudaExecutionPolicy& cudaPol,
            dtiles_t& vtemp,
            const dtiles_t& tris,
            const dtiles_t& ktris,
            const dtiles_t& kverts,
            dtiles_t& kc_buffer,
            dtiles_t& gh_buffer) {
                using namespace zs;
                constexpr auto space = execspace_e::cuda;
                int offset = 0;
                auto stBvh = bvh_t{};
                // std::cout << "try retrieve bounding volumes" << std::endl;
                auto bvs = retrieve_bounding_volumes(cudaPol,kverts,ktris,wrapv<3>{},(T)0.0,"x");
                // std::cout << "end retrieve bounding volumes " << std::endl;
                stBvh.build(cudaPol,bvs);
                auto thickness = kine_out_collisionEps + kine_in_collisionEps;
                // calculate facet normal
                TILEVEC_OPS::fill<2>(cudaPol,kc_buffer,"inds",zs::vec<int,2>::uniform(-1).template reinterpret_bits<T>());
                std::cout << "do collision detection" << std::endl;
                cudaPol(zs::range(tris.size()),[in_collisionEps = kine_in_collisionEps,
                        out_collisionEps = kine_out_collisionEps,
                        tris = proxy<space>({},tris),
                        ktris = proxy<space>({},ktris),
                        kverts = proxy<space>({},kverts),
                        vtemp = proxy<space>({},vtemp),
                        kc_buffer = proxy<space>({},kc_buffer),
                        stBvh = proxy<space>(stBvh),thickness = thickness] ZS_LAMBDA(int ti) mutable {
                    auto p = vec3::zeros();
                    auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                    for(int i = 0;i != 3;++i)
                        p += vtemp.pack(dim_c<3>,"xn",tri[i]);
                    p = p/(T)3.0;
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

                        dist = seg.dot(nrm);
                        // change dist -> dist > 0 from dist < 0
                        auto collisionEps = dist < 0 ? out_collisionEps : in_collisionEps;
                        if(barySum > 1.1)
                            return;

                        if(distance > collisionEps)
                            return;

                        if(!LSL_GEO::pointProjectsInsideTriangle(kt0,kt1,kt2,p)){
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
                        }
                        kc_buffer.template tuple<2>("inds",ti * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) =
                                zs::vec<int,2>(ti,kti).template reinterpret_bits<T>();           
                        nm_collision_pairs++;  
                    };
                    stBvh.iter_neighbors(bv,process_kinematic_vertex_face_collision_pairs);
                });
                std::cout << "evaluate collision " << std::endl;
                cudaPol(zs::range(kc_buffer.size()),
                    [dt = dt,kd_theta = kd_theta,
                        kc_buffer = proxy<space>({},kc_buffer),
                        vtemp = proxy<space>({},vtemp),
                        tris = proxy<space>({},tris),
                        ktris = proxy<space>({},ktris),
                        kverts = proxy<space>({},kverts),
                        gh_buffer = proxy<space>({},gh_buffer),
                        in_collisionEps = kine_in_collisionEps,
                        out_collisionEps = kine_out_collisionEps,
                        stiffness = kineCollisionStiffness] ZS_LAMBDA(int cpi) mutable {
                    auto inds = kc_buffer.pack(dim_c<2>,"inds",cpi).reinterpret_bits(int_c);
                    for(int i = 0;i != 2;++i)
                        if(inds[i] < 0)
                            return;
                    auto p = vec3::zeros();
                    auto tri = tris.pack(dim_c<3>,"inds",inds[0]).reinterpret_bits(int_c);
                    for(int i = 0;i != 3;++i)
                        p += vtemp.pack(dim_c<3>,"xn",tri[i]);
                    p = p/(T)3.0;
                    vec3 cv[4] = {};
                    cv[0] = p;
                    auto ktri = ktris.pack(dim_c<3>,"inds",inds[1]).reinterpret_bits(int_c);
                    for(int j = 0;j != 3;++j)
                        cv[j + 1] = kverts.pack(dim_c<3>,"x",ktri[j]);
                    auto ceps = out_collisionEps;
                    auto alpha = stiffness;
                    auto beta = (T)1.0;
                    auto mu = tris("mu",inds[0]);
                    auto lam = tris("lam",inds[0]);
                    auto cgrad = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps,true)/(T)3.0;
                    auto cH = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps,true)/(T)9.0;

                    auto cforce = vec3{cgrad[0],cgrad[1],cgrad[2]};
                    auto cK = mat3{
                        cH(0,0),cH(0,1),cH(0,2),
                        cH(1,0),cH(1,1),cH(1,2),
                        cH(2,0),cH(2,1),cH(2,2)};
                    auto cC = kd_theta * cK;
                    cK += cC/dt;

                    vec3 v0[3] = {vtemp.pack(dim_c<3>,"vn", tri[0]),
                                    vtemp.pack(dim_c<3>,"vn", tri[1]),
                                    vtemp.pack(dim_c<3>,"vn", tri[2])}; 

                    for(int i = 0;i != 3;++i){
                        auto cdforce = cforce - cC * v0[i];
                        for(int j = 0;j != 3;++j)
                            atomic_add(exec_cuda,&gh_buffer("grad",i*3 + j,inds[0]),cdforce[j]);
                        for(int k = 0;k != 9;++k)
                            atomic_add(exec_cuda,&gh_buffer("H",(i*3+k/3)*9 + i*3 + k % 3,inds[0]),cK(k/3,k%3));
                    }
                });
        }


        void computePlaneConstraintGradientAndHessian2(zs::CudaExecutionPolicy& cudaPol,
                            const dtiles_t& vtemp,
                            const dtiles_t& kverts,
                            const dtiles_t& ktris,
                            const std::string& planeConsBaryTag,
                            const std::string& planeConsIDTag,
                            dtiles_t& nodal_gh_buffer) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            cudaPol(zs::range(verts.size()),[
                    verts = proxy<space>({},verts),
                    vtemp = proxy<space>({},vtemp),
                    kverts = proxy<space>({},kverts),
                    ktris = proxy<space>({},ktris),
                    planeConsBaryTag = zs::SmallString(planeConsBaryTag),
                    planeConsIDTag = zs::SmallString(planeConsIDTag),
                    kine_out_collisionEps = kine_out_collisionEps,
                    plane_constraint_stiffness = plane_constraint_stiffness,
                    nodal_gh_buffer = proxy<space>({},nodal_gh_buffer)] ZS_LAMBDA(int vi) mutable {
                auto idx = reinterpret_bits<int>(verts(planeConsIDTag,vi));
                if(idx < 0)
                    return;      
                auto ktri = ktris.pack(dim_c<3>,"inds",idx).reinterpret_bits(int_c);
                auto bary = verts.pack(dim_c<3>,planeConsBaryTag,vi);
                auto plane_root = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    plane_root += kverts.pack(dim_c<3>,"x",ktri[i]) * bary[i];
                auto plane_nrm = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    plane_nrm += kverts.pack(dim_c<3>,"nrm",ktri[i]) * bary[i];
                plane_nrm /= (plane_nrm.norm() + 1e-6);    

                auto mu = verts("mu",vi);
                auto lam = verts("lam",vi);
                    // if(distance > collisionEps)
                auto eps = kine_out_collisionEps;
                auto p = vtemp.pack(dim_c<3>,"xn",vi);
                auto seg = p - plane_root;

                auto fc = vec3::zeros();
                auto Hc = mat3::zeros();
                auto dist = seg.dot(plane_nrm) - eps;
                if(dist < (T)0){
                    fc = -dist * mu * plane_constraint_stiffness * plane_nrm;
                    Hc = mu * plane_constraint_stiffness * dyadic_prod(plane_nrm,plane_nrm);
                }

                // printf("apply plane constraint with force : %f %f\n",(float)dist,(float)fc.norm());

                nodal_gh_buffer.tuple(dim_c<3>,"grad",vi) = fc;
                nodal_gh_buffer.tuple(dim_c<3,3>,"H",vi) = Hc;

            });
        }

        void computeKinematicCollisionGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
            dtiles_t& vtemp,
            dtiles_t& ttemp,
            const dtiles_t& kvtemp,
            dtiles_t& kc_buffer,
            dtiles_t& gh_buffer) {
                using namespace zs;
                constexpr auto space = execspace_e::cuda;
                int offset = 0;
                // building the bounding volumes of surface mesh
                auto stBvh = bvh_t{};
                auto bvs = retrieve_bounding_volumes(cudaPol,vtemp,tris,wrapv<3>{},(T)0.0,"xn");
                stBvh.build(cudaPol,bvs);
                auto thickness = kine_out_collisionEps + kine_in_collisionEps;
                // calculate facet normal
                cudaPol(zs::range(tris.size()),
                    [ttemp = proxy<space>({},ttemp),
                        vtemp = proxy<space>({},vtemp),
                        tris = proxy<space>({},tris)] ZS_LAMBDA(int ti) {
                    auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                    auto v0 = vtemp.template pack<3>("xn",tri[0]);
                    auto v1 = vtemp.template pack<3>("xn",tri[1]);
                    auto v2 = vtemp.template pack<3>("xn",tri[2]);

                    auto e01 = v1 - v0;
                    auto e02 = v2 - v0;

                    auto nrm = e01.cross(e02);
                    auto nrm_norm = nrm.norm();
                    if(nrm_norm < 1e-8)
                        nrm = zs::vec<T,3>::zeros();
                    else
                        nrm = nrm / nrm_norm;

                    ttemp.tuple(dim_c<3>,"nrm",ti) = nrm;
                });
                // find all the collision pairs
                TILEVEC_OPS::fill<2>(cudaPol,kc_buffer,"inds",zs::vec<int,2>::uniform(-1).template reinterpret_bits<T>());
                cudaPol(zs::range(kvtemp.size()),[in_collisionEps = kine_in_collisionEps,
                    out_collisionEps = kine_out_collisionEps,
                    tris = proxy<space>({},tris),
                    vtemp = proxy<space>({},vtemp),
                    ttemp = proxy<space>({},ttemp),
                    kvtemp = proxy<space>({},kvtemp),
                    kc_buffer = proxy<space>({},kc_buffer),
                    stBvh = proxy<space>(stBvh),thickness = thickness] ZS_LAMBDA(int kvi) mutable {
                        auto kp = kvtemp.pack(dim_c<3>,"x",kvi);
                        auto bv = bv_t{get_bounding_box(kp - thickness,kp + thickness)};

                        int nm_collision_pairs = 0;
                        auto process_kinematic_vertex_face_collision_pairs = [&](int ti) {
                            if(nm_collision_pairs >= MAX_FP_COLLISION_PAIRS)
                                return;
                            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);

                            T dist = (T)0.0;

                            auto nrm = ttemp.pack(dim_c<3>,"nrm",ti);
                            auto seg = kp - vtemp.pack(dim_c<3>,"xn",tri[0]);


                            auto t0 = vtemp.pack(dim_c<3>,"xn",tri[0]);
                            auto t1 = vtemp.pack(dim_c<3>,"xn",tri[1]);
                            auto t2 = vtemp.pack(dim_c<3>,"xn",tri[2]);

                            auto e01 = (t0 - t1).norm();
                            auto e02 = (t0 - t2).norm();
                            auto e12 = (t1 - t2).norm();

                            T barySum = (T)1.0;
                            T distance = LSL_GEO::pointTriangleDistance(t0,t1,t2,kp,barySum);

                            dist = seg.dot(nrm);
                            // increase the stability, the tri must already in collided in the previous frame before been penerated in the current frame
                            auto collisionEps = dist < 0 ? out_collisionEps : in_collisionEps;

                            if(barySum > 1.1)
                                return;

                            if(distance > collisionEps)
                                return;

                            // if the triangle cell is too degenerate
                            if(!LSL_GEO::pointProjectsInsideTriangle(t0,t1,t2,kp)){
                                auto ntris = tris.pack(dim_c<3>,"ff_inds",ti).reinterpret_bits(int_c);
                                for(int i = 0;i != 3;++i) {
                                    auto nti = ntris[i];
                                    auto bisector_normal = ttemp.pack(dim_c<3>,"nrm",ti) + ttemp.pack(dim_c<3>,"nrm",nti);
                                    bisector_normal = (bisector_normal)/(bisector_normal.norm() + (T)1e-6);
                                    seg = kp - vtemp.pack(dim_c<3>,"xn",tri[i]);
                                    if(bisector_normal.dot(seg) < 0)
                                        return;
                                }
                            }

                            kc_buffer.template tuple<2>("inds",kvi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) =
                                    zs::vec<int,2>(kvi,ti).template reinterpret_bits<T>();           
                            nm_collision_pairs++;  
                        };
                        stBvh.iter_neighbors(bv,process_kinematic_vertex_face_collision_pairs);
                });                

                cudaPol(zs::range(kc_buffer.size()),
                    [kc_buffer = proxy<space>({},kc_buffer),
                        vtemp = proxy<space>({},vtemp),
                        tris = proxy<space>({},tris),
                        kvtemp = proxy<space>({},kvtemp),
                        gh_buffer = proxy<space>({},gh_buffer),
                        in_collisionEps = kine_in_collisionEps,
                        out_collisionEps = kine_out_collisionEps,
                        stiffness = kineCollisionStiffness] ZS_LAMBDA(int cpi) mutable {
                    auto inds = kc_buffer.pack(dim_c<2>,"inds",cpi).reinterpret_bits(int_c);
                    for(int i = 0;i != 2;++i)
                        if(inds[i] < 0)
                            return;
                    vec3 cv[4] = {};
                    cv[0] = kvtemp.pack(dim_c<3>,"x",inds[0]);
                    auto tri = tris.pack(dim_c<3>,"inds",inds[1]).reinterpret_bits(int_c);
                    for(int j = 0;j != 3;++j)
                        cv[j+1] = vtemp.pack(dim_c<3>,"xn",tri[j]);
                    auto ceps = out_collisionEps;
                    auto alpha = stiffness;
                    auto beta = (T)1.0;
                    auto mu = tris("mu",inds[1]);
                    auto lam = tris("lam",inds[1]);
                    auto cgrad = -alpha * beta * VERTEX_FACE_SQRT_COLLISION::gradient(cv,mu,lam,ceps,true);
                    auto cH = alpha * beta * VERTEX_FACE_SQRT_COLLISION::hessian(cv,mu,lam,ceps,true);

                    for(int i = 3;i != 12;++i){
                        int d0 = i % 3;
                        atomic_add(exec_cuda,&gh_buffer("grad",i-3,inds[1]),cgrad[i]);
                        for(int j = 3;j != 12;++j){
                            int d1 = j % 3;
                            atomic_add(exec_cuda,&gh_buffer("H",(i-3)*9 + (j-3),inds[1]),cH(i,j));
                        }
                    }
                });                             
        }

        TendonDynamicSteppingSystem(const tiles_t &verts,const tiles_t& tris,
                const vec3& volf,const T& density,const T& _dt,
                const T& kine_in_collisionEps,const T& kine_out_collisionEps,
                const T& kineCollisionStiffness,const T& kd_theta,
                const T& binderThickness,const T& binderStiffness,const T& plane_constraint_stiffness)
            : verts{verts}, tris{tris},volf{volf},density{density},kd_theta{kd_theta},
                    binderThickness{binderThickness},binderStiffness{binderStiffness},plane_constraint_stiffness{plane_constraint_stiffness},
                    kine_in_collisionEps{kine_in_collisionEps},kine_out_collisionEps{kine_out_collisionEps},
                    kineCollisionStiffness{kineCollisionStiffness},dt{_dt}, dt2{_dt * _dt}{}

        const tiles_t &verts;
        const tiles_t &tris;
        T kd_theta;
        T density;
        vec3 volf;
        T dt;
        T dt2;
        T in_collisionEps;
        T out_collisionEps;

        T collisionStiffness;

        T binderThickness;
        T binderStiffness;

        T kine_in_collisionEps;
        T kine_out_collisionEps;
        T kineCollisionStiffness;

        T plane_constraint_stiffness;
    };


    void apply() override {
        using namespace zs;
        auto zssurf = get_input<ZenoParticles>("zssurf");

        auto gravity = zeno::vec<3,T>(0);
        if(has_input("gravity"))
            gravity = get_input2<zeno::vec<3,T>>("gravity");
        T armoji = (T)1e-4;
        T wolfe = (T)0.9;

        T cg_res = get_param<float>("cg_res");
        auto models = zssurf->getModel();
        auto& verts = zssurf->getParticles();
        auto& tris = zssurf->getQuadraturePoints();

        if(tris.getChannelSize("inds") != 3){
            fmt::print(fg(fmt::color::red),"only tris mesh is supported");
            throw std::runtime_error("the tendon must be a trimesh");
        }

        auto newton_res = get_input2<float>("newton_res");

        auto dt = get_input2<float>("dt");

        auto volf = vec3::from_array(gravity * models.density);

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();


        auto kverts = typename ZenoParticles::particles_t({
                {"x",3},
                {"xp",3},
                {"b_fail",1},
                {"binderStiffness",1},
                {"nrm",3},
                {"area",1}},0,zs::memsrc_e::device,0);
        auto ktris = typename ZenoParticles::particles_t({
                {"inds",3},{"nrm",3}},0,zs::memsrc_e::device,0);

        if(has_input<ZenoParticles>("kboundary")){
            auto kinematic_boundary = get_input<ZenoParticles>("kboundary");
            // if (kinematic_boundary.empty())

            // const auto& prim_kverts = kinematic_boundary.verts;
            // auto& prim_kverts_area = kinematic_boundary.attr<float>("area");
            auto& kb_verts = kinematic_boundary->getParticles();
            auto& kb_tris = kinematic_boundary->getQuadraturePoints();

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
            TILEVEC_OPS::copy<3>(cudaPol,kb_verts,"nrm",kverts,"nrm");
            TILEVEC_OPS::fill(cudaPol,kverts,"area",(T)1.0);
            if(kb_verts.hasProperty("b_fail"))
                TILEVEC_OPS::copy(cudaPol,kb_verts,"b_fail",kverts,"b_fail");
            else 
                TILEVEC_OPS::fill(cudaPol,kverts,"b_fail",(T)0.0);
            if(kb_verts.hasProperty("binderStiffness"))
                TILEVEC_OPS::copy(cudaPol,kb_verts,"binderStiffness",kverts,"binderStiffness");
            else 
                TILEVEC_OPS::fill(cudaPol,kverts,"binderStiffness",(T)1.0);

            ktris.resize(kb_tris.size());
            TILEVEC_OPS::copy<3>(cudaPol,kb_tris,"nrm",ktris,"nrm");
            TILEVEC_OPS::copy<3>(cudaPol,kb_tris,"inds",ktris,"inds");   
        }

        dtiles_t vtemp{verts.get_allocator(),
            {
                {"dir",3},
                {"grad",3},
                {"P",9},
                {"xn",3},
                {"xp",3},
                {"vn",3},
                {"vp",3},
                {"active",1},
                {"k_active",1},
                {"bou_tag",1},
                {"grad",3},
                {"H",9},
                {"inds",1}
            },verts.size()};
        dtiles_t ttemp{tris.get_allocator(),
            {
                {"shrink",1},
                {"nrm",3}
            },tris.size()};
        dtiles_t kc_buffer{kverts.get_allocator(),
            {
                {"inds",2}
            },kverts.size() * MAX_FP_COLLISION_PAIRS};
        // dtiles_t kc_buffer{tris.get_allocator(),
        //     {
        //         {"inds",2}
        //     },tris.size() * MAX_FP_COLLISION_PAIRS};
        dtiles_t gh_buffer(tris.get_allocator(),{
            {"inds",3},
            {"H",9 * 9},
            {"grad",9}
        },tris.size());

        auto shrink = get_input2<float>("shrink");
        auto kineCollisionStiffness = get_input2<float>("kineColStiff");
        auto kineInCollisionEps = get_input2<float>("kineInColEps");
        auto kineOutCollisionEps = get_input2<float>("kineOutColEps");
        
        auto binderThickness = get_input2<float>("binderThickness");
        auto binderStiffness = get_input2<float>("binderStiffness");
        auto binderTag = get_param<std::string>("binderTag");
        auto binderThicknessTag = get_param<std::string>("binderThicknessTag");
        auto binderInversionTag = get_param<std::string>("binderInversionTag");

        auto planeConsPosTag = get_param<std::string>("planeConsPosTag");
        auto planeConsNrmTag = get_param<std::string>("planeConsNrmTag");
        auto planeConsIDTag = get_param<std::string>("planeConsIDTag");
        auto planeConsBaryTag = get_param<std::string>("planeConsBaryTag");

        auto planeConsStiffness = get_input2<float>("planeConsStiffness");


        if(!tris.hasProperty(binderTag)) {
            fmt::print(fg(fmt::color::red),"the tris has no binderTag {}\n",binderTag);
            throw std::runtime_error("the tris has no binderTag");
        }

        auto kd_theta = get_input2<float>("kd_theta");

        TendonDynamicSteppingSystem A{verts,tris,
            volf,models.density,dt,
            kineInCollisionEps,kineOutCollisionEps,
            kineCollisionStiffness,kd_theta,
            binderThickness,binderStiffness,planeConsStiffness};
        
        // step forward
        TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"xp");
        TILEVEC_OPS::copy<3>(cudaPol,verts,"v",vtemp,"vp");
        // set initial guess for system equation
        // TILEVEC_OPS::copy(cudaPol,verts,"v",vtemp,"vn");  
        TILEVEC_OPS::copy(cudaPol,verts,"x",vtemp,"xn");
        if(verts.hasProperty("k_thickness"))
            TILEVEC_OPS::copy(cudaPol,verts,"k_thickness",vtemp,"k_thickness");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"k_thickness",(T)1.0);
        if(verts.hasProperty("is_inverted"))
            TILEVEC_OPS::copy(cudaPol,verts,"is_inverted",vtemp,"is_inverted");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"is_inverted",(T)0.0);
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                vtemp("inds",vi) = reinterpret_bits<T>(vi);
        });

        if(verts.hasProperty("active"))
            TILEVEC_OPS::copy(cudaPol,verts,"active",vtemp,"active");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"active",(T)1.0);

        if(verts.hasProperty("k_active"))
            TILEVEC_OPS::copy(cudaPol,verts,"k_active",vtemp,"k_active");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"k_active",(T)1.0);

        TILEVEC_OPS::fill(cudaPol,ttemp,"shrink",(T)shrink);
        TILEVEC_OPS::fill(cudaPol,vtemp,"vn",(T)0.0);

        if(verts.hasProperty("bou_tag")) {
            TILEVEC_OPS::copy(cudaPol,verts,"bou_tag",vtemp,"bou_tag");
        }else{
            TILEVEC_OPS::fill(cudaPol,vtemp,"bou_tag",(T)0.0);
        }

        bool use_plane_constraint = get_input2<int>("use_plane_constraint") > 0;


        int max_newton_iterations = get_param<int>("max_newton_iters");
        int nm_iters = 0;            
        auto max_cg_iters = get_param<int>("max_cg_iters");

        while(nm_iters < max_newton_iterations) {
            // clear 'grad' and 'H' buffer
            TILEVEC_OPS::fill(cudaPol,gh_buffer,"grad",(T)0.0);
            TILEVEC_OPS::fill(cudaPol,gh_buffer,"H",(T)0.0);
            TILEVEC_OPS::fill(cudaPol,vtemp,"grad",(T)0.0);
            TILEVEC_OPS::fill(cudaPol,vtemp,"H",(T)0.0);
            // evaluate element-wise gradient and hessian
            // std::cout << "eval hessian" << std::endl;
            A.computeGradientAndHessian(cudaPol,vtemp,ttemp,gh_buffer);
            A.computePositionConstraintGradientAndHessian(cudaPol,
                vtemp,
                binderTag,
                binderThicknessTag,
                binderInversionTag,
                tris,
                kverts,
                gh_buffer);

            if(verts.hasProperty(planeConsPosTag) && verts.hasProperty(planeConsNrmTag) && verts.hasProperty(planeConsIDTag) && verts.hasProperty(planeConsBaryTag) && use_plane_constraint){
                std::cout << "apply plane constraint" << std::endl;
                // A.computePlaneConstraintGradientAndHessian(cudaPol,
                //     vtemp,
                //     planeConsPosTag,
                //     planeConsNrmTag,
                //     planeConsIDTag,
                //     vtemp);
                A.computePlaneConstraintGradientAndHessian2(cudaPol,
                    vtemp,
                    kverts,
                    ktris,
                    planeConsBaryTag,
                    planeConsIDTag,
                    vtemp);
            }else{
                std::cout << "apply no plane constraint : " << 
                    verts.hasProperty(planeConsPosTag) << "\t" << 
                    verts.hasProperty(planeConsNrmTag) << "\t" << 
                    verts.hasProperty(planeConsIDTag) << "\t" << use_plane_constraint << std::endl;
            }
            // std::cout << "eval kinematic hessian" << std::endl;
            // A.computeKinematicCollisionGradientAndHessian2(cudaPol,
            //     vtemp,tris,ktris,kverts,kc_buffer,gh_buffer);
            // std::cout << "finish eval hessian" << std::endl;
            // A.computeKinematicCollisionGradientAndHessian(cudaPol,
            //     vtemp,ttemp,kverts,kc_buffer,gh_buffer);
            // assemble element-wise gradient
            TILEVEC_OPS::assemble(cudaPol,gh_buffer,"grad","inds",vtemp,"grad");
            TILEVEC_OPS::fill(cudaPol,vtemp,"P",(T)0.0);
            PCG::prepare_block_diagonal_preconditioner<3,3>(cudaPol,"H",gh_buffer,"P",vtemp,false,true);
            PCG::prepare_block_diagonal_preconditioner<1,3>(cudaPol,"H",vtemp,"P",vtemp,true,true);
            

            // auto nP = TILEVEC_OPS::dot<9>(cudaPol,vtemp,"P","P");
            // // std::cout << "P : " << nP << std::endl;
            // auto ng = TILEVEC_OPS::dot<9>(cudaPol,vtemp,"grad","grad");
            // std::cout << "ng : " << TILEVEC_OPS::dot<9>(cudaPol,vtemp,"grad","grad") << std::endl;
            TILEVEC_OPS::fill(cudaPol,vtemp,"dir",(T)0.0);
            std::cout << "launch cg solver" << std::endl;
            auto nm_CG_iters = PCG::pcg_with_fixed_sol_solve<3,3>(cudaPol,vtemp,gh_buffer,"dir","bou_tag","grad","P","inds","H",cg_res,max_cg_iters,100);
            std::cout << "finish cg solver" << std::endl;
            auto ndir = TILEVEC_OPS::inf_norm<3>(cudaPol,vtemp,"dir");
            // std::cout << "ndir : " << ndir << std::endl;

            fmt::print(fg(fmt::color::cyan),"nm_cg_iters : {}\n",nm_CG_iters);

            T alpha = 1.;
            cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),alpha,dt] __device__(int i) mutable {
                vtemp.template tuple<3>("xn", i) =
                    vtemp.template pack<3>("xn", i) + alpha * vtemp.template pack<3>("dir", i);
                vtemp.template tuple<3>("vn",i) = 
                    (vtemp.template pack<3>("xn",i) - vtemp.template pack<3>("xp",i))/dt; 
            });

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

        set_output("zssurf", zssurf);   
    }
};

ZENDEFNODE(TendonDynamicStepping, {{"zssurf","gravity","kboundary",
                                        {"float","kineColStiff","1"},
                                        {"float","kineInColEps","0.01"},
                                        {"float","kineOutColEps","0.02"},
                                        {"float","binderThickness","0.01"},
                                        {"float","binderStiffness","1.0"},
                                        {"float","dt","0.5"},
                                        {"float","newton_res","0.001"},
                                        {"float","shrink","1.0"},
                                        {"float","kd_theta","1.0"},
                                        {"float","planeConsStiffness","0.01"},
                                        {"int","use_plane_constraint","0"}
                                    },
                                  {"zssurf"},
                                  {
                                    {"int","max_cg_iters","1000"},
                                    {"int","max_newton_iters","5"},
                                    {"float","cg_res","0.0001"},
                                    {"string","binderTag","binderTag"},
                                    {"string","binderThicknessTag","binderThicknessTag"},
                                    {"string","thicknessTag","thicknessTag"},
                                    {"string","binderInversionTag","binderInversionTag"},
                                    {"string","planeConsPosTag","planeConsPosTag"},
                                    {"string","planeConsNrmTag","planeConsNrmTag"},
                                    {"string","planeConsIDTag","planeConsIDTag"},
                                    {"string","planeConsBaryTag","planeConsBaryTag"},                                    
                                  },
                                  {"FEM"}});

};