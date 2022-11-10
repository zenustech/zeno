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
#include "../geometry/kernel/calculate_bisector_normal.hpp"
#include "../geometry/kernel/calculate_facet_normal.hpp"
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/compute_characteristic_length.hpp"

#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"

// #include "collision_energy/vertex_face_collision.hpp"
// #include "collision_energy/vertex_face_sqrt_collision.hpp"
// #include "collision_energy/edge_edge_collision.hpp"
// #include "collision_energy/edge_edge_sqrt_collition.hpp"

#include "collision_energy/evaluate_collision.hpp"

#define DEBUG_FLESH_DYN_STEPPING 1

namespace zeno {

// TODO : boundary force
// TODO : fixed points
// Anisotropic Cardiac

#define MAX_FP_COLLISION_PAIRS 6

struct FleshDynamicStepping : INode {

    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
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

        constexpr auto dFAdF(const mat3& A) {
            mat9 M{};
            M(0,0) = M(1,1) = M(2,2) = A(0,0);
            M(3,0) = M(4,1) = M(5,2) = A(0,1);
            M(6,0) = M(7,1) = M(8,2) = A(0,2);

            M(0,3) = M(1,4) = M(2,5) = A(1,0);
            M(3,3) = M(4,4) = M(5,5) = A(1,1);
            M(6,3) = M(7,4) = M(8,5) = A(1,2);

            M(0,6) = M(1,7) = M(2,8) = A(2,0);
            M(3,6) = M(4,7) = M(5,8) = A(2,1);
            M(6,6) = M(7,7) = M(8,8) = A(2,2);

            return M;        
        }

        template <typename Model>
        void computeCollisionEnergy(zs::CudaExecutionPolicy& cudaPol,const Model& model,
                dtiles_t& vtemp,
                dtiles_t& etemp,
                dtiles_t& sttemp,
                dtiles_t& setemp,
                dtiles_t& cptemp,
                // const bvh_t& stBvh,
                // const bvh_t& seBvh,
                const T& thickness) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            T lambda = model.lam;
            T mu = model.mu;


        }


        template <typename Model>
        void computeCollisionGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,const Model& model,
                            dtiles_t& vtemp,
                            dtiles_t& etemp,
                            dtiles_t& sttemp,
                            dtiles_t& setemp,
                            dtiles_t& cptemp,
                            // const bvh_t& stBvh,
                            // const bvh_t& seBvh,
                            const T& thickness,
                            bool explicit_collision = false,
                            bool neglect_inverted = true) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            T lambda = model.lam;
            T mu = model.mu; 

            #if DEBUG_FLESH_DYN_STEPPING
                if(!vtemp.hasProperty("grad"))
                    fmt::print(fg(fmt::color::red),"the vtemp has no 'grad' channel\n");
                if(!vtemp.hasProperty("xn"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'xn' channel\n");
                if(!vtemp.hasProperty("xp"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'xn' channel\n");
                if(!vtemp.hasProperty("is_inverted"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'is_inverted' channel\n");
                if(!vtemp.hasProperty("vp"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'vp' channel\n");

                if(!etemp.hasProperty("H"))
                    fmt::print(fg(fmt::color::red),"the etemp has no 'H' channel\n");
                if(!etemp.hasProperty("ActInv"))
                    fmt::print(fg(fmt::color::red),"the etemp has no 'ActInv' channel\n");
                
                if(!verts.hasProperty("m"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'm' channel\n");

                if(!eles.hasProperty("inds"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'IB' channel\n");        
                if(!eles.hasProperty("IB"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'IB' channel\n");
                if(!eles.hasProperty("m"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'm' channel\n");
                if(!eles.hasProperty("vol"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'vol' channel\n");

                // fmt::print(fg(fmt::color::blue),"the size of tris : {}\n",tris.size());
                if(!tris.hasProperty("inds"))
                    fmt::print(fg(fmt::color::red),"the tris has no 'inds' channel\n");
                if(!tris.hasProperty("area"))
                    fmt::print(fg(fmt::color::red),"the tris has no 'area' channel\n");
                if(!points.hasProperty("area"))
                    fmt::print(fg(fmt::color::red),"the points has no 'area' channel\n");

            #endif            

            auto xtag = zs::SmallString("xn");
            if(explicit_collision)
                xtag = zs::SmallString("xp");


            if(neglect_inverted) {
            // // figure out all the vertices which is incident to an inverted tet
                TILEVEC_OPS::fill(cudaPol,vtemp,"is_inverted",reinterpret_bits<T>((int)0));  
                cudaPol(zs::range(eles.size()),
                    [vtemp = proxy<space>({},vtemp),quads = proxy<space>({},eles),xtag] ZS_LAMBDA(int ei) mutable {
                        auto DmInv = quads.template pack<3,3>("IB",ei);
                        auto inds = quads.template pack<4>("inds",ei).reinterpret_bits(int_c);
                        vec3 x1[4] = {vtemp.template pack<3>(xtag, inds[0]),
                                vtemp.template pack<3>(xtag, inds[1]),
                                vtemp.template pack<3>(xtag, inds[2]),
                                vtemp.template pack<3>(xtag, inds[3])};   

                        mat3 F{};
                        {
                            auto x1x0 = x1[1] - x1[0];
                            auto x2x0 = x1[2] - x1[0];
                            auto x3x0 = x1[3] - x1[0];
                            auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                                            x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                            F = Ds * DmInv;
                        } 
                        if(zs::determinant(F) < 0.0)
                            for(int i = 0;i < 4;++i)
                                vtemp("is_inverted",inds[i]) = reinterpret_bits<T>((int)1);                  
                });

            }


#if 0
            TILEVEC_OPS::fill<4>(cudaPol,cptemp,"inds",zs::vec<int,4>::uniform(-1).template reinterpret_bits<T>());
            // TILEVEC_OPS::fill<12*12>(cudaPol,cptemp,"H",zs::vec<T,12*12>::zeros());

            // compute vertex facet contact pairs
            cudaPol(zs::range(points.size()),[lambda = lambda,mu = mu,collisionStiffness = collisionStiffness,
                            in_collisionEps = in_collisionEps,out_collisionEps = out_collisionEps,
                            vtemp = proxy<space>({},vtemp),
                            etemp = proxy<space>({},etemp),
                            sttemp = proxy<space>({},sttemp),
                            setemp = proxy<space>({},setemp),
                            cptemp = proxy<space>({},cptemp),
                            points = proxy<space>({},points),
                            lines = proxy<space>({},lines),
                            tris = proxy<space>({},tris),
                            stbvh = proxy<space>(stBvh),thickness = thickness,
                            neglect_inverted = neglect_inverted,xtag] ZS_LAMBDA(int svi) mutable {
                // if(svi == 0)    {
                //     if(tris.hasProperty("inds"))
                //         printf("compare size : %d %d %d\n",(int)vtemp.size(),(int)tris.size(),(int)tris.propertySize("inds"));
                //     else
                //         printf("the tris has no inds channel!!!\n"); 
                // }


                auto vi = reinterpret_bits<int>(points("inds",svi));

                if(neglect_inverted)   {
                    auto is_vertex_inverted = reinterpret_bits<int>(vtemp("is_inverted",vi));
                    if(is_vertex_inverted)
                        return;
                }

                auto p = vtemp.template pack<3>(xtag,vi);
                auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};


                // check whether there is collision happening, and if so, apply the collision force and addup the collision hessian
                int nm_collision_pairs = 0;
                auto process_vertex_face_collision_pairs = [&](int stI) {

                    if(nm_collision_pairs >= MAX_FP_COLLISION_PAIRS)     
                        return;   

                    auto tri = tris.pack(dim_c<3>, "inds",stI).reinterpret_bits(int_c);
                    if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
                        return;

                    auto t0 = vtemp.template pack<3>(xtag,tri[0]);
                    auto t1 = vtemp.template pack<3>(xtag,tri[1]);
                    auto t2 = vtemp.template pack<3>(xtag,tri[2]);
                    // check whether the triangle is degenerate
                    auto restArea = tris("area",stI);
                    // skip the triangle too small at rest configuration
                    // if(restArea < (T)1e-6)
                    //     return;

                    const auto e10 = t1 - t0;
                    const auto e20 = t2 - t0;
                    auto deformedArea = (T)0.5 * e10.cross(e20).norm();
                    const T degeneracyEps = 1e-4;
                    // skip the degenerate triangles
                    const T relativeArea = deformedArea / (restArea + (T)1e-6);
                    if(relativeArea < degeneracyEps)
                        return;

                    bool collide = false;

                    if(COLLISION_UTILS::is_inside_the_cell(vtemp,xtag,
                            lines,tris,
                            sttemp,"nrm",
                            setemp,"nrm",
                            stI,p,in_collisionEps,out_collisionEps)){
                        // printf("find collision facet-vertex collision in-cell pair : %d %d\n",stI,svi);
                        collide = true;
                    }

                    if(!collide)
                        return;

                    // now there is collision, build the "collision tets"
                    // if(!vtemp.hasProperty("oneRingArea"))
                    //     printf("vtemp has no oneRingArea");

                    cptemp.template tuple<4>("inds",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = zs::vec<int,4>(vi,tri[0],tri[1],tri[2]).template reinterpret_bits<T>();

                    auto vertexFaceCollisionAreas = restArea + points("area",svi);
                    
                    vec3 collision_verts[4] = {};
                    collision_verts[0] = p;
                    collision_verts[1] = t0;
                    collision_verts[1] = t1;
                    collision_verts[1] = t2;

                    auto collisionEps = in_collisionEps;

                    auto grad = collisionStiffness * VERTEX_FACE_SQRT_COLLISION::gradient(collision_verts,mu,lambda,collisionEps) * vertexFaceCollisionAreas;
                    auto hessian = collisionStiffness * VERTEX_FACE_SQRT_COLLISION::hessian(collision_verts,mu,lambda,collisionEps) * vertexFaceCollisionAreas;
                    cptemp.template tuple<12*12>("H",svi * MAX_FP_COLLISION_PAIRS + nm_collision_pairs) = hessian;

                    for(int i = 0;i != 4;++i) {
                        auto g_vi = i == 0 ? vi : tri[i-1];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp("grad", d, g_vi), grad(i * 3 + d));
                    }
                    nm_collision_pairs++;

                };
                stbvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
            });

#else

        COLLISION_UTILS::do_facet_point_collision_detection<MAX_FP_COLLISION_PAIRS>(cudaPol,
            vtemp,"xn",
            points,
            lines,
            tris,
            sttemp,
            setemp,
            cptemp,
            // stBvh,
            in_collisionEps,out_collisionEps);


        // output all the collision pairs
        // cudaPol(zs::range(cptemp.size()),
        //     [cptemp = proxy<space>({},cptemp)] ZS_LAMBDA(int cpi) mutable {
        //         auto inds = cptemp.template pack<4>("inds",cpi).reinterpret_bits(int_c);
        //         bool collide = true;
        //         for(int i = 0;i != 4;++i)
        //             if(inds[i] < 0)
        //                 collide = false;
        //         if(collide)
        //             printf("collision_pair[%d] : %d %d %d %d\n",
        //                 cpi,inds[0],inds[1],inds[2],inds[3]);
        // });

        COLLISION_UTILS::evaluate_collision_grad_and_hessian<MAX_FP_COLLISION_PAIRS>(cudaPol,
            vtemp,"xn",
            cptemp,
            in_collisionEps,out_collisionEps,
            (T)collisionStiffness,
            (T)mu,(T)lambda);



        // project out all the neglect verts
        if(neglect_inverted) {
            cudaPol(zs::range(cptemp.size()),
                [cptemp = proxy<space>({},cptemp),vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int cpi) {
                    auto inds = cptemp.template pack<4>("inds",cpi).reinterpret_bits(int_c);
                    for(int i = 0;i != 4;++i)
                        if(inds[i] < 0)
                            return;

                    bool is_inverted = false;
                    for(int i = 0;i != 4;++i){
                        auto vi = inds[i];
                        auto is_vertex_inverted = reinterpret_bits<int>(vtemp("is_inverted",vi));
                        if(is_vertex_inverted)
                            is_inverted = true;
                    }

                    if(is_inverted){
                        cptemp.template tuple<12*12>("H",cpi) = zs::vec<T,12,12>::zeros();
                        cptemp.template tuple<12>("grad",cpi) = zs::vec<T,12>::zeros();
                    }
            });    
        }

        // auto gradN = TILEVEC_OPS::inf_norm<12>(cudaPol,cptemp,"grad");
        // fmt::print(fg(fmt::color::red),"collision gradN = {}\n",gradN);
        // TILEVEC_OPS::fill<12*12>(cudaPol,cptemp,"H",zs::vec<T,12*12>::zeros());

        TILEVEC_OPS::assemble<3,4>(cudaPol,cptemp,"grad",vtemp,"grad");


#endif


        }


        template <typename Model>
        void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                            const Model& model,
                            dtiles_t& vtemp,
                            dtiles_t& etemp) {        
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            #if DEBUG_FLESH_DYN_STEPPING
                // std::cout << "CHECK THE PROPERTY CHANNEL" << std::endl;
                if(!vtemp.hasProperty("grad"))
                    fmt::print(fg(fmt::color::red),"the vtemp has no 'grad' channel\n");
                if(!vtemp.hasProperty("xn"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'xn' channel\n");
                if(!vtemp.hasProperty("xp"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'xp' channel\n");
                if(!vtemp.hasProperty("vp"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'vp' channel\n");

                if(!etemp.hasProperty("H"))
                    fmt::print(fg(fmt::color::red),"the etemp has no 'H' channel\n");
                if(!etemp.hasProperty("ActInv"))
                    fmt::print(fg(fmt::color::red),"the etemp has no 'ActInv' channel\n");
                
                if(!verts.hasProperty("m"))
                    fmt::print(fg(fmt::color::red),"the verts has no 'm' channel\n");

                if(!eles.hasProperty("IB"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'IB' channel\n");
                if(!eles.hasProperty("m"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'm' channel\n");
                if(!eles.hasProperty("vol"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'vol' channel\n");
                if(!eles.hasProperty("inds"))
                    fmt::print(fg(fmt::color::red),"the eles has no 'inds' channel\n");
            #endif

            TILEVEC_OPS::fill<3>(cudaPol,vtemp,"grad",zs::vec<T,3>::zeros());
            TILEVEC_OPS::fill<144>(cudaPol,etemp,"H",zs::vec<T,144>::zeros());         
            
            // eval the inertia term gradient
            cudaPol(zs::range(vtemp.size()), [dt2 = dt2,
                        vtemp = proxy<space>({},vtemp),
                        verts = proxy<space>({},verts),
                        dt = dt] ZS_LAMBDA(int vi) mutable {
                auto m = verts("m",vi);// nodal mass
                auto x1 = vtemp.pack<3>("xn",vi);
                auto x0 = vtemp.pack<3>("xp",vi);
                auto v0 = vtemp.pack<3>("vp",vi);
                vtemp.template tuple<3>("grad",vi) = -m * (x1 - x0 - v0 * dt) / dt2;                    
            });

            cudaPol(zs::range(eles.size()), [this,dt2 = dt2,
                            vtemp = proxy<space>({}, vtemp),
                            etemp = proxy<space>({}, etemp),
                            bcws = proxy<space>({},b_bcws),
                            b_verts = proxy<space>({},b_verts),
                            verts = proxy<space>({}, verts),
                            eles = proxy<space>({}, eles),
                            model, volf = volf] ZS_LAMBDA (int ei) mutable {
                    auto DmInv = eles.template pack<3,3>("IB",ei);
                    auto dFdX = dFdXMatrix(DmInv);
                    auto inds = eles.template pack<4>("inds",ei).reinterpret_bits(int_c);
                    vec3 x1[4] = {vtemp.template pack<3>("xn", inds[0]),
                            vtemp.template pack<3>("xn", inds[1]),
                            vtemp.template pack<3>("xn", inds[2]),
                            vtemp.template pack<3>("xn", inds[3])};   

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
                    auto P = model.first_piola(FAct);
                    auto vole = eles("vol", ei);
                    auto vecP = flatten(P);
                    vecP = dFActdF.transpose() * vecP;
                    auto dFdXT = dFdX.transpose();
                    auto vf = -vole * (dFdXT * vecP);     

                    auto mg = volf * vole / 4;
                    for (int i = 0; i != 4; ++i) {
                        auto vi = inds[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp("grad", d, vi), vf(i * 3 + d) + mg(d));
                    }

                    // assemble element-wise hessian matrix
                    auto Hq = model.first_piola_derivative(FAct, true_c);
                    auto dFdAct_dFdX = dFActdF * dFdX; 
                    // dFdAct_dFdX = dFdX; 
                    auto H = dFdAct_dFdX.transpose() * Hq * dFdAct_dFdX * vole;
                    etemp.template tuple<12 * 12>("H", ei) = H;

                    // add inertia hessian term
                    auto m = eles("m",ei);// element-wise mass
                    for(int i = 0;i < 12;++i){
                        // Mass(i,i) = 1;
                        etemp("H",i * 12 + i,ei) += m /dt2/4;
                    }


            });
        // Bone Driven Potential Energy
            T lambda = model.lam;
            T mu = model.mu;
            auto nmEmbedVerts = b_verts.size();
            cudaPol(zs::range(nmEmbedVerts), [this,
                    bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    eles = proxy<space>({},eles),lambda,mu,bone_driven_weight = bone_driven_weight] ZS_LAMBDA(int vi) mutable {
                        auto ei = reinterpret_bits<int>(bcws("inds",vi));
                        if(ei < 0)
                            return;
                        auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                        auto w = bcws.pack<4>("w",vi);
                        auto tpos = vec3::zeros();
                        for(size_t i = 0;i != 4;++i)
                            tpos += w[i] * vtemp.pack<3>("xn",inds[i]);
                        auto pdiff = tpos - b_verts.pack<3>("x",vi);

                        T stiffness = 2.0066 * mu + 1.0122 * lambda;

                        for(size_t i = 0;i != 4;++i){
                            auto tmp = pdiff * (-stiffness * bcws("cnorm",vi) * bone_driven_weight * w[i] * eles("vol",ei)); 
                            // tmp = pdiff * (-lambda * bcws("cnorm",vi) * bone_driven_weight * w[i]);
                            for(size_t d = 0;d != 3;++d)
                                atomic_add(exec_cuda,&vtemp("grad",d,inds[i]),(T)tmp[d]);
                        }
                        for(int i = 0;i != 4;++i)
                            for(int j = 0;j != 4;++j){
                                T alpha = stiffness * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi) * eles("vol",ei);
                                for(int d = 0;d != 3;++d){
                                    atomic_add(exec_cuda,&etemp("H",(i * 3 + d) * 12 + j * 3 + d,ei),alpha);
                                }
                            }

            });

        }


        FEMDynamicSteppingSystem(const tiles_t &verts, const tiles_t &eles,
                const tiles_t& points,const tiles_t& lines,const tiles_t& tris,
                T in_collisionEps,T out_collisionEps,
                const tiles_t &b_bcws, const tiles_t& b_verts,T bone_driven_weight,
                vec3 volf,const T& _dt,const T& collisionStiffness)
            : verts{verts}, eles{eles},points{points}, lines{lines}, tris{tris},
                    in_collisionEps{in_collisionEps},out_collisionEps{out_collisionEps},
                    b_bcws{b_bcws}, b_verts{b_verts}, bone_driven_weight{bone_driven_weight},
                    volf{volf},
                    dt{_dt}, dt2{dt * dt},collisionStiffness{collisionStiffness},use_edge_edge_collision{true}, use_vertex_facet_collision{true} {}

        const tiles_t &verts;
        const tiles_t &eles;
        const tiles_t &points;
        const tiles_t &lines;
        const tiles_t &tris;
        const tiles_t &b_bcws;  // the barycentric interpolation of embeded bones 
        const tiles_t &b_verts; // the position of embeded bones

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
            gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec<3,T>>();
        T armijo = (T)1e-4;
        T wolfe = (T)0.9;
        // T cg_res = (T)0.001;
        T cg_res = (T)0.0001;
        T btl_res = (T)0.1;
        auto models = zsparticles->getModel();
        auto& verts = zsparticles->getParticles();
        auto& eles = zsparticles->getQuadraturePoints();

        if(eles.getChannelSize("inds") != 4)
            throw std::runtime_error("the input zsparticles is not a tetrahedra mesh");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
            throw std::runtime_error("the input zsparticles has no surface tris");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
            throw std::runtime_error("the input zsparticles has no surface lines");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) 
            throw std::runtime_error("the input zsparticles has no surface points");
        // if(!zsparticles->hasBvh(ZenoParticles::s_surfTriTag)) {
        //     throw std::runtime_error("the input zsparticles has no surface tris's spacial structure");
        // }
        // if(!zsparticles->hasBvh(ZenoParticles::s_surfEdgeTag)) {
        //     throw std::runtime_error("the input zsparticles has no surface edge's spacial structure");
        // }
        // if(!zsparticles->hasBvh(ZenoParticles::s_surfVertTag))  {
        //     throw std::runtime_error("the input zsparticles has no surface vert's spacial structure");
        // }

        auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
        auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

        // auto& stBvh = zsparticles->bvh(ZenoParticles::s_surfTriTag);
        // auto& seBvh = zsparticles->bvh(ZenoParticles::s_surfEdgeTag);


        auto zsbones = get_input<ZenoParticles>("driven_boudary");
        auto driven_tag = get_input2<std::string>("driven_tag");
        auto bone_driven_weight = get_input2<float>("driven_weight");
        auto muscle_id_tag = get_input2<std::string>("muscle_id_tag");
        // auto bone_driven_weight = (T)0.02;



        auto newton_res = (T)0.01;

        auto dt = get_input2<float>("dt");

        auto volf = vec3::from_array(gravity * models.density);

        std::vector<float> act_;    
        std::size_t nm_acts = 0;

        if(has_input("Acts")) {
            act_ = get_input<zeno::ListObject>("Acts")->getLiterial<float>();
            nm_acts = act_.size();
        }

        constexpr auto host_space = zs::execspace_e::openmp;
        auto ompExec = zs::omp_exec();
        auto act_buffer = dtiles_t{{{"act",1}},nm_acts,zs::memsrc_e::host};
        ompExec(zs::range(act_buffer.size()),
            [act_buffer = proxy<host_space>({},act_buffer),act_] (int i) mutable {
                act_buffer("act",i) = act_[i];
        });
        act_buffer = act_buffer.clone({zs::memsrc_e::device, 0});

        // the temp buffer only store the data that will change every iterations or every frame
        static dtiles_t vtemp{verts.get_allocator(),
                            {
                                {"grad", 3},
                                {"P", 9},
                                {"bou_tag",1},
                                {"dir", 3},
                                {"xn", 3},
                                {"xp",3},
                                {"vp",3},
                                {"is_inverted",1},
                                {"active",1}
                            },verts.size()};

        // auto max_collision_pairs = tris.size() / 10; 
        static dtiles_t etemp{eles.get_allocator(), {
                {"H", 12 * 12},
                {"inds",4},
                {"ActInv",3*3},
                // {"muscle_ID",1},
                // {"fiber",3}
                }, eles.size()};

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

        static dtiles_t cptemp(points.get_allocator(),{
            {"inds",4},
            {"area",1},
            {"grad",12},
            {"inverted",1},
            {"H",12 * 12}
        },points.size() * MAX_FP_COLLISION_PAIRS);


        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec().sync(false);
    

        // TILEVEC_OPS::fill<4>(cudaPol,etemp,"inds",zs::vec<int,4>::uniform(-1).template reinterpret_bits<T>())
        TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",etemp,"inds");

        auto avgl = compute_average_edge_length(cudaPol,verts,"x",tris);
        // auto avgl = (T)1.0;

        auto collisionStiffness = get_input2<float>("cstiffness");


        // auto inset_ratio = get_input2<float>("collision_inset");
        // auto outset_ratio = get_input2<float>("collision_outset");    

        auto in_collisionEps = get_input2<float>("in_collisionEps");
        auto out_collisionEps = get_input2<float>("out_collisionEps");

        FEMDynamicSteppingSystem A{
            verts,eles,
            points,lines,tris,
            (T)in_collisionEps,(T)out_collisionEps,
            (*zsparticles)[driven_tag],zsbones->getParticles(),bone_driven_weight,
            volf,dt,collisionStiffness};


        // TILEVEC_OPS::fill<9>(cudaPol,etemp,"ActInv",zs::vec<T,9>{1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0});
        // apply muscle activation
        cudaPol(zs::range(eles.size()),
            [etemp = proxy<space>({},etemp),eles = proxy<space>({},eles),
                act_buffer = proxy<space>({},act_buffer),muscle_id_tag = SmallString(muscle_id_tag),nm_acts,avgl] ZS_LAMBDA(int ei) mutable {
                // auto act = eles.template pack<3>("act",ei);
                // auto fiber = etemp.template pack<3>("fiber",ei);
                zs::vec<T,3> fiber{};
                if(!eles.hasProperty("fiber"))
                    fiber = eles.template pack<3>("fiber",ei);
                else 
                    fiber = zs::vec<T,3>(1.0,0.0,0.0);
                vec3 act{1.0,1.0,1.0};


                auto nfiber = fiber.norm();
                // auto ID = etemp("muscle_ID",ei);
                int ID = -1;
                if(eles.hasProperty(muscle_id_tag))
                    ID = (int)eles(muscle_id_tag,ei);
                
                if(nm_acts > 0 && ID > -1){
                    float a = 1. - act_buffer("act",ID);
                    act = vec3{1,zs::sqrt(1./a),zs::sqrt(1./a)};
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

                // if(ei == 0) {
                //     printf("Act[0]:\n%f %f %f\n%f %f %f\n%f %f %f\n",
                //         (float)Act(0,0),(float)Act(0,1),(float)Act(0,2),
                //         (float)Act(1,0),(float)Act(1,1),(float)Act(1,2),
                //         (float)Act(2,0),(float)Act(2,1),(float)Act(2,2));
                // }

                etemp.template tuple<9>("ActInv",ei) = zs::inverse(Act);
        });
        // std::cout << "set initial guess" << std::endl;
        // setup initial guess
        TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"xp");
        TILEVEC_OPS::copy<3>(cudaPol,verts,"v",vtemp,"vp");
        TILEVEC_OPS::copy(cudaPol,verts,"active",vtemp,"active");
        if(verts.hasProperty("init_x"))
            TILEVEC_OPS::copy<3>(cudaPol,verts,"init_x",vtemp,"xn");   
        else {
            // TILEVEC_OPS::add<3>(cudaPol,vtemp,"xp",1.0,"vp",dt,"xn");  
            TILEVEC_OPS::add<3>(cudaPol,vtemp,"xp",1.0,"vp",(T)0.0,"xn");  
        }
        TILEVEC_OPS::fill<1>(cudaPol,vtemp,"bou_tag",zs::vec<T,1>::zeros());


        auto bvh_thickness = 5 * avgl;

        int max_newton_iterations = 5;
        int nm_iters = 0;

        while(nm_iters < max_newton_iterations) {


            match([&](auto &elasticModel) {
                A.computeGradientAndHessian(cudaPol, elasticModel,vtemp,etemp);
            })(models.getElasticModel());

            bool include_collision = true;
            if(include_collision) {

                // if(!calculate_facet_normal(cudaPol,vtemp,"xn",tris,sttemp,"nrm")){
                //         throw std::runtime_error("fail updating facet normal");
                // }

                // if(!COLLISION_UTILS::calculate_cell_bisector_normal(cudaPol,
                //     vtemp,"xn",
                //     lines,
                //     tris,
                //     sttemp,"nrm",
                //     setemp,"nrm")){
                //         throw std::runtime_error("fail calculate cell bisector normal");
                // }

                // auto stbvs = retrieve_bounding_volumes(cudaPol,vtemp,tris,wrapv<3>{},(T)0.0,"xn");
                // auto sebvs = retrieve_bounding_volumes(cudaPol,vtemp,lines,wrapv<2>{},(T)0.0,"xn");
                // stBvh.refit(cudaPol,stbvs);
                // seBvh.refit(cudaPol,sebvs);

                match([&](auto &elasticModel) {
                    A.computeCollisionGradientAndHessian(cudaPol,elasticModel,
                        vtemp,
                        etemp,
                        sttemp,
                        setemp,
                        cptemp,
                        // stBvh,
                        // seBvh,
                        bvh_thickness);
                })(models.getElasticModel());

            }

            PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"H",etemp,cptemp,"P",vtemp);
            // PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"H",etemp,"P",vtemp);
            // if the grad is too small, return the result
            // Solve equation using PCG
            TILEVEC_OPS::fill<3>(cudaPol,vtemp,"dir",zs::vec<T,3>::zeros());
            // std::cout << "solve using pcg" << std::endl;
            PCG::pcg_with_fixed_sol_solve<3,4>(cudaPol,vtemp,etemp,cptemp,"dir","bou_tag","grad","P","inds","H",cg_res,1000,50);
            // PCG::pcg_with_fixed_sol_solve<3,4>(cudaPol,vtemp,etemp,"dir","bou_tag","grad","P","inds","H",cg_res,1000,50);
            // std::cout << "finish solve pcg" << std::endl;
            PCG::project<3>(cudaPol,vtemp,"dir","bou_tag");
            T alpha = 1.;
            cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),alpha] __device__(int i) mutable {
                vtemp.tuple<3>("xn", i) =
                    vtemp.pack<3>("xn", i) + alpha * vtemp.pack<3>("dir", i);
            });


            cudaPol(zs::range(verts.size()),
                    [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),dt] __device__(int vi) mutable {
                        auto newX = vtemp.pack<3>("xn", vi);
                        verts.tuple<3>("x", vi) = newX;
                        verts.tuple<3>("v",vi) = (vtemp.pack<3>("xn",vi) - vtemp.pack<3>("xp",vi))/dt;
                    });

            T res = TILEVEC_OPS::inf_norm<3>(cudaPol, vtemp, "dir");// this norm is independent of descriterization
            std::cout << "res[" << nm_iters << "] : " << res << std::endl;
            if(res < 1e-3)
                break;
            nm_iters++;
        }



        dtiles_t nodalForceVis(verts.get_allocator(),
            {
                {"x",3},
                {"dir",3},
            },verts.size());


        cudaPol.syncCtx();
        TILEVEC_OPS::copy<3>(cudaPol,vtemp,"xn",nodalForceVis,"x");
        TILEVEC_OPS::fill<3>(cudaPol,nodalForceVis,"dir",zs::vec<T,3>::zeros());
        TILEVEC_OPS::assemble<3,4>(cudaPol,cptemp,"grad",nodalForceVis,"dir");





        set_output("ZSParticles", zsparticles);
    }


};

ZENDEFNODE(FleshDynamicStepping, {{"ZSParticles",
                                    "gravity","Acts",
                                    "driven_boudary",
                                    {"string","driven_tag","bone_bw"},
                                    {"float","driven_weight","0.02"},
                                    {"string","muscle_id_tag","ms_id_tag"},
                                    {"float","cstiffness","0.0"},
                                    {"float","in_collisionEps","0.01"},
                                    {"float","out_collisionEps","0.01"},
                                    {"float","dt","0.5"}
                                    },
                                  {"ZSParticles"},
                                  {
                                  },
                                  {"FEM"}});



};