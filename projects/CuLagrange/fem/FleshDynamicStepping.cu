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

#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"

#include "collision_energy/vertex_face_collision.hpp"
#include "collision_energy/vertex_face_sqrt_collision.hpp"
#include "collision_energy/edge_edge_collision.hpp"
#include "collision_energy/edge_edge_sqrt_collition.hpp"

#define DEBUG_FLESH_DYN_STEPPING 1

namespace zeno {

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

        void computeInvertedVertex(zs::CudaExecutionPolicy& cudaPol,
                            const zs::SmallString& x_tag,
                            const zs::SmallString& inversion_tag,
                            dtiles_t& vtemp,
                            const tiles_t& quads) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            // compute the inverted vertices
            TILEVEC_OPS::fill(cudaPol,vtemp,inversion_tag,(T)0);
            cudaPol(zs::range(quads.size()),
                [vtemp = proxy<space>({},vtemp),quads = proxy<space>({},quads),x_tag,inversion_tag] ZS_LAMBDA(int ei) mutable {
                    auto DmInv = quads.template pack<3,3>("IB",ei);
                    auto inds = quads.template pack<4>("inds",ei).template reinterpret_bits<int>();
                    vec3 x1[4] = {vtemp.template pack<3>(x_tag, inds[0]),
                            vtemp.template pack<3>(x_tag, inds[1]),
                            vtemp.template pack<3>(x_tag, inds[2]),
                            vtemp.template pack<3>(x_tag, inds[3])};   

                    mat3 F{};
                    {
                        auto x1x0 = x1[1] - x1[0];
                        auto x2x0 = x1[2] - x1[0];
                        auto x3x0 = x1[3] - x1[0];
                        auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                                        x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                        F = Ds * DmInv;
                    } 
                    T inversion = zs::determinant(F) > (T)0 ? (T) 0 : (T) -1;
                    if(inversion > 0)
                        for(int i = 0;i < 4;++i)
                            vtemp(inversion_tag,inds[i]) = (T)1;                  
            });
        }



        T computeCollisionEnergy(zs::CudaExecutionPolicy& cudaPol,
                            const zs::SmallString& x_tag,
                            const zs::SmallString& inversion_tag,
                            dtiles_t& vtemp,
                            const tiles_t& points,
                            const tiles_t& tris,
                            const tiles_t& quads,
                            const T& thick_ness) {
                                return 0;
                            }
        template <typename Model>
        void computeCollisionGradientHessian(zs::CudaExecutionPolicy& cudaPol,
                            const T& lambda,
                            const T& mu,
                            const zs::SmallString& x_tag,
                            const zs::SmallString& inversion_tag,
                            const zs::SmallString& grad_tag,
                            const zs::SmallString& bisector_normal_tag,
                            dtiles_t& vtemp,
                            const tiles_t& points,
                            const tiles_t& lines,
                            const tiles_t& tris,
                            const zs::SmallString& hessian_tag,
                            dtiles_t& etemp,
                            const T& thickness) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            // compute vertex facet contact pairs
            cudaPol(zs::range(points.size()),[lambda = lambda,mu = mu,
                            collisionEps = collisionEps,
                            x_tag,inversion_tag,vtemp = proxy<space>({},vtemp),
                            hessian_tag,etemp = proxy<space>({},etemp),
                            points = proxy<space>({},points),
                            lines = proxy<space>({},lines),
                            tris = proxy<space>({},tris),
                            bisector_normal_tag,
                            stbvh = proxy<space>(stBvh),thickness] ZS_LAMBDA(int svi) mutable {
                auto vi = reinterpret_bits<int>(points("inds",svi));

                auto inversion = vtemp(inversion_tag,vi);
                if(inversion > 0)
                    return;

                auto p = vtemp.template pack<3>(x_tag,vi);
                auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};

                // check whether there is collision happening, and if so, apply the collision force and addup the collision hessian
                auto process_vertex_face_collision_pairs = [&](int stI) {
                    auto tri = tris.template pack<3>("inds",stI).reinterpret_bits<int>();
                    if(tri[0] == vi || tri[1] == vi || tri[2] == vi)
                        return;

                    auto t0 = vtemp.template pack<3>(x_tag,tri[0]);
                    auto t1 = vtemp.template pack<3>(x_tag,tri[1]);
                    auto t2 = vtemp.template pack<3>(x_tag,tri[2]);
                    // check whether the triangle is degenerate
                    auto restArea = tris("area",stI);
                    // skip the triangle too small at rest configuration
                    if(restArea < (T)1e-6)
                        return;

                    const auto e10 = t1 - t0;
                    const auto e20 = t2 - t0;
                    auto deformedArea = (T)0.5 * e10.cross(e20).norm();
                    const T degeneracyEps = 1e-4;
                    // skip the degenerate triangles
                    const T relativeArea = deformedArea / (restArea + (T)1e-6);
                    if(relativeArea < degeneracyEps)
                        return;

                    const T distance = COLLISION_UTILS::pointTriangleDistance(t0,t1,t2,p);

                    bool collide = false;

                    if (distance < collisionEps){
                        // if the point, projected onto the face's plane, is inside the face,
                        // then record the collision now
                        if (COLLISION_UTILS::pointProjectsInsideTriangle(t0, t1, t2, p))
                            collide = true;
                        else if(is_inside_the_cell(vtemp,lines,tris,bisector_normal_tag,x_tag,stI,p,collisionEps,collisionEps))
                            collide = true;
                            // check whether the point is inside the cell
                    //         vec3 neigh_normals[3];
                    //         auto plane_normal = e10.cross(e20).normalized();
                    //         auto neighbors = tris.template pack<3>("neigh_inds",stI).template reinterpret_bits<int>();

                    //         vec3 nt[3];
                    //         // get the normals of the three adjacent faces
                    //         for(int i = 0;i < 3;++i) {
                    //             auto ntri = tris.template pack<3>("inds",neighbors[i]).reinterpret_bits<int>();
                    //             nt[0] = vtemp.template pack<3>(x_tag,ntri[0]);
                    //             nt[1] = vtemp.template pack<3>(x_tag,ntri[1]);
                    //             nt[2] = vtemp.template pack<3>(x_tag,ntri[2]);       

                    //             auto ne10 = nt[1] - nt[0];
                    //             auto ne20 = nt[2] - nt[0];

                    //             neigh_normals[i] = ne10.cross(ne20).normalized();                      
                    //         }
                    //         // there is some consistent of neigh_inds and inds here
                    //         collide = true;
                    //         for(int i = 0;i < 3;++i) {
                    //             auto ne = (neigh_normals[i] + plane_normal).normalized();
                    //             auto eij = nt[(i + 1) % 3] - nt[i];
                    //             auto neb = ne.cross(eij);
                    //             auto nebHat = neb.normalized();
                    //             auto deplane = nebHat.dot(p - nt[i]);

                    //             if(deplane < 0.0) {
                    //                 collide = false;
                    //                 break;
                    //             }
                    //         }
                    //     }
                    }

                    if(!collide)
                        return;

                    // now there is collision, build the "collision tets"
                    if(!vtemp.hasProperty("oneRingArea"))
                        printf("vtemp has no oneRingArea");

                    auto vertexFaceCollisionAreas = restArea + vtemp("oneRingArea",vi);
                    
                    std::vector<vec3> collision_verts(4);
                    collision_verts[0] = p;
                    collision_verts[1] = t0;
                    collision_verts[1] = t1;
                    collision_verts[1] = t2;

                    auto grad = VERTEX_FACE_SQRT_COLLISION::gradient(collision_verts,mu,lambda,collisionEps) * vertexFaceCollisionAreas;
                    auto hessian = VERTEX_FACE_SQRT_COLLISION::hessian(collision_verts,mu,lambda,collisionEps) * vertexFaceCollisionAreas;

                    etemp.template tuple<12*12>(hessian_tag,stI) = etemp.template pack<12,12>(hessian_tag,stI) + hessian;

                    for(int i = 0;i != 4;++i) {
                        auto g_vi = i == 0 ? vi : tri[i-1];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp("grad", d, g_vi), grad(i * 3 + d));
                    }

                };
                stbvh.iter_neighbors(bv,process_vertex_face_collision_pairs);
            });
        }

        // void computeCollisionDetection(zs::CudaExecutionPolicy& cudaPol,
        //                     const zs::SmallString& p1_tag,
        //                     const zs::SmallString& inversion_tag,
        //                     dtiles_t& vtemp,
        //                     const tiles_t& tris,
        //                     const tiles_t& lines,
        //                     const tiles_t& points,
        //                     const tiles_t& quads,
        //                     const T& thickness) {
        //     using namespace zs;
        //     constexpr auto space = execspace_e::cuda;
        //     #if DEBUG_FLESH_DYN_STEPPING
        //         if(!vtemp.hasProperty(p1_tag))
        //             fmt::print(fg(fmt::color::red),"the verts has no '{}' channel\n",p1_tag.asString());
        //     #endif

        //     auto triBvs = retrieve_bounding_volumes(cudaPol,vtemp,tris,zs::wrapv<3>{},thickness,p1_tag);
        //     stBvh.refit(cudaPol,triBvs);
        //     auto edgeBvs = retrieve_bounding_volumes(cudaPol,vtemp,lines,zs::wrapv<2>{},thickness,p1_tag);
        //     seBvh.refit(cudaPol,edgeBvs);

        //     computeInvertedVertex(cudaPol,p1_tag,inversion_tag,vtemp,quads);
        //     // if(use_vertex_facet_collision)
        //     // computeVertexFaceCollisions(cudaPol,p1_tag,inversion_tag,vtemp,quads);
        // }

        template <typename Model>
        void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                            const Model& model,
                            const zs::SmallString& p1_tag,
                            const zs::SmallString& p0_tag,
                            const zs::SmallString& v0_tag,
                            dtiles_t& vtemp,
                            const zs::SmallString& H_tag,
                            dtiles_t& etemp) {        
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            #if DEBUG_FLESH_DYN_STEPPING
                // std::cout << "CHECK THE PROPERTY CHANNEL" << std::endl;
                if(!vtemp.hasProperty("grad"))
                    fmt::print(fg(fmt::color::red),"the vtemp has no 'grad' channel\n");
                if(!vtemp.hasProperty(p1_tag))
                    fmt::print(fg(fmt::color::red),"the verts has no '{}' channel\n",p1_tag.asString());
                if(!vtemp.hasProperty(p0_tag))
                    fmt::print(fg(fmt::color::red),"the verts has no '{}' channel\n",p0_tag.asString());
                if(!vtemp.hasProperty(v0_tag))
                    fmt::print(fg(fmt::color::red),"the verts has no '{}' channel\n",v0_tag.asString());

                if(!etemp.hasProperty(H_tag))
                    fmt::print(fg(fmt::color::red),"the etemp has no '{}' channel\n",H_tag.asString());
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
            #endif

            TILEVEC_OPS::fill<3>(cudaPol,vtemp,"grad",zs::vec<T,3>::zeros());
            TILEVEC_OPS::fill<144>(cudaPol,etemp,H_tag,zs::vec<T,144>::zeros());         
            
            // eval the inertia term gradient
            cudaPol(zs::range(vtemp.size()), [dt2 = dt2,   
                        vtemp = proxy<space>({},vtemp),
                        verts = proxy<space>({},verts),
                        dt = dt,p1_tag,p0_tag,v0_tag] ZS_LAMBDA(int vi) mutable {
                auto m = verts("m",vi);// nodal mass
                auto x1 = vtemp.pack<3>(p1_tag,vi);
                auto x0 = vtemp.pack<3>(p0_tag,vi);
                auto v0 = vtemp.pack<3>(v0_tag,vi);

                vtemp.template tuple<3>("grad",vi) = m * (x1 - x0 - v0 * dt) / dt2;                    
            });

            cudaPol(zs::range(eles.size()), [this,dt2 = dt2,
                            vtemp = proxy<space>({}, vtemp),
                            etemp = proxy<space>({}, etemp),
                            bcws = proxy<space>({},b_bcws),
                            b_verts = proxy<space>({},b_verts),
                            verts = proxy<space>({}, verts),
                            eles = proxy<space>({}, eles),p1_tag,p0_tag,v0_tag,H_tag,
                            model, volf = volf] ZS_LAMBDA (int ei) mutable {
                    auto DmInv = eles.template pack<3,3>("IB",ei);
                    auto dFdX = dFdXMatrix(DmInv);
                    auto inds = eles.template pack<4>("inds",ei).template reinterpret_bits<int>();
                    vec3 x1[4] = {vtemp.template pack<3>(p1_tag, inds[0]),
                            vtemp.template pack<3>(p1_tag, inds[1]),
                            vtemp.template pack<3>(p1_tag, inds[2]),
                            vtemp.template pack<3>(p1_tag, inds[3])};   

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
                    etemp.template tuple<12 * 12>(H_tag, ei) = H;

                    // add inertia hessian term
                    auto m = eles("m",ei);// element-wise mass
                    for(int i = 0;i < 12;++i){
                        // Mass(i,i) = 1;
                        etemp(H_tag,i * 12 + i,ei) += m /dt2/4;
                    }


            });
        // Bone Driven Potential Energy
            T lambda = model.lam;
            T mu = model.mu;
            auto nmEmbedVerts = b_verts.size();
            cudaPol(zs::range(nmEmbedVerts), [this,
                    bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    eles = proxy<space>({},eles),lambda,mu,p1_tag,H_tag,bone_driven_weight = bone_driven_weight] ZS_LAMBDA(int vi) mutable {
                        auto ei = reinterpret_bits<int>(bcws("inds",vi));
                        if(ei < 0)
                            return;
                        auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                        auto w = bcws.pack<4>("w",vi);
                        auto tpos = vec3::zeros();
                        for(size_t i = 0;i != 4;++i)
                            tpos += w[i] * vtemp.pack<3>(p1_tag,inds[i]);
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
                                    atomic_add(exec_cuda,&etemp(H_tag,(i * 3 + d) * 12 + j * 3 + d,ei),alpha);
                                }
                            }

            });

        }


        FEMDynamicSteppingSystem(const tiles_t &verts, const tiles_t &eles, const tiles_t &b_bcws, const tiles_t& b_verts,T bone_driven_weight,vec3 volf,const T& _dt)
            : verts{verts}, eles{eles}, b_bcws{b_bcws}, b_verts{b_verts}, bone_driven_weight{bone_driven_weight},volf{volf},dt{_dt}, dt2{dt * dt}, use_edge_edge_collision{true}, use_vertex_facet_collision{true} {}

        const tiles_t &verts;
        const tiles_t &eles;
        const tiles_t &b_bcws;  // the barycentric interpolation of embeded bones 
        const tiles_t &b_verts; // the position of embeded bones

        T bone_driven_weight;
        vec3 volf;
        T dt;
        T dt2;
        T collisionEps;

        T collisionStiffness;

        bvh_t stBvh, seBvh;  

        bool bvh_initialized;
        bool use_edge_edge_collision;
        bool use_vertex_facet_collision;
    };



    void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("ZSParticles");
        auto gravity = zeno::vec<3,T>(0);
        if(has_input("gravity"))
            gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec<3,T>>();
        T armijo = (T)1e-4;
        T wolfe = (T)0.9;
        T cg_res = (T)0.01;
        T btl_res = (T)0.1;
        auto models = zstets->getModel();
        auto& verts = zstets->getParticles();
        auto& eles = zstets->getQuadraturePoints();
        auto zsbones = get_input<ZenoParticles>("driven_boudary");
        auto tag = get_param<std::string>("driven_tag");
        auto muscle_id_tag = get_param<std::string>("muscle_id_tag");
        auto bone_driven_weight = (T)1.0;
        auto newton_res = (T)0.01;

        auto dt = get_param<float>("dt");

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
            [act_buffer = proxy<host_space>({},act_buffer),act_] (int i) mutable{
                act_buffer("act",i) = act_[i];
        });
        act_buffer = act_buffer.clone({zs::memsrc_e::device, 0});

        static dtiles_t vtemp{verts.get_allocator(),
                            {{"grad", 3},
                            {"P", 9},
                            {"bou_tag",1},
                            {"dir", 3},
                            {"xn", 3},
                            {"xp",3},
                            {"vp",3}},
                            verts.size()};
        static dtiles_t etemp{eles.get_allocator(), {{"H", 12 * 12},{"inds",4},{"ActInv",3*3},{"muscle_ID",1},{"fiber",3}}, eles.size()};
        vtemp.resize(verts.size());
        etemp.resize(eles.size());

        FEMDynamicSteppingSystem A{verts,eles,(*zstets)[tag],zsbones->getParticles(),bone_driven_weight,volf,dt};

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec().sync(false);

        TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",etemp,"inds");


        if(!eles.hasProperty("fiber")){
            TILEVEC_OPS::fill<3>(cudaPol,etemp,"fiber",{1.,0.,0.});
        }else {
        if(eles.getChannelSize("fiber") != 3){
            fmt::print("The input fiber  has wrong channel size\n");
            throw std::runtime_error("The input fiber has wrong channel size");
        }
            TILEVEC_OPS::copy<3>(cudaPol,eles,"fiber",etemp,"fiber");
        }
        if(!eles.hasProperty(muscle_id_tag)) {
            TILEVEC_OPS::fill(cudaPol,etemp,"muscle_ID",-1);
        }else {
            TILEVEC_OPS::copy(cudaPol,eles,muscle_id_tag,etemp,"muscle_ID");
        }

        // apply muscle activation
        cudaPol(zs::range(etemp.size()),
            [etemp = proxy<space>({},etemp),act_buffer = proxy<space>({},act_buffer),muscle_id_tag = SmallString(muscle_id_tag),nm_acts] ZS_LAMBDA(int ei) mutable {
                // auto act = eles.template pack<3>("act",ei);
                auto fiber = etemp.template pack<3>("fiber",ei);
                
                vec3 act{0};

                auto nfiber = fiber.norm();
                auto ID = etemp("muscle_ID",ei);
                if(nfiber < 0.5 || ID < -1e-6 || nm_acts == 0){ // if there is no local fiber orientaion, use the default act and fiber
                    fiber = vec3{1.0,0.0,0.0};
                    act = vec3{1.0,1.0,1.0};
                }else{
                    // a test
                    int id = (int)ID;
                    float a = 1. - act_buffer("act",id);
                    act = vec3{1,zs::sqrt(1./a),zs::sqrt(1./a)};
                    fiber /= nfiber;// in case there is some floating-point error

                    // printf("use act[%d] : %f\n",id,(float)a);
                }

                vec3 dir[3];
                dir[0] = fiber;
                auto tmp = vec3{1.0,0.0,0.0};
                dir[1] = dir[0].cross(tmp);
                if(dir[1].length() < 1e-3) {
                    tmp = vec3{0.0,1.0,0.0};
                    dir[1] = dir[0].cross(tmp);
                }

                dir[1] = dir[1] / dir[1].length();
                dir[2] = dir[0].cross(dir[1]);

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
        });
        // std::cout << "set initial guess" << std::endl;
        // setup initial guess
        TILEVEC_OPS::copy<3>(cudaPol,verts,"x",vtemp,"xp");
        TILEVEC_OPS::copy<3>(cudaPol,verts,"v",vtemp,"vp");
        if(verts.hasProperty("init_x"))
            TILEVEC_OPS::copy<3>(cudaPol,verts,"init_x",vtemp,"xn");   
        else
            TILEVEC_OPS::add<3>(cudaPol,vtemp,"xp",1.0,"vp",dt,"xn");  
        TILEVEC_OPS::fill<1>(cudaPol,vtemp,"bou_tag",zs::vec<T,1>::zeros());


        match([&](auto &elasticModel) {
            A.computeGradientAndHessian(cudaPol, elasticModel,"xn","xp","vp",vtemp,"H",etemp);
        })(models.getElasticModel());

        PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"H",etemp,"P",vtemp);

        // if the grad is too small, return the result
        // Solve equation using PCG
        TILEVEC_OPS::fill<3>(cudaPol,vtemp,"dir",zs::vec<T,3>::zeros());
        // std::cout << "solve using pcg" << std::endl;
        PCG::pcg_with_fixed_sol_solve<3,4>(cudaPol,vtemp,etemp,"dir","bou_tag","grad","P","inds","H",cg_res,1000,50);
        // std::cout << "finish solve pcg" << std::endl;
        PCG::project<3>(cudaPol,vtemp,"dir","bou_tag");
        T alpha = 1.;
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                            alpha] __device__(int i) mutable {
            vtemp.tuple<3>("xn", i) =
                vtemp.pack<3>("xn", i) + alpha * vtemp.pack<3>("dir", i);
        });


        cudaPol(zs::range(verts.size()),
                [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),dt] __device__(int vi) mutable {
                    auto newX = vtemp.pack<3>("xn", vi);
                    verts.tuple<3>("x", vi) = newX;
                    verts.tuple<3>("v",vi) = (vtemp.pack<3>("xn",vi) - vtemp.pack<3>("xp",vi))/dt;
                });

        cudaPol.syncCtx();
        set_output("ZSParticles", zstets);
    }


};

ZENDEFNODE(FleshDynamicStepping, {{"ZSParticles","driven_boudary","gravity","Acts"},
                                  {"ZSParticles"},
                                  {
                                    {"string","driven_tag","bone_bw"},
                                    {"string","muscle_id_tag","ms_id_tag"},
                                    {"float","dt","0.03"}
                                  },
                                  {"FEM"}});



};