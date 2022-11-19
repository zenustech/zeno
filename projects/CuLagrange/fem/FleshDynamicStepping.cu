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
#include "collision_energy/edge_edge_sqrt_collision.hpp"
#include "collision_energy/edge_edge_collision.hpp"

#include "collision_energy/evaluate_collision.hpp"

namespace zeno {

#define MAX_FP_COLLISION_PAIRS 4

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

        template <typename Model>
        void computeCollisionGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,const Model& model,
                            dtiles_t& vtemp,
                            dtiles_t& etemp,
                            dtiles_t& sttemp,
                            dtiles_t& setemp,
                            dtiles_t& ee_buffer,
                            dtiles_t& fp_buffer,
                            dtiles_t& gh_buffer,
                            bool explicit_collision = false,
                            bool neglect_inverted = true) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            int offset = eles.size() + b_verts.size();

            T lambda = model.lam;
            T mu = model.mu; 

            if(neglect_inverted) {
            // // figure out all the vertices which is incident to an inverted tet
                TILEVEC_OPS::fill(cudaPol,vtemp,"is_inverted",reinterpret_bits<T>((int)0));  
                cudaPol(zs::range(eles.size()),
                    [vtemp = proxy<space>({},vtemp),quads = proxy<space>({},eles)] ZS_LAMBDA(int ei) mutable {
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
                        if(zs::determinant(F) < 0.0)
                            for(int i = 0;i < 4;++i)
                                vtemp("is_inverted",inds[i]) = reinterpret_bits<T>((int)1);                  
                });

            }

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
                vtemp,"xn",
                fp_buffer,
                gh_buffer,offset,
                in_collisionEps,out_collisionEps,
                (T)collisionStiffness,
                (T)mu,(T)lambda);


            COLLISION_UTILS::do_edge_edge_collision_detection(cudaPol,
                vtemp,"xn",
                points,
                lines,
                tris,
                sttemp,
                setemp,
                ee_buffer,
                in_collisionEps,out_collisionEps);

            // COLLISION_UTILS::evaluate_ee_collision_grad_and_hessian(cudaPol,
            //     vtemp,"xn",
            //     ee_buffer,
            //     gh_buffer,offset + fp_buffer.size(),
            //     in_collisionEps,out_collisionEps,
            //     (T)collisionStiffness,
            //     (T)mu,(T)lambda);

            // project out all the neglect verts
            if(neglect_inverted) {
                cudaPol(zs::range(fp_buffer.size() + ee_buffer.size()),
                    [gh_buffer = proxy<space>({},gh_buffer),vtemp = proxy<space>({},vtemp),offset] ZS_LAMBDA(int cpi) {
                        auto inds = gh_buffer.template pack<4>("inds",cpi + offset).reinterpret_bits(int_c);
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
                            gh_buffer.template tuple<12*12>("H",cpi + offset) = zs::vec<T,12,12>::zeros();
                            gh_buffer.template tuple<12>("grad",cpi + offset) = zs::vec<T,12>::zeros();
                        }
                });    
            }
        }


        template <typename Model>
        void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                            const Model& model,
                            const dtiles_t& vtemp,
                            const dtiles_t& etemp,
                            dtiles_t& gh_buffer) {        
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            int offset = 0;
            TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",gh_buffer,"inds",offset);   
            // eval the inertia term gradient
            cudaPol(zs::range(eles.size()),[dt2 = dt2,
                        eles = proxy<space>({},eles),
                        vtemp = proxy<space>({},vtemp),
                        gh_buffer = proxy<space>({},gh_buffer),
                        dt = dt,offset = offset] ZS_LAMBDA(int ei) mutable {
                auto m = eles("m",ei)/(T)4.0;
                auto inds = eles.pack(dim_c<4>,"inds",ei).reinterpret_bits(int_c);
                auto pgrad = zs::vec<T,12>::zeros();
                for(int i = 0;i != 4;++i){
                    auto x1 = vtemp.pack(dim_c<3>,"xn",inds[i]);
                    auto x0 = vtemp.pack(dim_c<3>,"xp",inds[i]);
                    auto v0 = vtemp.pack(dim_c<3>,"vp",inds[i]);
                    auto nodal_pgrad = -m * (x1 - x0 - v0 * dt) / dt2;
                    for(int d = 0;d != 3;++d)
                        pgrad[i * 3 + d] = nodal_pgrad[d];
                }
                gh_buffer.tuple(dim_c<12>,"grad",ei + offset) = pgrad;
            });

            // if(!gh_buffer.hasProperty("H") || gh_buffer.getChannelSize("H") != 144)
            //     throw std::runtime_error("invalid gh_buffer's H channel");
            // if(!vtemp.hasProperty("xn"))
            //     throw std::runtime_error("invalid vtemp's xn channel");
            // if(!etemp.hasProperty("ActInv"))
            //     throw std::runtime_error("invalid etemp ActInv channel");

            // if(!eles.hasProperty("vol"))
            //     throw std::runtime_error("invalid eles vol channel");
            // if(!eles.hasProperty("m"))
            //     throw std::runtime_error("invalid eles m channel");

            // std::cout << "gh_buffer.size() = " << gh_buffer.size() << std::endl;
            // std::cout << "eles.size() = " << eles.size() << std::endl;
            // std::cout << "etemp.size() = " << etemp.size() << std::endl;

            cudaPol(zs::range(eles.size()), [dt2 = dt2,
                            vtemp = proxy<space>({}, vtemp),
                            etemp = proxy<space>({}, etemp),
                            gh_buffer = proxy<space>({},gh_buffer),
                            eles = proxy<space>({}, eles),
                            model, volf = volf,offset = offset] ZS_LAMBDA (int ei) mutable {
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
                auto P = model.first_piola(FAct);
                auto vole = eles("vol", ei);
                auto vecP = flatten(P);
                vecP = dFActdF.transpose() * vecP;
                auto dFdXT = dFdX.transpose();
                auto vf = -vole * (dFdXT * vecP);     

                auto mg = volf * vole / (T)4.0;
                for(int i = 0;i != 4;++i)
                    for(int d = 0;d !=3 ;++d)
                        vf[i*3 + d] += mg[d];
                gh_buffer.tuple(dim_c<12>,"grad",ei + offset) = gh_buffer.pack(dim_c<12>,"grad",ei + offset) + vf; 

                // assemble element-wise hessian matrix
                auto Hq = model.first_piola_derivative(FAct, true_c);
                auto dFdAct_dFdX = dFActdF * dFdX; 
                // add inertia hessian term
                auto H = dFdAct_dFdX.transpose() * Hq * dFdAct_dFdX * vole;
                auto m = eles("m",ei)/dt2/(T)4.0;
                for(int i = 0;i != 12;++i)
                    H(i,i) += m;
                gh_buffer.template tuple<12*12>("H",ei + offset) = H;
            });
        // Bone Driven Potential Energy
            T lambda = model.lam;
            T mu = model.mu;
            auto nmEmbedVerts = b_verts.size();

            // TILEVEC_OPS::fill_range<4>(cudaPol,gh_buffer,"inds",zs::vec<int,4>::uniform(-1).reinterpret_bits(float_c),eles.size() + offset,b_verts.size());
            // TILEVEC_OPS::fill_range<3>(cudaPol,gh_buffer,"grad",zs::vec<T,3>::zeros(),eles.size() + offset,b_verts.size());
            // TILEVEC_OPS::fill_range<144>(cudaPol,gh_buffer,"H",zs::vec<T,144>::zeros(),eles.size() + offset,b_verts.size());


            cudaPol(zs::range(nmEmbedVerts), [
                    gh_buffer = proxy<space>({},gh_buffer),
                    bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    eles = proxy<space>({},eles),lambda,mu,bone_driven_weight = bone_driven_weight,offset] ZS_LAMBDA(int vi) mutable {
                        auto ei = reinterpret_bits<int>(bcws("inds",vi));
                        if(ei < 0)
                            return;
                        auto inds = eles.pack<4>("inds",ei).reinterpret_bits(int_c);
                        gh_buffer.tuple(dim_c<4>,"inds",vi + offset + eles.size()) = eles.pack(dim_c<4>,"inds",ei);
                        auto w = bcws.pack<4>("w",vi);
                        auto tpos = vec3::zeros();
                        for(size_t i = 0;i != 4;++i)
                            tpos += w[i] * vtemp.pack<3>("xn",inds[i]);
                        auto pdiff = tpos - b_verts.pack<3>("x",vi);

                        T stiffness = 2.0066 * mu + 1.0122 * lambda;

                        zs::vec<T,12> elm_grad{};
                        auto elm_H = zs::vec<T,12,12>::zeros();

                        for(size_t i = 0;i != 4;++i){
                            auto tmp = pdiff * (-stiffness * bcws("cnorm",vi) * bone_driven_weight * w[i] * eles("vol",ei)); 
                            for(size_t d = 0;d != 3;++d)
                                elm_grad[i*3 + d] = tmp[d];
                        }
                        for(int i = 0;i != 4;++i)
                            for(int j = 0;j != 4;++j){
                                T alpha = stiffness * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi) * eles("vol",ei);
                                for(int d = 0;d != 3;++d){
                                    elm_H(i*3 + d,j*3 + d) = alpha;
                                }
                            }
                        
                        gh_buffer.tuple(dim_c<12>,"grad",vi + eles.size() + offset) = elm_grad;
                        gh_buffer.tuple(dim_c<12*12>,"H",vi + eles.size() + offset) = elm_H;
            });

        }

        FEMDynamicSteppingSystem(const tiles_t &verts, const tiles_t &eles,
                const tiles_t& points,const tiles_t& lines,const tiles_t& tris,
                T in_collisionEps,T out_collisionEps,
                const tiles_t &b_bcws, const tiles_t& b_verts,T bone_driven_weight,
                const vec3& volf,const T& _dt,const T& collisionStiffness)
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
            gravity = get_input2<zeno::vec<3,T>>("gravity");
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

        auto& tris  = (*zsparticles)[ZenoParticles::s_surfTriTag];
        auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];

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

        const auto& bverts = zsbones->getParticles();
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
                                {"vp",3},
                                {"is_inverted",1},
                                {"active",1}
                            },verts.size()};

        // auto max_collision_pairs = tris.size() / 10; 
        static dtiles_t etemp{eles.get_allocator(), {
                // {"H", 12 * 12},
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

        static dtiles_t fp_buffer(points.get_allocator(),{
            {"inds",4},
            {"area",1},
            {"inverted",1},
        },points.size() * MAX_FP_COLLISION_PAIRS);

        static dtiles_t ee_buffer(lines.get_allocator(),{
            {"inds",4},
            {"area",1},
            {"inverted",1},
            {"abary",2},
            {"bbary",2}
        },lines.size());

        static dtiles_t gh_buffer(eles.get_allocator(),{
            {"inds",4},
            {"H",12*12},
            {"grad",12}
        },eles.size() + bbw.size() + fp_buffer.size() + ee_buffer.size());


        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        // TILEVEC_OPS::fill<4>(cudaPol,etemp,"inds",zs::vec<int,4>::uniform(-1).template reinterpret_bits<T>())
        TILEVEC_OPS::copy<4>(cudaPol,eles,"inds",etemp,"inds");
        // TILEVEC_OPS::fill<9>(cudaPol,etemp,"ActInv",zs::vec<T,9>{1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0});
        // apply muscle activation
        cudaPol(zs::range(eles.size()),
            [etemp = proxy<space>({},etemp),eles = proxy<space>({},eles),
                act_buffer = proxy<space>({},act_buffer),muscle_id_tag = SmallString(muscle_id_tag),nm_acts] ZS_LAMBDA(int ei) mutable {
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
                etemp.template tuple<9>("ActInv",ei) = zs::inverse(Act);
        });
        auto collisionStiffness = get_input2<float>("cstiffness");


        // auto inset_ratio = get_input2<float>("collision_inset");
        // auto outset_ratio = get_input2<float>("collision_outset");    

        auto in_collisionEps = get_input2<float>("in_collisionEps");
        auto out_collisionEps = get_input2<float>("out_collisionEps");

        FEMDynamicSteppingSystem A{
            verts,eles,
            points,lines,tris,
            (T)in_collisionEps,(T)out_collisionEps,
            bbw,bverts,bone_driven_weight,
            volf,dt,collisionStiffness};

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
        TILEVEC_OPS::fill(cudaPol,vtemp,"bou_tag",(T)0.0);

        int max_newton_iterations = 5;
        int nm_iters = 0;
        // make sure, at least one baraf simi-implicit step will be taken
        auto res0 = 1e10;

        while(nm_iters < max_newton_iterations) {
            TILEVEC_OPS::fill(cudaPol,gh_buffer,"grad",(T)0.0);
            TILEVEC_OPS::fill(cudaPol,gh_buffer,"H",(T)0.0);  
            TILEVEC_OPS::fill<4>(cudaPol,gh_buffer,"inds",zs::vec<int,4>::uniform(-1).reinterpret_bits(float_c));    


            match([&](auto &elasticModel) {
                A.computeCollisionGradientAndHessian(cudaPol,elasticModel,
                    vtemp,
                    etemp,
                    sttemp,
                    setemp,
                    ee_buffer,
                    fp_buffer,
                    gh_buffer);
            })(models.getElasticModel());

            // auto collisionGradNorm = TILEVEC_OPS::inf_norm<12>(cudaPol,gh_buffer,"grad");
            // auto collisionHNorm = TILEVEC_OPS::inf_norm<12*12>(cudaPol,gh_buffer,"H");

            // std::cout << "collisionGradNorm : " << collisionGradNorm << std::endl;
            // std::cout << "collisionHNorm : " << collisionHNorm << std::endl;

            match([&](auto &elasticModel) {
                A.computeGradientAndHessian(cudaPol, elasticModel,vtemp,etemp,gh_buffer);
            })(models.getElasticModel());
            // break;


            TILEVEC_OPS::fill(cudaPol,vtemp,"grad",(T)0.0);
            
 
            TILEVEC_OPS::assemble(cudaPol,gh_buffer,"grad","inds",vtemp,"grad");
            // break;

            PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"H",gh_buffer,"P",vtemp);
            // PCG::prepare_block_diagonal_preconditioner<4,3>(cudaPol,"H",etemp,"P",vtemp);
            // if the grad is too small, return the result
            // Solve equation using PCG
            TILEVEC_OPS::fill(cudaPol,vtemp,"dir",(T)0.0);
            // std::cout << "solve using pcg" << std::endl;
            PCG::pcg_with_fixed_sol_solve<3,4>(cudaPol,vtemp,gh_buffer,"dir","bou_tag","grad","P","inds","H",cg_res,1000,50);
            T alpha = 1.;
            cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),alpha] __device__(int i) mutable {
                vtemp.tuple<3>("xn", i) =
                    vtemp.pack<3>("xn", i) + alpha * vtemp.pack<3>("dir", i);
            });

            T res = TILEVEC_OPS::inf_norm<3>(cudaPol, vtemp, "dir");// this norm is independent of descriterization
            std::cout << "res[" << nm_iters << "] : " << res << std::endl;
            if(res < 1e-3)
                break;
            nm_iters++;
        }

        cudaPol(zs::range(verts.size()),
                [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),dt] __device__(int vi) mutable {
                    auto newX = vtemp.pack(dim_c<3>,"xn", vi);
                    verts.tuple<3>("x", vi) = newX;
                    verts.tuple<3>("v",vi) = (vtemp.pack<3>("xn",vi) - vtemp.pack<3>("xp",vi))/dt;
                });

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