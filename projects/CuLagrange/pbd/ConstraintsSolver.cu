#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "constraint_function_kernel/constraint.cuh"
#include "../geometry/kernel/tiled_vector_ops.hpp"
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/geo_math.hpp"
#include "../geometry/kernel/bary_centric_weights.hpp"
// #include "../fem/collision_energy/evaluate_collision.hpp"
#include "constraint_function_kernel/constraint_types.hpp"
#include "../fem/collision_energy/evaluate_collision.hpp"

namespace zeno {


// solve a specific type of constraint for one iterations
struct XPBDSolve : INode {

    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;

        using vec3 = zs::vec<float,3>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto constraints = get_input<ZenoParticles>("constraints");

        // auto target = get_input<ZenoParticles>("kbounadry");


        auto dt = get_input2<float>("dt");   
        auto ptag = get_input2<std::string>("ptag");
        auto pptag = get_input2<std::string>("pptag");

        auto substeps_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substeps_id + 1) / (float)nm_substeps;

        // auto current_substep_id = get_input2<int>("substep_id");
        // auto total_substeps = get_input2<int>("total_substeps");
        auto category = constraints->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});

        if(category != category_c::empty_constraint) {

        auto coffsets = constraints->readMeta(CONSTRAINT_COLOR_OFFSET,zs::wrapt<zs::Vector<int>>{});  
        int nm_group = coffsets.size();

        auto& verts = zsparticles->getParticles();
        auto& cquads = constraints->getQuadraturePoints();


        auto target = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
        const auto& kverts = target->getParticles();
        const auto& kcells = target->getQuadraturePoints();

        // std::cout << "IN XPBDSolve " << std::endl;

        for(int g = 0;g != nm_group;++g) {
            auto coffset = coffsets.getVal(g);
            int group_size = 0;
            if(g == nm_group - 1)
                group_size = cquads.size() - coffsets.getVal(g);
            else
                group_size = coffsets.getVal(g + 1) - coffsets.getVal(g);

            cudaPol(zs::range(group_size),[
                coffset = coffset,
                verts = proxy<space>({},verts),
                category = category,
                dt = dt,
                w = w,
                substeps_id = substeps_id,
                nm_substeps = nm_substeps,
                ptag = zs::SmallString(ptag),
                pptag = zs::SmallString(pptag),
                kverts = proxy<space>({},kverts),
                kcells = proxy<space>({},kcells),
                cquads = proxy<space>({},cquads)] ZS_LAMBDA(int gi) mutable {
                    float alpha = cquads("xpbd_affiliation",coffset + gi);
                    float lambda = cquads("lambda",coffset + gi);
                    float kd = cquads("damping_coeff",coffset + gi);

                    if(category == category_c::volume_pin_constraint) {
                        auto pair = cquads.pack(dim_c<2>,"inds",coffset + gi,int_c);
                        auto bary = cquads.pack(dim_c<4>,"bary",coffset + gi);
                        auto pi = pair[0];
                        auto kti = pair[1];
                        if(kti < 0)
                            return;
                        auto ktet = kcells.pack(dim_c<4>,"inds",kti,int_c);

                        // printf("volume_pin[%d %d] : bary[%f %f %f %f]\n",pi,kti,
                        //     (float)bary[0],(float)bary[1],(float)bary[2],(float)bary[3]);

                        auto ktp = vec3::zeros();
                        for(int i = 0;i != 4;++i) 
                            ktp += kverts.pack(dim_c<3>,"x",ktet[i]) * bary[i];
                        auto pktp = vec3::zeros();
                        for(int i = 0;i != 4;++i) 
                            pktp += kverts.pack(dim_c<3>,"px",ktet[i]) * bary[i];
                        verts.tuple(dim_c<3>,ptag,pi) = (1 - w) * pktp + w * ktp;
                    }

                    if(category == category_c::follow_animation_constraint) {
                        // auto vi = coffset + gi;
                        auto pi = zs::reinterpret_bits<int>(cquads("inds",coffset + gi));
                        auto kminv = cquads("follow_weight",pi);
                        auto p = verts.pack(dim_c<3>,ptag,pi);

                        auto kp = kverts.pack(dim_c<3>,"x",pi);
                        auto pkp = kverts.pack(dim_c<3>,"px",pi);
                        
                        auto tp = (1 - w) * pkp + w * kp;

                        // float pminv = 1;
                        // float kpminv = pminv * fw;
                        vec3 dp{},dkp{};
                        // auto ori_lambda = lambda;
                        if(!CONSTRAINT::solve_DistanceConstraint(
                                p,1.f,
                                tp,(float)kminv * 10.f,
                                0.f,
                                alpha,
                                dp,dkp))
                            return;
                        verts.tuple(dim_c<3>,ptag,pi) = p + dp;                      
                    }

                    if(category == category_c::pt_pin_constraint) {
                        auto pair = cquads.pack(dim_c<2>,"inds",coffset + gi,int_c);
                        if(pair[0] <= 0 || pair[1] <= 0) {
                            printf("invalid pair[%d %d] detected %d %d\n",pair[0],pair[1],coffset,gi);
                            return;
                        }
                        auto pi = pair[0];
                        auto kti = pair[1];
                        if(kti < 0)
                            return;
                        auto ktri = kcells.pack(dim_c<3>,"inds",kti,int_c);
                        auto rd = cquads("rd",coffset + gi);
                        auto bary = cquads.pack(dim_c<3>,"bary",coffset + gi);

                        vec3 kps[3] = {};
                        auto kc = vec3::zeros();
                        for(int i = 0;i != 3;++i){
                            kps[i] = kverts.pack(dim_c<3>,"x",ktri[i]) * w + kverts.pack(dim_c<3>,"px",ktri[i]) * (1 - w);
                            kc += kps[i] * bary[i];
                        }
                            
                        auto knrm = LSL_GEO::facet_normal(kps[0],kps[1],kps[2]);
                        verts.tuple(dim_c<3>,ptag,pi) = kc + knrm * rd;
                    }

                    if(category == category_c::edge_length_constraint || category == category_c::dihedral_spring_constraint) {
                        // printf("do xpbd solve\n");

                        auto edge = cquads.pack(dim_c<2>,"inds",coffset + gi,int_c);
                        // vec3 p0{},p1{};
                        auto p0 = verts.pack(dim_c<3>,ptag,edge[0]);
                        auto p1 = verts.pack(dim_c<3>,ptag,edge[1]);
                        auto pp0 = verts.pack(dim_c<3>,pptag,edge[0]);
                        auto pp1 = verts.pack(dim_c<3>,pptag,edge[1]);

                        float minv0 = verts("minv",edge[0]);
                        float minv1 = verts("minv",edge[1]);
                        float r = cquads("r",coffset + gi);

                        vec3 dp0{},dp1{};
                        // if(!CONSTRAINT::solve_DistanceConstraint(
                        //     p0,minv0,
                        //     p1,minv1,
                        //     r,
                        //     alpha,
                        //     dt,
                        //     lambda,
                        //     dp0,dp1))
                        //         return;
                        if(!CONSTRAINT::solve_DistanceConstraint(
                            p0,minv0,
                            p1,minv1,
                            pp0,pp1,
                            r,
                            alpha,
                            kd,
                            dt,
                            lambda,
                            dp0,dp1))
                                return;

                        verts.tuple(dim_c<3>,ptag,edge[0]) = p0 + dp0;
                        verts.tuple(dim_c<3>,ptag,edge[1]) = p1 + dp1;
                    }
                    if(category == category_c::isometric_bending_constraint) {
                        auto quad = cquads.pack(dim_c<4>,"inds",coffset + gi,int_c);
                        vec3 p[4] = {};
                        float minv[4] = {};
                        for(int i = 0;i != 4;++i) {
                            p[i] = verts.pack(dim_c<3>,ptag,quad[i]);
                            minv[i] = verts("minv",quad[i]);
                        }

                        auto Q = cquads.pack(dim_c<4,4>,"Q",coffset + gi);
                        auto C0 = cquads("C0",coffset + gi);

                        vec3 dp[4] = {};
                        if(!CONSTRAINT::solve_IsometricBendingConstraint(
                            p[0],minv[0],
                            p[1],minv[1],
                            p[2],minv[2],
                            p[3],minv[3],
                            Q,
                            alpha,
                            dt,
                            C0,
                            lambda,
                            dp[0],dp[1],dp[2],dp[3]))
                                return;

                        for(int i = 0;i != 4;++i) {
                            // printf("dp[%d][%d] : %f %f %f %f\n",gi,i,s,(float)dp[i][0],(float)dp[i][1],(float)dp[i][2]);
                            verts.tuple(dim_c<3>,ptag,quad[i]) = p[i] + dp[i];
                        }
                    }

                    if(category == category_c::dihedral_bending_constraint) {
                        auto quad = cquads.pack(dim_c<4>,"inds",coffset + gi,int_c);
                        vec3 p[4] = {};
                        vec3 pp[4] = {};
                        float minv[4] = {};
                        for(int i = 0;i != 4;++i) {
                            p[i] = verts.pack(dim_c<3>,ptag,quad[i]);
                            pp[i] = verts.pack(dim_c<3>,pptag,quad[i]);
                            minv[i] = verts("minv",quad[i]);
                        }

                        auto ra = cquads("ra",coffset + gi);
                        auto ras = cquads("sign",coffset + gi);
                        vec3 dp[4] = {};
                        // if(!CONSTRAINT::solve_DihedralConstraint(
                        //     p[0],minv[0],
                        //     p[1],minv[1],
                        //     p[2],minv[2],
                        //     p[3],minv[3],
                        //     ra,
                        //     ras,
                        //     alpha,
                        //     dt,
                        //     lambda,
                        //     dp[0],dp[1],dp[2],dp[3]))
                        //         return;
                        if(!CONSTRAINT::solve_DihedralConstraint(
                            p[0],minv[0],
                            p[1],minv[1],
                            p[2],minv[2],
                            p[3],minv[3],
                            pp[0],pp[1],pp[2],pp[3],
                            ra,
                            ras,
                            alpha,
                            dt,
                            kd,
                            lambda,
                            dp[0],dp[1],dp[2],dp[3]))
                                return;                        
                        for(int i = 0;i != 4;++i) {
                            // printf("dp[%d][%d] : %f %f %f %f\n",gi,i,s,(float)dp[i][0],(float)dp[i][1],(float)dp[i][2]);
                            verts.tuple(dim_c<3>,ptag,quad[i]) = p[i] + dp[i];
                        }                        
                    }
                    cquads("lambda",coffset + gi) = lambda;
            });
        }      
        }

        set_output("constraints",constraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolve, {{{"zsparticles"},
                            {"constraints"},
                            {"int","substep_id","0"},
                            {"int","nm_substeps","1"},
                            {"string","ptag","x"},
                            {"string","pptag","px"},
                            {"float","dt","0.5"}},
							{{"zsparticles"},{"constraints"}},
							{},
							{"PBD"}});

struct XPBDSolveSmooth : INode {

    using bvh_t = ZenoLinearBvh::lbvh_t;
    using bv_t = bvh_t::Box;
    using dtiles_t = zs::TileVector<T,32>;

    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;

        using vec3 = zs::vec<float,3>;
        using vec4 = zs::vec<float,4>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};

        auto zsparticles = get_input<ZenoParticles>("zsparticles");

        // auto all_constraints = RETRIEVE_OBJECT_PTRS(ZenoParticles, "all_constraints");
        auto constraints = get_input<ZenoParticles>("constraints");
        // auto ptag = get_param<std::string>("ptag");
        auto relaxs = get_input2<float>("relaxation_strength");

        auto& verts = zsparticles->getParticles();
        auto nm_smooth_iters = get_input2<int>("nm_smooth_iters");

        zs::Vector<float> dp_buffer{verts.get_allocator(),verts.size() * 3};
        cudaPol(zs::range(dp_buffer),[]ZS_LAMBDA(auto& v) {v = 0;});
        zs::Vector<int> dp_count{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(dp_count),[]ZS_LAMBDA(auto& c) {c = 0;});

        auto category = constraints->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});

        // if(category == category_c::follow_animation_constraint) {
        //     auto substep_id = get_input2<int>("substep_id");
        //     auto nm_substeps = get_input2<int>("nm_substeps");
        //     auto w = (float)(substep_id + 1) / (float)nm_substeps;
        //     auto pw = (float)(substep_id) / (float)nm_substeps;
        // }

        if(category == category_c::dcd_collision_constraint) {
            constexpr auto eps = 1e-6;

            const auto& cquads = constraints->getQuadraturePoints();
            const auto& tris = zsparticles->getQuadraturePoints();
            const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 

            if(!constraints->hasMeta(NM_DCD_COLLISIONS))
                return;
            auto nm_dcd_collisions = constraints->readMeta<size_t>(NM_DCD_COLLISIONS);
            auto imminent_thickness = constraints->readMeta<float>(GLOBAL_DCD_THICKNESS);

            auto has_input_collider = constraints->hasMeta(CONSTRAINT_TARGET);

            auto substep_id = get_input2<int>("substep_id");
            auto nm_substeps = get_input2<int>("nm_substeps");
            auto w = (float)(substep_id + 1) / (float)nm_substeps;
            auto pw = (float)(substep_id) / (float)nm_substeps;

            auto nm_verts = verts.size();
            auto nm_tris = tris.size();
            auto nm_edges = edges.size();       

            if(has_input_collider) {
                auto collider = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
                nm_verts += collider->getParticles().size();
                nm_edges += (*collider)[ZenoParticles::s_surfEdgeTag].size();
                nm_tris += collider->getQuadraturePoints().size();
            }    

            dtiles_t vtemp{verts.get_allocator(),{
                {"x",3},
                {"v",3},
                {"minv",1},
                {"m",1}
            },nm_verts};

            auto pptag = get_input2<std::string>("pptag");


            TILEVEC_OPS::copy<3>(cudaPol,verts,pptag,vtemp,"x");
            TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv");
            TILEVEC_OPS::copy(cudaPol,verts,"m",vtemp,"m");
            cudaPol(zs::range(verts.size()),[
                vtemp = proxy<space>({},vtemp),
                pptag = zs::SmallString(pptag),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                    vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,"x",vi) - verts.pack(dim_c<3>,pptag,vi);
            });  

            if(has_input_collider) {
                auto boundary_velocity_scale = get_input2<float>("boundary_velocity_scale");

                auto collider = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
                const auto& kverts = collider->getParticles();
                const auto& kedges = (*collider)[ZenoParticles::s_surfEdgeTag];
                const auto& ktris = collider->getQuadraturePoints();  

                auto voffset = verts.size();
                cudaPol(zs::range(kverts.size()),[
                    kverts = proxy<space>({},kverts),
                    voffset = voffset,
                    pw = pw,
                    boundary_velocity_scale = boundary_velocity_scale,
                    w = w,
                    nm_substeps = nm_substeps,
                    vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int kvi) mutable {
                        auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                        auto cur_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - w) + kverts.pack(dim_c<3>,"x",kvi) * w;
                        vtemp.tuple(dim_c<3>,"x",voffset + kvi) = pre_kvert;
                        vtemp("minv",voffset + kvi) = 0;  
                        vtemp("m",voffset + kvi) = (T)1000;
                        vtemp.tuple(dim_c<3>,"v",voffset + kvi) = (cur_kvert - pre_kvert) * boundary_velocity_scale;
                });            
            }

            auto add_repulsion_force = get_input2<bool>("add_repulsion_force");

            for(auto iter = 0;iter != nm_smooth_iters;++iter) {
                cudaPol(zs::range(verts.size()),[
                    dp_buffer = proxy<space>(dp_buffer),
                    dp_count = proxy<space>(dp_count)] ZS_LAMBDA(int vi) mutable {
                        for(int d = 0;d != 3;++d)
                            dp_buffer[vi * 3 + d] = 0;
                });

                cudaPol(zs::range(nm_dcd_collisions),[
                    cquads = proxy<space>({},cquads),
                    vtemp = proxy<space>({},vtemp),
                    exec_tag = exec_tag,
                    eps = eps,
                    add_repulsion_force = add_repulsion_force,
                    imminent_thickness = imminent_thickness,
                    dp_buffer = proxy<space>(dp_buffer),
                    dp_count = proxy<space>(dp_count)] ZS_LAMBDA(int ci) mutable {
                        auto inds = cquads.pack(dim_c<4>,"inds",ci,int_c);
                        auto bary = cquads.pack(dim_c<4>,"bary",ci);
                        auto type = zs::reinterpret_bits<int>(cquads("type",ci));
                        
                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        vec4 minvs{};
                        vec4 ms{};

                        for(int i = 0;i != 4;++i) {
                            ps[i] = vtemp.pack(dim_c<3>,"x",inds[i]);
                            vs[i] = vtemp.pack(dim_c<3>,"v",inds[i]);
                            minvs[i] = vtemp("minv",inds[i]);
                            ms[i] = vtemp("m",inds[i]);
                        }

                        vec3 imps[4] = {};
                        if(!COLLISION_UTILS::compute_imminent_collision_impulse(ps,vs,bary,ms,minvs,imps,imminent_thickness,type,add_repulsion_force))
                            return;
                        for(int i = 0;i != 4;++i) {
                            if(minvs[i] < eps)
                                continue;

                            if(isnan(imps[i].norm())) {
                                printf("nan imps detected : %f %f %f %f %f %f %f\n",
                                    (float)imps[i][0],(float)imps[i][1],(float)imps[i][2],
                                    (float)bary[0],(float)bary[1],(float)bary[2],(float)bary[3]);
                                return;
                            }
                            atomic_add(exec_tag,&dp_count[inds[i]],(int)1);
                            for(int d = 0;d != 3;++d)
                                atomic_add(exec_tag,&dp_buffer[inds[i] * 3 + d],imps[i][d]);
                        }
                });

                cudaPol(zs::range(verts.size()),[
                    vtemp = proxy<space>({},vtemp),relaxs = relaxs,
                    dp_count = proxy<space>(dp_count),
                    dp_buffer = proxy<space>(dp_buffer)] ZS_LAMBDA(int vi) mutable {
                        if(dp_count[vi] > 0) {
                            auto dp = relaxs * vec3{dp_buffer[vi * 3 + 0],dp_buffer[vi * 3 + 1],dp_buffer[vi * 3 + 2]};
                            vtemp.tuple(dim_c<3>,"v",vi) = vtemp.pack(dim_c<3>,"v",vi) + dp / (T)dp_count[vi];
                        }
                });

            }

            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                    verts.tuple(dim_c<3>,"x",vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
            });
        
        }

        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolveSmooth, {{{"zsparticles"},
                                {"constraints"},
                                {"float","relaxation_strength","1"},
                                {"int","nm_smooth_iters","1"},
                                {"int","nm_substeps","1"},
                                {"int","substep_id","0"},
                                {"bool","add_repulsion_force","0"},
                                {"float","boundary_velocity_scale","1"},
                                {"string","pptag","px"}
                            }, 
							{{"zsparticles"}},
							{},
							{"PBD"}});



struct XPBDSolveSmoothAll : INode {

    using bvh_t = ZenoLinearBvh::lbvh_t;
    using bv_t = bvh_t::Box;
    using dtiles_t = zs::TileVector<T,32>;

    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;

        using vec3 = zs::vec<float,3>;
        using vec4 = zs::vec<float,4>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = 1e-6;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");

        // auto all_constraints = RETRIEVE_OBJECT_PTRS(ZenoParticles, "all_constraints");
        auto constraint_ptr_list = RETRIEVE_OBJECT_PTRS(ZenoParticles, "constraints");
        // auto ptag = get_param<std::string>("ptag");
        // auto relaxs = get_input2<float>("relaxation_strength");

        auto& verts = zsparticles->getParticles();
        auto ptag = get_input2<std::string>("ptag");
        auto pptag = get_input2<std::string>("pptag");
        auto dptag = get_input2<std::string>("dptag");
        auto dt = get_input2<float>("dt");

        // auto dt = get_input2<bool>("dt");

        zs::Vector<float> weight_sum{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(weight_sum),[]ZS_LAMBDA(auto& w) {w = 0;});

        for(auto& constraint_ptr : constraint_ptr_list) {
            auto category = constraint_ptr->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});
            const auto& cquads = constraint_ptr->getQuadraturePoints();

            if(category == category_c::edge_length_constraint || category == category_c::dihedral_bending_constraint) {
                cudaPol(zs::range(cquads.size()),[
                    cquads = proxy<space>({},cquads),
                    dt = dt,
                    stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                    affiliationOffset = cquads.getPropertyOffset("xpbd_affiliation"),
                    dampingOffset = cquads.getPropertyOffset("damping_coeff"),
                    indsOffset = cquads.getPropertyOffset("inds"),
                    weight_sum = proxy<space>(weight_sum),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    pptagOffset = verts.getPropertyOffset(pptag),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    minvOffset = verts.getPropertyOffset("minv"),
                    verts = view<space>(verts),
                    category = category,
                    exec_tag = exec_tag] ZS_LAMBDA(int ci) mutable {
                        auto w = cquads(stiffnessOffset,ci);
                        auto aff = cquads(affiliationOffset,ci);
                        auto kd = cquads(dampingOffset,ci);
                        if(category == category_c::edge_length_constraint) {
                            auto edge = cquads.pack(dim_c<2>,indsOffset,ci,int_c);
                            auto p0 = verts.pack(dim_c<3>,ptagOffset,edge[0]);
                            auto p1 = verts.pack(dim_c<3>,ptagOffset,edge[1]);
                            auto pp0 = verts.pack(dim_c<3>,pptagOffset,edge[0]);
                            auto pp1 = verts.pack(dim_c<3>,pptagOffset,edge[1]);
                            auto  minv0 = verts(minvOffset,edge[0]);
                            auto minv1 = verts(minvOffset,edge[1]);
                            auto r = cquads("r",ci);
                            vec3 dp[2] = {};

                            auto lambda = (T)0;
                            if(!CONSTRAINT::solve_DistanceConstraint(
                                p0,minv0,
                                p1,minv1,
                                pp0,pp1,
                                r,
                                aff,
                                kd,
                                dt,
                                lambda,
                                dp[0],dp[1]))
                                    return;
                            // printf("smooth stretch update : %f %f\n",(float)dp[0].norm(),(float)dp[1].norm());
                            for(int i = 0;i != 2;++i) {
                                if(isnan(dp[i].norm()))
                                    printf("nan dp[%d] detected at stretch\n",i);
                                atomic_add(exec_tag,&weight_sum[edge[i]],w);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(dptagOffset + d,edge[i]),dp[i][d]);
                            }
                        }

                        if(category == category_c::dihedral_bending_constraint) {
                            auto quad = cquads.pack(dim_c<4>,indsOffset,ci,int_c);
                            vec3 p[4] = {};
                            vec3 pp[4] = {};
                            float minv[4] = {};
                            for(int i = 0;i != 4;++i) {
                                p[i] = verts.pack(dim_c<3>,ptagOffset,quad[i]);
                                pp[i] = verts.pack(dim_c<3>,pptagOffset,quad[i]);
                                minv[i] = verts(minvOffset,quad[i]);
                            }
    
                            auto ra = cquads("ra",ci);
                            auto ras = cquads("sign",ci);
                            vec3 dp[4] = {};
                            auto lambda = (T)0;
                            if(!CONSTRAINT::solve_DihedralConstraint(
                                p[0],minv[0],
                                p[1],minv[1],
                                p[2],minv[2],
                                p[3],minv[3],
                                pp[0],pp[1],pp[2],pp[3],
                                ra,
                                ras,
                                aff,
                                dt,
                                kd,
                                lambda,
                                dp[0],dp[1],dp[2],dp[3]))
                                    return;    
                            for(int i = 0;i != 4;++i) {
                                if(isnan(dp[i].norm()))
                                    printf("nan dp[%d] detected at stretch\n",i);
                                atomic_add(exec_tag,&weight_sum[quad[i]],w);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(dptagOffset + d,quad[i]),dp[i][d]);
                            }                        
                        }
                });
            }

            // if(category == category_c::dcd_collision_constraint) {
            //     const auto& tris = zsparticles->getQuadraturePoints();
            //     const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
                     
            //     if(!constraint_ptr->hasMeta(NM_DCD_COLLISIONS))
            //         continue;
                
            //     auto nm_dcd_collisions = constraints->readMeta<size_t>(NM_DCD_COLLISIONS);
            //     auto imminent_thickness = constraints->readMeta<float>(GLOBAL_DCD_THICKNESS);  
                 
            //     auto has_input_collider = constraints->hasMeta(CONSTRAINT_TARGET);

            //     auto substep_id = get_input2<int>("substep_id");
            //     auto nm_substeps = get_input2<int>("nm_substeps"); 
            //     auto w = (float)(substep_id + 1) / (float)nm_substeps;
            //     auto pw = (float)(substep_id) / (float)nm_substeps;

            //     auto nm_verts = verts.size();
            //     auto nm_tris = tris.size();
            //     auto nm_edges = edges.size();    

            //     if(has_input_collider) {
            //         auto collider = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
            //         nm_verts += collider->getParticles().size();
            //         nm_edges += (*collider)[ZenoParticles::s_surfEdgeTag].size();
            //         nm_tris += collider->getQuadraturePoints().size();
            //     }   
            // }
        }

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            eps = eps,
            dptagOffset = verts.getPropertyOffset(dptag),
            weight_sum = proxy<space>(weight_sum)] ZS_LAMBDA(int vi) mutable {
                if(weight_sum[vi] > eps)
                    verts.tuple(dim_c<3>,dptagOffset,vi) = verts.pack(dim_c<3>,dptagOffset,vi) / (T)weight_sum[vi];
                else
                    verts.tuple(dim_c<3>,dptagOffset,vi) = vec3::zeros();
        });

        set_output("zsparticles",get_input("zsparticles"));
        set_output("constraints",get_input("constraints"));
    };
};


ZENDEFNODE(XPBDSolveSmoothAll, {{{"zsparticles"},
                                {"constraints"},
                                {"string","ptag","x"},
                                {"string","pptag","px"},
                                {"string","dptag","dx"},
                                {"float","dt","1.0"},
                            },
							{{"zsparticles"},{"constraints"}},
							{},
							{"PBD"}});

struct VisualizeDCDProximity : zeno::INode {

    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;
        using dtiles_t = zs::TileVector<T,32>;

        using vec3 = zs::vec<float,3>;
        using vec4 = zs::vec<float,4>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto constraints = get_input<ZenoParticles>("constraints");
        auto& verts = zsparticles->getParticles();

        const auto& cquads = constraints->getQuadraturePoints().clone({zs::memsrc_e::host});
        const auto& tris = zsparticles->getQuadraturePoints();
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 

        auto nm_dcd_collisions = constraints->readMeta<size_t>(NM_DCD_COLLISIONS);
        auto imminent_thickness = constraints->readMeta<float>(GLOBAL_DCD_THICKNESS);
    
        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;

        auto nm_verts = verts.size();
        auto nm_tris = tris.size();
        auto nm_edges = edges.size();     

        auto collider = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
        nm_verts += collider->getParticles().size();
        nm_edges += (*collider)[ZenoParticles::s_surfEdgeTag].size();
        nm_tris += collider->getQuadraturePoints().size();  
        
        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3}
        },nm_verts};

        TILEVEC_OPS::copy<3>(cudaPol,verts,"px",vtemp,"x");
        cudaPol(zs::range(verts.size()),[
            vtemp = proxy<space>({},vtemp),
            verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,"x",vi) - verts.pack(dim_c<3>,"px",vi);
        });  

        const auto& kverts = collider->getParticles();
        const auto& kedges = (*collider)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = collider->getQuadraturePoints();  

        auto voffset = verts.size();
        cudaPol(zs::range(kverts.size()),[
            kverts = proxy<space>({},kverts),
            voffset = voffset,
            pw = pw,
            nm_substeps = nm_substeps,
            vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int kvi) mutable {
                vtemp.tuple(dim_c<3>,"x",voffset + kvi) = kverts.pack(dim_c<3>,"x",kvi) * pw + kverts.pack(dim_c<3>,"px",kvi) * (1 - pw);
                // vtemp("minv",voffset + kvi) = 0;  
                vtemp.tuple(dim_c<3>,"v",voffset + kvi) = (kverts.pack(dim_c<3>,"x",kvi) - kverts.pack(dim_c<3>,"px",kvi)) / (float)nm_substeps;
        });   

        std::cout << "nm_dcd_collisions : " << nm_dcd_collisions << std::endl;

        auto dcd_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& dcd_vis_verts = dcd_vis->verts;
        auto& dcd_vis_lines = dcd_vis->lines;
        dcd_vis_verts.resize(nm_dcd_collisions * 2);
        dcd_vis_lines.resize(nm_dcd_collisions);

        vtemp = vtemp.clone({zs::memsrc_e::host});
        auto ompPol = omp_exec();
        constexpr auto omp_space = execspace_e::openmp;    

        ompPol(zs::range(nm_dcd_collisions),[
            &dcd_vis_verts,&dcd_vis_lines,
            vtemp = proxy<omp_space>({},vtemp),
            cquads = proxy<omp_space>({},cquads)] (int ci) mutable {
                auto inds = cquads.pack(dim_c<4>,"inds",ci,int_c);
                auto bary = cquads.pack(dim_c<4>,"bary",ci);

                auto type = zs::reinterpret_bits<int>(cquads("type",ci));
                vec3 ps[4] = {};
                for(int i = 0;i != 4;++i)
                    ps[i] = vtemp.pack(dim_c<3>,"x",inds[i]);
                
                vec3 v0{},v1{};
                if(type == 0) {
                    v0 = -(ps[0] * bary[0] + ps[1] * bary[1] + ps[2] * bary[2]);
                    v1 = ps[3] * bary[3]; 
                }
                else if(type == 1) {
                    v0 = -(ps[0] * bary[0] + ps[1] * bary[1]);
                    v1 = (ps[2] * bary[2] + ps[3] * bary[3]);
                }else {
                    printf("invalid type detected\n");
                    return;
                }

                dcd_vis_verts[ci * 2 + 0] = v0.to_array();
                dcd_vis_verts[ci * 2 + 1] = v1.to_array();
                dcd_vis_lines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });

        set_output("dcd_vis",std::move(dcd_vis));
    }
};

ZENDEFNODE(VisualizeDCDProximity, {{{"zsparticles"},
                                {"constraints"},
                                {"int","nm_substeps","1"},
                                {"int","substep_id","0"}                         
                            },
                            {{"dcd_vis"}},
                            {},
                            {"ZSGeometry"}});



};
