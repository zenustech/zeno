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

#include "zensim/math/matrix/QRSVD.hpp"

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

                    if(category == category_c::edge_length_constraint || category == category_c::dihedral_spring_constraint || category == category_c::long_range_attachment) {
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
                        bool stretch_resistence_only = category == category_c::long_range_attachment;
                        if(!CONSTRAINT::solve_DistanceConstraint(
                            p0,minv0,
                            p1,minv1,
                            pp0,pp1,
                            r,
                            alpha,
                            kd,
                            dt,
                            lambda,
                            dp0,dp1,
                            stretch_resistence_only))
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

                        auto ra = cquads("r",coffset + gi);
                        // auto ras = cquads("sign",coffset + gi);
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
                            // ras,
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
        auto dptag = get_input2<std::string>("dptag");
        auto ptag = get_input2<std::string>("ptag");
        auto pptag = get_input2<std::string>("pptag");
        auto relaxs = get_input2<float>("relaxation_strength");

        auto& verts = zsparticles->getParticles();
        if(!verts.hasProperty(dptag))
            verts.append_channels(cudaPol,{{dptag,3}});
        auto nm_smooth_iters = get_input2<int>("nm_smooth_iters");

        // zs::Vector<float> dp_buffer{verts.get_allocator(),verts.size() * 3};
        // cudaPol(zs::range(dp_buffer),[]ZS_LAMBDA(auto& v) {v = 0;});
        // zs::Vector<int> dp_count{verts.get_allocator(),verts.size()};
        // cudaPol(zs::range(dp_count),[]ZS_LAMBDA(auto& c) {c = 0;});



        auto category = constraints->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});

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
            // auto nm_tris = tris.size();
            // auto nm_edges = edges.size();       

            // if(has_input_collider) {
            //     auto collider = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
                // nm_verts += collider->getParticles().size();
                // nm_edges += (*collider)[ZenoParticles::s_surfEdgeTag].size();
                // nm_tris += collider->getQuadraturePoints().size();
            // }    

            // dtiles_t vtemp{verts.get_allocator(),{
            //     {"x",3},
            //     {"v",3},
            //     {"minv",1},
            //     {"m",1},
            //     // {"collision_cancel",1}
            // },nm_verts};



            // TILEVEC_OPS::copy<3>(cudaPol,verts,pptag,vtemp,"x");
            // TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv");
            // TILEVEC_OPS::copy(cudaPol,verts,"m",vtemp,"m");

            // cudaPol(zs::range(verts.size()),[
            //     vtemp = proxy<space>({},vtemp),
            //     pptag = zs::SmallString(pptag),
            //     verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
            //         vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,"x",vi) - verts.pack(dim_c<3>,pptag,vi);
            // });  

            // if(has_input_collider) {
            //     auto boundary_velocity_scale = get_input2<float>("boundary_velocity_scale");

            auto collider = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
            const auto& kverts = collider->getParticles();
            const auto& kedges = (*collider)[ZenoParticles::s_surfEdgeTag];
            const auto& ktris = collider->getQuadraturePoints();  

            //     auto voffset = verts.size();
            //     cudaPol(zs::range(kverts.size()),[
            //         kverts = proxy<space>({},kverts),
            //         voffset = voffset,
            //         pw = pw,
            //         boundary_velocity_scale = boundary_velocity_scale,
            //         w = w,
            //         nm_substeps = nm_substeps,
            //         // hasKCollisionCancel = kverts.hasProperty("collision_cancel"),
            //         // kCollisionCancelOffset = kverts.getPropertyOffset("collision_cancel"),
            //         vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int kvi) mutable {
            //             auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
            //             auto cur_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - w) + kverts.pack(dim_c<3>,"x",kvi) * w;
            //             vtemp.tuple(dim_c<3>,"x",voffset + kvi) = pre_kvert;
            //             vtemp("minv",voffset + kvi) = 0;  
            //             vtemp("m",voffset + kvi) = (T)1000;
            //             vtemp.tuple(dim_c<3>,"v",voffset + kvi) = (cur_kvert - pre_kvert) * boundary_velocity_scale;
            //             // if(hasKCollisionCancel)
            //             //     vtemp("collision_cancel",voffset + kvi) = kverts("collision_cancel",kvi);
            //     });            
            // }

            auto add_repulsion_force = get_input2<bool>("add_repulsion_force");

            const auto& dp_count = (*zsparticles)[DCD_COUNTER_BUFFER];

            // std::cout << "nm_dcd_collisions : " << nm_dcd_collisions << std::endl;

            for(auto iter = 0;iter != nm_smooth_iters;++iter) {
                
                // cudaPol(zs::range(verts.size()),[
                //     dp_buffer = proxy<space>(dp_buffer)] ZS_LAMBDA(int vi) mutable {
                //         for(int d = 0;d != 3;++d)
                //             dp_buffer[vi * 3 + d] = 0;
                //         // dp_count[vi] = 0;
                // });
                TILEVEC_OPS::fill(cudaPol,verts,dptag,(T)0);

                cudaPol(zs::range(nm_dcd_collisions),[
                    cquads = proxy<space>({},cquads),
                    verts = proxy<space>({},verts),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    pptagOffset = verts.getPropertyOffset(pptag),
                    kverts = proxy<space>({},kverts),
                    exec_tag = exec_tag,
                    eps = eps,
                    pw = pw,
                    w = w,
                    nm_verts = nm_verts,
                    add_repulsion_force = add_repulsion_force,
                    imminent_thickness = imminent_thickness] ZS_LAMBDA(int ci) mutable {
                        auto inds = cquads.pack(dim_c<4>,"inds",ci,int_c);
                        auto bary = cquads.pack(dim_c<4>,"bary",ci);
                        auto type = zs::reinterpret_bits<int>(cquads("type",ci));
                        
                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        vec4 minvs{};
                        vec4 ms{};

                        for(int i = 0;i != 4;++i) {
                            if(inds[i] < nm_verts) {
                                ps[i] = verts.pack(dim_c<3>,pptagOffset,inds[i]);
                                vs[i] = verts.pack(dim_c<3>,ptagOffset,inds[i]) - verts.pack(dim_c<3>,pptagOffset,inds[i]);
                                minvs[i] = verts("minv",inds[i]);
                                ms[i] = verts("m",inds[i]);
                            } else {
                                auto kp = kverts.pack(dim_c<3>,"x",(size_t)(inds[i] - nm_verts));
                                auto kpp = kverts.pack(dim_c<3>,"px",(size_t)(inds[i] - nm_verts));
                                auto pre_kvert = kpp * (1 - pw) + kp * pw;
                                auto cur_kvert = kpp * (1 - w) + kp * w;
                                ps[i] = pre_kvert;
                                vs[i] = cur_kvert - pre_kvert;
                                minvs[i] = (T)0;
                                ms[i] = (T)1000;
                            }
                        }

                        vec3 imps[4] = {};
                        if(!COLLISION_UTILS::compute_imminent_collision_impulse(ps,
                                vs,
                                bary,
                                ms,
                                minvs,
                                imps,
                                imminent_thickness,
                                type,
                                add_repulsion_force))
                            return;
                        for(int i = 0;i != 4;++i) {
                            if(minvs[i] < eps || inds[i] >= nm_verts)
                                continue;
                            

                            // if(isnan(imps[i].norm())) {
                            //     printf("nan imps detected : %f %f %f %f %f %f %f\nvs : %d %d %d %d\n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
                            //         (float)imps[i][0],(float)imps[i][1],(float)imps[i][2],
                            //         (float)bary[0],(float)bary[1],(float)bary[2],(float)bary[3],
                            //         inds[0],inds[1],inds[2],inds[3],
                            //         (float)ps[0][0],(float)ps[0][1],(float)ps[0][2],
                            //         (float)ps[1][0],(float)ps[1][1],(float)ps[1][2],
                            //         (float)ps[2][0],(float)ps[2][1],(float)ps[2][2],
                            //         (float)ps[3][0],(float)ps[3][1],(float)ps[3][2]);
                            //     return;
                            // }
                            // atomic_add(exec_tag,&dp_count[inds[i]],(int)1);
                            // printf("imps : %f\n",(float)imps[i].norm());
                            for(int d = 0;d != 3;++d)
                                atomic_add(exec_tag,&verts(dptagOffset + d,inds[i]),imps[i][d]);
                        }
                });

                // auto ndp = TILEVEC_OPS::dot<3>(cudaPol,verts,dptag,dptag);
                // std::cout << "ndp : " << ndp << std::endl;

                cudaPol(zs::range(verts.size()),[
                    verts = proxy<space>({},verts),
                    relaxs = relaxs,
                    dp_count = proxy<space>({},dp_count),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    ptagOffset = verts.getPropertyOffset(ptag)] ZS_LAMBDA(int vi) mutable {
                        if(dp_count("cnt",vi) > 0.5) {
                            // auto dp = relaxs * vec3{dp_buffer[vi * 3 + 0],dp_buffer[vi * 3 + 1],dp_buffer[vi * 3 + 2]};
                            auto dp = verts.pack(dim_c<3>,dptagOffset,vi) * relaxs;
                            // printf("update %d : %f %f\n",vi,(float)dp.norm(),dp_count("cnt",vi));
                            verts.tuple(dim_c<3>,ptagOffset,vi) = verts.pack(dim_c<3>,ptagOffset,vi) + dp / (T)dp_count("cnt",vi);
                        }
                });
            }

            // cudaPol(zs::range(verts.size()),[
            //     verts = proxy<space>({},verts),
            //     vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
            //         verts.tuple(dim_c<3>,"x",vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
            // });
        
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
                                {"string","ptag","x"},
                                {"string","pptag","px"},
                                {"string","dptag","dx"}
                            }, 
                            {{"zsparticles"}},
                            {},
                            {"PBD"}});

struct XPBDSolveSmoothDCD : INode {

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

        auto constraints = get_input<ZenoParticles>("constraints");
        auto relaxs = get_input2<float>("relaxation_strength");

        auto& verts = zsparticles->getParticles();
        auto nm_smooth_iters = get_input2<int>("nm_smooth_iters");

        if(!verts.hasProperty("dx"))
            verts.append_channels(cudaPol,{{"dx",3}});
        if(!verts.hasProperty("w"))
            verts.append_channels(cudaPol,{{"w",1}});

        auto category = constraints->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});

        if(category == category_c::dcd_collision_constraint) {
            constexpr auto eps = 1e-6;

            const auto& cquads = constraints->getQuadraturePoints();
            if(!constraints->hasMeta(NM_DCD_COLLISIONS))
                return;

            auto nm_dcd_collisions = constraints->readMeta<size_t>(NM_DCD_COLLISIONS);
            auto imminent_thickness = constraints->readMeta<float>(GLOBAL_DCD_THICKNESS);

            auto has_input_collider = constraints->hasMeta(CONSTRAINT_TARGET);

            auto substep_id = get_input2<int>("substep_id");
            auto nm_substeps = get_input2<int>("nm_substeps");
            auto w = (float)(substep_id + 1) / (float)nm_substeps;
            auto pw = (float)(substep_id) / (float)nm_substeps;

            const auto& kverts = has_input_collider ? constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{})->getParticles() : 
                    ZenoParticles::particles_t{verts.get_allocator(),{
                        {"x",3},
                        {"v",3},
                        {"minv",1},
                        {"m",1}
                    },(size_t)0};

            auto pptag = get_input2<std::string>("pptag");

            auto add_repulsion_force = get_input2<bool>("add_repulsion_force");

            for(auto iter = 0;iter != nm_smooth_iters;++iter) {
                TILEVEC_OPS::fill(cudaPol,verts,"dx",0);
                TILEVEC_OPS::fill(cudaPol,verts,"w",0);

                cudaPol(zs::range(nm_dcd_collisions),[
                    kvertsOffset = verts.size(),
                    cquadsIndsOffset = cquads.getPropertyOffset("inds"),
                    cquadsBaryOffset = cquads.getPropertyOffset("bary"),
                    cquadsTypeOffset = cquads.getPropertyOffset("type"),
                    cquads = proxy<space>({},cquads),
                    wTagOffset = verts.getPropertyOffset("w"),
                    dxTagOffset = verts.getPropertyOffset("dx"),
                    xTagOffset = verts.getPropertyOffset("x"),
                    pxTagOffset = verts.getPropertyOffset(pptag),
                    minvOffset = verts.getPropertyOffset("minv"),
                    mOffset = verts.getPropertyOffset("m"),
                    verts = proxy<space>({},verts),
                    kxTagOffset = kverts.getPropertyOffset("x"),
                    kpxTagOffset = kverts.getPropertyOffset("px"),
                    kverts = proxy<space>({},kverts),
                    w = w,
                    pw = pw,
                    exec_tag = exec_tag,
                    eps = eps,
                    add_repulsion_force = add_repulsion_force,
                    imminent_thickness = imminent_thickness] ZS_LAMBDA(int ci) mutable {
                        auto inds = cquads.pack(dim_c<4>,cquadsIndsOffset,ci,int_c);
                        auto bary = cquads.pack(dim_c<4>,cquadsBaryOffset,ci);
                        auto type = zs::reinterpret_bits<int>(cquads(cquadsTypeOffset,ci));
                        
                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        vec4 minvs{};
                        vec4 ms{};

                        for(int i = 0;i != 4;++i) {
                            if(inds[i] < kvertsOffset)  {
                                auto vi = inds[i];
                                ps[i] = verts.pack(dim_c<3>,pxTagOffset,vi);
                                vs[i] = verts.pack(dim_c<3>,xTagOffset,vi) - verts.pack(dim_c<3>,pxTagOffset,vi);
                                minvs[i] = verts(minvOffset,vi);
                                ms[i] = verts(mOffset,vi);
                            } else {
                                auto kvi = inds[i] - kvertsOffset;
                                auto pre_kvert = kverts.pack(dim_c<3>,kpxTagOffset,kvi) * (1 - pw) + kverts.pack(dim_c<3>,kxTagOffset,kvi) * pw;
                                auto cur_kvert = kverts.pack(dim_c<3>,kpxTagOffset,kvi) * (1 - w) + kverts.pack(dim_c<3>,kxTagOffset,kvi) * w;
                                ps[i] = pre_kvert;
                                vs[i] = cur_kvert - pre_kvert;
                                minvs[i] = 0;
                                ms[i] = 1000;
                            }
                        }

                        vec3 imps[4] = {};
                        if(!COLLISION_UTILS::compute_imminent_collision_impulse(ps,
                                vs,
                                bary,
                                ms,
                                minvs,
                                imps,
                                imminent_thickness,
                                type,
                                add_repulsion_force))
                            return;
                        for(int i = 0;i != 4;++i) {
                            if(minvs[i] < eps)
                                continue;

                            if(imps[i].norm() < 1e-6)
                                continue;

                            atomic_add(exec_tag,&verts(wTagOffset,inds[i]),(float)1);
                            for(int d = 0;d != 3;++d)
                                atomic_add(exec_tag,&verts(dxTagOffset + d,inds[i]),imps[i][d]);
                        }
                });

                cudaPol(zs::range(verts.size()),[
                    verts = proxy<space>({},verts),relaxs = relaxs,
                    xTagOffset = verts.getPropertyOffset("x"),
                    dxTagOffset = verts.getPropertyOffset("dx"),
                    wTagOffset = verts.getPropertyOffset("w")] ZS_LAMBDA(int vi) mutable {
                        if(verts(wTagOffset,vi) > 0) {
                            auto dp = relaxs * verts.pack(dim_c<3>,dxTagOffset,vi) / verts(wTagOffset,vi);
                            // auto dp = relaxs * verts.pack(dim_c<3>,dxTagOffset,vi) * (float)0.25;
                            verts.tuple(dim_c<3>,xTagOffset,vi) = verts.pack(dim_c<3>,xTagOffset,vi) + dp;
                        }
                });

            }
        
        }

        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolveSmoothDCD, {{{"zsparticles"},
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

        using vec2 = zs::vec<float,2>;
        using vec3 = zs::vec<float,3>;
        using vec4 = zs::vec<float,4>;
        using mat3 = zs::vec<float,3,3>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = 1e-6;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");

        auto constraint_ptr_list = RETRIEVE_OBJECT_PTRS(ZenoParticles, "constraints");

        auto& verts = zsparticles->getParticles();
        auto ptag = get_input2<std::string>("ptag");
        auto pptag = get_input2<std::string>("pptag");
        auto dptag = get_input2<std::string>("dptag");
        auto dt = get_input2<float>("dt");


        if(!verts.hasProperty("w")) {
            verts.append_channels(cudaPol,{{"w",1}});
        }
        TILEVEC_OPS::fill(cudaPol,verts,"w",0);
        if(!verts.hasProperty(dptag))
            verts.append_channels(cudaPol,{{dptag,3}});
        TILEVEC_OPS::fill(cudaPol,verts,dptag,0);

        auto iter_id = get_input2<int>("iter_id");

        for(auto& constraint_ptr : constraint_ptr_list) {
            auto category = constraint_ptr->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});
            const auto& cquads = constraint_ptr->getQuadraturePoints();


            if(constraint_ptr->userData().has("stride")) {
                auto stride = objectToLiterial<int>(constraint_ptr->userData().get("stride"));
                // std::cout << "find constraint with stride = " << stride << std::endl;
                // if(stride <= 0 && iter_id != 0)
                //     continue;
                if(iter_id % stride != 0) {
                    // std::cout << "skip constraint solving due to stride-skipping" << std::endl;
                    continue;
                }
            } else {
                // std::cout << "the constraint has no stride information" << std::endl;
            }

            if(category == category_c::shape_matching_constraint) {
                auto shape_matching_rest_cm = constraint_ptr->readMeta<std::vector<vec3>>(SHAPE_MATCHING_REST_CM);
                auto shape_matching_weight_sum = constraint_ptr->readMeta<std::vector<float>>(SHAPE_MATCHING_WEIGHT_SUM);
                auto shape_matching_offsets = constraint_ptr->readMeta<zs::Vector<int>>(SHAPE_MATCHING_SHAPE_OFFSET);
                auto nm_shapes = shape_matching_rest_cm.size();

                auto dAs = constraint_ptr->readMeta<zs::Vector<mat3>>(SHAPE_MATCHING_MATRIX_BUFFER);

                zs::Vector<vec3> cmVec{verts.get_allocator(),1};
                
                // auto wsum = constraint_ptr->readMeta<float>(SHAPE_MATCHING_WEIGHT_SUM);
                // auto restCM = constraint_ptr->readMeta<vec3>(SHAPE_MATCHING_REST_CM);
                for(int shape_id = 0;shape_id != nm_shapes;++shape_id) {
                    auto shape_size = shape_matching_offsets.getVal(shape_id + 1) - shape_matching_offsets.getVal(shape_id);
                    if(shape_size == 0)
                        continue;

                    auto restCM = shape_matching_rest_cm[shape_id];
                    auto wsum = shape_matching_weight_sum[shape_id];

                    cmVec.setVal(vec3::zeros());
                    cudaPol(zs::range(shape_size),[
                        exec_tag = exec_tag,
                        verts = proxy<space>({},verts),
                        offset = shape_matching_offsets.getVal(shape_id),
                        ptagOffset = verts.getPropertyOffset(ptag),
                        minvOffset = verts.getPropertyOffset("minv"),
                        indsOffset = cquads.getPropertyOffset("inds"),
                        cquads = proxy<space>({},cquads),
                        cmVec = proxy<space>(cmVec)] ZS_LAMBDA(int ci) mutable {
                            auto vi = zs::reinterpret_bits<int>(cquads(indsOffset,ci + offset));
                            auto pi = verts.pack(dim_c<3>,ptagOffset,vi);
                            auto wi = static_cast<T>(1.0) / (static_cast<T>(1e-6) + verts(minvOffset,vi));
                            auto pw = pi * wi;
                            for(int d = 0;d != 3;++d)
                                atomic_add(exec_tag,&cmVec[0][d],pw[d]); 
                    });

                    auto cm = cmVec.getVal(0) / wsum;
                    // dAs.setVal(mat3::zeros());

                    cudaPol(zs::range(shape_size),[
                        offset = shape_matching_offsets.getVal(shape_id),
                        cquads = proxy<space>({},cquads),
                        indsOffset = cquads.getPropertyOffset("inds"),
                        verts = proxy<space>({},verts),
                        XtagOffset = verts.getPropertyOffset("X"),
                        minvOffset = verts.getPropertyOffset("minv"),
                        ptagOffset = verts.getPropertyOffset(ptag),
                        restCM = restCM,
                        cm = cm,
                        dAs = proxy<space>(dAs)] ZS_LAMBDA(int ci) mutable {
                            auto vi = zs::reinterpret_bits<int>(cquads(indsOffset,ci + offset));
                            auto q = verts.pack(dim_c<3>,XtagOffset,vi) - restCM;
                            auto p = verts.pack(dim_c<3>,ptagOffset,vi) - cm;
                            auto w = static_cast<float>(1.0) / (verts(minvOffset,vi) + static_cast<float>(1e-6));
                            p *= w;
                            dAs[ci + offset] = dyadic_prod(p,q);
                    });

                    zs::Vector<mat3> A{verts.get_allocator(),1};
                    A.setVal(mat3::zeros());

                    cudaPol(zs::range(shape_size * 9),[
                        exec_tag = exec_tag,
                        offset = shape_matching_offsets.getVal(shape_id),
                        A = proxy<space>(A),
                        dAs = proxy<space>(dAs)] ZS_LAMBDA(int dof) mutable {
                            auto dAid = dof / 9;
                            auto Aoffset = dof % 9;
                            auto r = Aoffset / 3;
                            auto c = Aoffset % 3;
                            const auto& dA = dAs[dAid + offset];
                            atomic_add(exec_tag,&A[0][r][c],dA[r][c]);
                    });

                    auto Am = A.getVal(0);
                    Am /= wsum;

                    auto [R,S] = math::polar_decomposition(Am);
                    cudaPol(zs::range(shape_size),[
                        offset = shape_matching_offsets.getVal(shape_id),
                        cquads = proxy<space>({},cquads),
                        stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                        R = R,
                        cm = cm,
                        restCM = restCM,
                        wOffset = verts.getPropertyOffset("w"),
                        dptagOffset = verts.getPropertyOffset(dptag),
                        verts = proxy<space>({},verts),
                        XtagOffset = verts.getPropertyOffset("X"),
                        ptagOffset = verts.getPropertyOffset(ptag)] ZS_LAMBDA(int ci) mutable {
                            auto vi = zs::reinterpret_bits<int>(cquads("inds",ci + offset));
                            auto Xi = verts.pack(dim_c<3>,XtagOffset,vi);
                            auto w = cquads(stiffnessOffset,ci + offset);
                            auto goal = cm + R * (Xi - restCM);

                            auto dp = goal - verts.pack(dim_c<3>,ptagOffset,vi);

                            verts.tuple(dim_c<3>,dptagOffset,vi) = verts.pack(dim_c<3>,dptagOffset,vi) + dp * w;
                            verts(wOffset,vi) += w;
                    });

                }
            
            }

            if(category == category_c::edge_length_constraint || category == category_c::dihedral_bending_constraint || category == category_c::long_range_attachment) {
                // if(category == category_c::edge_length_constraint)
                //     std::cout << "solve edge length constraint" << std::endl;
                // if(category == category_c::dihedral_bending_constraint)
                //     std::cout << "solve dihedral bending constraint" << std::endl;
                cudaPol(zs::range(cquads.size()),[
                    cquads = proxy<space>({},cquads),
                    dt = dt,
                    stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                    affiliationOffset = cquads.getPropertyOffset("xpbd_affiliation"),
                    dampingOffset = cquads.getPropertyOffset("damping_coeff"),
                    indsOffset = cquads.getPropertyOffset("inds"),
                    rOffset = cquads.getPropertyOffset("r"),
                    // restScaleOffset = cquads.getPropertyOffset("rest_scale"),
                    // weight_sum = proxy<space>(weight_sum),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    pptagOffset = verts.getPropertyOffset(pptag),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    minvOffset = verts.getPropertyOffset("minv"),
                    wOffset = verts.getPropertyOffset("w"),
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
                            // auto rest_scale = cquads(restScaleOffset,ci);
                            // auto r = cquads("r",ci) * rest_scale;
                            auto r = cquads(rOffset,ci);
                            vec3 dp[2] = {};

                            auto lambda = (T)0;
                            bool do_stretch_resistence_only = (category == category_c::long_range_attachment);
                            if(!CONSTRAINT::solve_DistanceConstraint(
                                p0,minv0,
                                p1,minv1,
                                pp0,pp1,
                                r,
                                aff,
                                kd,
                                dt,
                                lambda,
                                dp[0],dp[1],
                                do_stretch_resistence_only))
                                    return;
                            // printf("smooth stretch update : %f %f\n",(float)dp[0].norm(),(float)dp[1].norm());
                            for(int i = 0;i != 2;++i) {
                                // if(isnan(dp[i].norm()))
                                //     printf("nan dp[%d] detected at bending\n",i);
                                // atomic_add(exec_tag,&weight_sum[edge[i]],w);
                                atomic_add(exec_tag,&verts(wOffset,edge[i]),w);
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
                            // auto rest_scale = cquads(restScaleOffset,ci);
                            // auto ra = cquads("ra",ci) * rest_scale;
                            // auto ras = cquads("sign",ci);
                            auto ra = cquads(rOffset,ci);
                            vec3 dp[4] = {};
                            auto lambda = (T)0;
                            if(!CONSTRAINT::solve_DihedralConstraint(
                                p[0],minv[0],
                                p[1],minv[1],
                                p[2],minv[2],
                                p[3],minv[3],
                                pp[0],pp[1],pp[2],pp[3],
                                ra,
                                // ras,
                                aff,
                                dt,
                                kd,
                                lambda,
                                dp[0],dp[1],dp[2],dp[3]))
                                    return;    
                            for(int i = 0;i != 4;++i) {
                                // if(isnan(dp[i].norm()))
                                //     printf("nan dp[%d] detected at bending\n",i);
                                // atomic_add(exec_tag,&weight_sum[quad[i]],w);
                                atomic_add(exec_tag,&verts(wOffset,quad[i]),w);
                                for(int d = 0;d != 3;++d)
                                    atomic_add(exec_tag,&verts(dptagOffset + d,quad[i]),dp[i][d] * w);
                            }                        
                        }
                });
            }
            
            if(category == category_c::vertex_pin_to_vertex_constraint) {
                auto target = constraint_ptr->readMeta<ZenoParticles*>(CONSTRAINT_TARGET);
                const auto& kverts = target->getParticles();

                auto substep_id = get_input2<int>("substep_id");
                auto nm_substeps = get_input2<int>("nm_substeps");
                auto anim_w = (float)(substep_id + 1) / (float)nm_substeps;
                auto anim_pw = (float)(substep_id) / (float)nm_substeps;

                cudaPol(zs::range(cquads.size()),[
                    cquads = proxy<space>({},cquads),
                    stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                    affiliationOffset = cquads.getPropertyOffset("xpbd_affiliation"),
                    dampingOffset = cquads.getPropertyOffset("damping_coeff"),
                    cquadsIndsOffset = cquads.getPropertyOffset("inds"),
                    cquadsROffset = cquads.getPropertyOffset("r"),
                    dt = dt,
                    anim_w = anim_w,
                    anim_pw = anim_pw,
                    verts = view<space>(verts),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    pptagOffset = verts.getPropertyOffset(pptag),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    minvOffset = verts.getPropertyOffset("minv"),
                    wOffset = verts.getPropertyOffset("w"),
                    kverts = proxy<space>({},kverts),
                    kptagOffset = kverts.getPropertyOffset("x"),
                    kpptagOffset = kverts.getPropertyOffset("px")] ZS_LAMBDA(int ei) mutable {
                        auto w = cquads(stiffnessOffset,ei);
                        auto inds = cquads.pack(dim_c<2>,cquadsIndsOffset,ei,int_c);
                        auto r = cquads(cquadsROffset,ei);
                        auto aff = cquads(affiliationOffset,ei);
                        auto kd = cquads(dampingOffset,ei);

                        auto vi = inds[0];
                        auto kvi = inds[1];
                        if(kvi < 0)
                            return;

                        auto p = verts.pack(dim_c<3>,ptagOffset,vi);
                        auto pp = verts.pack(dim_c<3>,pptagOffset,vi);
                        auto kp = anim_w * kverts.pack(dim_c<3>,kptagOffset,kvi) + (1.f - anim_w) * kverts.pack(dim_c<3>,kpptagOffset,kvi);
                        auto kpp = anim_pw * kverts.pack(dim_c<3>,kptagOffset,kvi) + (1.f - anim_pw) * kverts.pack(dim_c<3>,kpptagOffset,kvi);

                        auto minv = verts(minvOffset,vi);
                        if(minv < 1e-6)
                            return;
                        auto kminv = 0.f;

                        auto lambda = (T)0.0;
                        bool do_stretch_resistence_only = true;
                        vec3 dp[2] = {};

                        if(!CONSTRAINT::solve_DistanceConstraint(
                            p,minv,
                            kp,kminv,
                            pp,kpp,
                            r,
                            aff,
                            kd,
                            dt,
                            lambda,
                            dp[0],dp[1],
                            do_stretch_resistence_only))
                                return; 

                        // atomic_add(exec_tag,&verts(wOffset,vi),w);
                        verts(wOffset,vi) += w;
                        verts.tuple(dim_c<3>,dptagOffset,vi) = verts.pack(dim_c<3>,dptagOffset,vi) + dp[0];
                        // for(int d = 0;d != 3;++d)
                        //     atomic_add(exec_tag,&verts(dptagOffset + d,vi),dp[0][d]);
                });
            }


            if(category == category_c::self_dcd_collision_constraint) {
                const auto& tris = zsparticles->getQuadraturePoints();
                const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 
                auto imminent_thickness = constraint_ptr->readMeta<float>(GLOBAL_DCD_THICKNESS);
                auto enable_repulsion_force = constraint_ptr->readMeta<bool>(ENABLE_DCD_REPULSION_FORCE);
                auto nm_dcd_collisions = constraint_ptr->readMeta<size_t>(NM_DCD_COLLISIONS);

                cudaPol(zs::range(nm_dcd_collisions),[
                    imminent_thickness = imminent_thickness,
                    add_repulsion_force = enable_repulsion_force,
                    cquads = proxy<space>({},cquads),
                    dt = dt,
                    stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                    affiliationOffset = cquads.getPropertyOffset("xpbd_affiliation"),
                    dampingOffset = cquads.getPropertyOffset("damping_coeff"),
                    indsOffset = cquads.getPropertyOffset("inds"),
                    baryOffset = cquads.getPropertyOffset("bary"),
                    typeOffset = cquads.getPropertyOffset("type"),
                    // weight_sum = proxy<space>(weight_sum),
                    wOffset = verts.getPropertyOffset("w"),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    pptagOffset = verts.getPropertyOffset(pptag),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    minvOffset = verts.getPropertyOffset("minv"),
                    mOffset = verts.getPropertyOffset("m"),
                    verts = proxy<space>({},verts),
                    exec_tag = exec_tag] ZS_LAMBDA(int ci) mutable {
                        // auto w = (T)1.0;
                        auto w = cquads(stiffnessOffset,ci);
                        auto inds = cquads.pack(dim_c<4>,indsOffset,ci,int_c);
                        auto bary = cquads.pack(dim_c<4>,baryOffset,ci);
                        auto type = zs::reinterpret_bits<int>(cquads(typeOffset,ci));
                        
                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        vec4 minvs{};
                        vec4 ms{};        
                        
                        for(int i = 0;i != 4;++i) {
                            ps[i] = verts.pack(dim_c<3>,ptagOffset,inds[i]);
                            vs[i] = ps[i] - verts.pack(dim_c<3>,pptagOffset,inds[i]);
                            minvs[i] = verts(minvOffset,inds[i]);
                            ms[i] = verts(mOffset,inds[i]);
                        }

                        vec3 dp[4] = {};
                        if(!COLLISION_UTILS::compute_imminent_collision_impulse(ps,
                                vs,
                                bary,
                                ms,
                                minvs,
                                dp,
                                imminent_thickness,
                                type,
                                add_repulsion_force)) {
                            return;
                        }

                        // auto imp_norm2 = (T)0.0;
                        for(int i = 0;i != 4;++i) {
                            if(minvs[i] < eps)
                                continue;

                            if(isnan(dp[i].norm())) {
                                printf("nan dcd dp detected : %f %f %f %f %f %f %f\n",
                                    (float)dp[i][0],(float)dp[i][1],(float)dp[i][2],
                                    (float)bary[0],(float)bary[1],(float)bary[2],(float)bary[3]);
                                return;
                            }
                            // atomic_add(exec_tag,&weight_sum[inds[i]],w);
                            atomic_add(exec_tag,&verts(wOffset,inds[i]),w);
                            for(int d = 0;d != 3;++d){
                                atomic_add(exec_tag,&verts(dptagOffset + d,inds[i]),dp[i][d] * w);
                            }
                        }
                });
            }

            if(category == category_c::vertex_pin_to_cell_constraint) {
                // std::cout << "solve vertex cell pin constraint" << std::endl;

                auto target = constraint_ptr->readMeta<ZenoParticles*>(CONSTRAINT_TARGET);
                const auto& kverts = target->getParticles();
                const auto& ktris = target->getQuadraturePoints();

                auto enable_sliding = constraint_ptr->readMeta<bool>(ENABLE_SLIDING);

                auto substep_id = get_input2<int>("substep_id");
                auto nm_substeps = get_input2<int>("nm_substeps");
                auto anim_w = (float)(substep_id + 1) / (float)nm_substeps;

                auto& cell_buffer = (*target)[TARGET_CELL_BUFFER];
        
                cudaPol(zs::range(cell_buffer.size()),[
                    cell_buffer = proxy<space>({},cell_buffer),
                    kverts = proxy<space>({},kverts),
                    w = anim_w] ZS_LAMBDA(int vi) mutable {
                        cell_buffer.tuple(dim_c<3>,"cx",vi) = w * kverts.pack(dim_c<3>,"x",vi) + (1 - w) * kverts.pack(dim_c<3>,"px",vi);
                });

                auto thickness = constraint_ptr->readMeta<float>(GLOBAL_DCD_THICKNESS);

                compute_cells_and_vertex_normal(cudaPol,
                    cell_buffer,"cx",
                    cell_buffer,
                    ktris,
                    cell_buffer,
                    thickness);    
                    
                cudaPol(zs::range(cquads.size()),[
                    cquads = proxy<space>({},cquads),
                    cell_buffer = proxy<space>({},cell_buffer),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    cquadsIndsOffset = cquads.getPropertyOffset("inds"),
                    cquadsBaryOffset = cquads.getPropertyOffset("bary"),
                    cellBufferXOffset = cell_buffer.getPropertyOffset("x"),
                    cellBufferVOffset = cell_buffer.getPropertyOffset("v"),
                    ktris = ktris.begin("inds",dim_c<3>,int_c),
                    enable_sliding = enable_sliding,
                    wOffset = verts.getPropertyOffset("w"),
                    stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                    verts = proxy<space>({},verts)] ZS_LAMBDA(int ci) mutable {
                        auto w = cquads(stiffnessOffset,ci);
                        auto pair = cquads.pack(dim_c<2>,cquadsIndsOffset,ci,int_c);
                        auto vi = pair[0];
                        auto kti = pair[1];
                        auto ktri = ktris[kti];

                        auto bary = cquads.pack(dim_c<6>,cquadsBaryOffset,ci);
                        vec3 as[3] = {};
                        vec3 bs[3] = {};

                        for(int i = 0;i != 3;++i) {
                            as[i] = cell_buffer.pack(dim_c<3>,cellBufferXOffset,ktri[i]);
                            bs[i] = cell_buffer.pack(dim_c<3>,cellBufferVOffset,ktri[i]) + as[i];
                        }

                        auto tp = vec3::zeros();
                        for(int i = 0;i != 3;++i) {
                            tp += as[i] * bary[i];
                            tp += bs[i] * bary[i + 3];
                        }

                        auto dp = tp - verts.pack(dim_c<3>,ptagOffset,vi);


                        for(int d = 0;d != 3;++d){
                            atomic_add(exec_tag,&verts(dptagOffset + d,vi),dp[d]);
                        }
                });
            }

            if(category == category_c::follow_animation_constraint) {
                auto use_hard_constraint = constraint_ptr->readMeta<bool>(PBD_USE_HARD_CONSTRAINT);
                if(use_hard_constraint && iter_id > 0)
                    continue;

                auto animation = constraint_ptr->readMeta<ZenoParticles*>(CONSTRAINT_TARGET);
                const auto& averts = animation->getParticles();
                auto substep_id = get_input2<int>("substep_id");
                auto nm_substeps = get_input2<int>("nm_substeps");
                auto anim_w = (float)(substep_id + 1) / (float)nm_substeps;

                cudaPol(zs::range(cquads.size()),[
                    cquads = proxy<space>({},cquads),
                    verts = proxy<space>({},verts),
                    alpha = anim_w,
                    wOffset = verts.getPropertyOffset("w"),
                    averts = proxy<space>({},averts),
                    aPtagOffset = averts.getPropertyOffset("x"),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    apPtagOffset = averts.getPropertyOffset("px"),
                    use_hard_constraint = use_hard_constraint,
                    ptagOffset = verts.getPropertyOffset(ptag),
                    followWeightOffset = cquads.getPropertyOffset("follow_weight"),
                    indsOffset = cquads.getPropertyOffset("inds")] ZS_LAMBDA(int ei) mutable {
                        auto vi = zs::reinterpret_bits<int>(cquads(indsOffset,ei));
                        auto w = cquads(followWeightOffset,ei);
                        auto p = verts.pack(dim_c<3>,ptagOffset,vi);
                        auto tp = averts.pack(dim_c<3>,apPtagOffset,vi) * (1.f - alpha) + averts.pack(dim_c<3>,aPtagOffset,vi) * alpha;
                        if(use_hard_constraint)
                            verts.tuple(dim_c<3>,ptagOffset,vi) = tp;
                        else {
                            auto bp = tp * w + p * (1.f - w);
                            auto dp = bp - p;
                            atomic_add(exec_tag,&verts(wOffset,vi),w);
                            for(int d = 0;d != 3;++d){
                                atomic_add(exec_tag,&verts(dptagOffset + d,vi),dp[d] * w);
                            }
                        }
                });
            }

            if(category == category_c::volume_pin_constraint) {
                // std::cout << "solve volume pin constraint " << std::endl;
                auto use_hard_constraint = constraint_ptr->readMeta<bool>(PBD_USE_HARD_CONSTRAINT);
                if(use_hard_constraint && iter_id > 0)
                    continue;


                auto embed_volume = constraint_ptr->readMeta<ZenoParticles*>(CONSTRAINT_TARGET);
                const auto& vverts = embed_volume->getParticles();
                const auto& vtets = embed_volume->getQuadraturePoints();


                auto substep_id = get_input2<int>("substep_id");
                auto nm_substeps = get_input2<int>("nm_substeps");
                auto volume_anim_w = (float)(substep_id + 1) / (float)nm_substeps;
                // auto pw = (float)(substep_id) / (float)nm_substeps;

                if(!vverts.hasProperty("px")) {
                    throw std::runtime_error("the vverts has no px channel");
                }

                cudaPol(zs::range(cquads.size()),[
                    cquads = proxy<space>({},cquads),
                    verts = proxy<space>({},verts),
                    alpha = volume_anim_w,
                    stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                    // weight_sum = proxy<space>(weight_sum),
                    use_hard_constraint = use_hard_constraint,
                    wOffset = verts.getPropertyOffset("w"),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    vverts = proxy<space>({},vverts),
                    vtets = proxy<space>({},vtets)] ZS_LAMBDA(int ci) mutable {
                        auto w = cquads(stiffnessOffset,ci);
                        auto pair = cquads.pack(dim_c<2>,"inds",ci,int_c);
                        auto bary = cquads.pack(dim_c<4>,"bary",ci);
                        auto vi = pair[0];
                        auto vti = pair[1];
                        if(vti < 0)
                            return;

                        auto vtet = vtets.pack(dim_c<4>,"inds",vti,int_c);
                        vec3 vps[4] = {};
                        for(int i = 0;i != 4;++i)
                            vps[i] = (1 - alpha) * vverts.pack(dim_c<3>,"px",vtet[i]) + alpha * vverts.pack(dim_c<3>,"x",vtet[i]);

                        auto vtp = vec3::zeros();
                        for(int i = 0;i != 4;++i)
                            vtp += vps[i] * bary[i];
                        
                        if(use_hard_constraint) {
                            verts.tuple(dim_c<3>,ptagOffset,vi) = vtp;
                            // atomic_add(exec_tag,&weight_sum[vi],w);
                            // weight_sum[vi] = (T)1.0;
                            verts(wOffset,vi) = (T)1.0;
                        } else  {
                            auto dp = vtp - verts.pack(dim_c<3>,ptagOffset,vi);

                            atomic_add(exec_tag,&verts(wOffset,vi),w);
                            for(int d = 0;d != 3;++d){
                                atomic_add(exec_tag,&verts(dptagOffset + d,vi),dp[d] * w);
                            }
                        }
                });
            }

            if(category == category_c::kinematic_dcd_collision_constraint) {
                const auto& tris = zsparticles->getQuadraturePoints();
                const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 
                auto imminent_thickness = constraint_ptr->readMeta<float>(GLOBAL_DCD_THICKNESS);
                auto enable_repulsion_force = constraint_ptr->readMeta<bool>(ENABLE_DCD_REPULSION_FORCE);
                
                cudaPol(zs::range(cquads.size()),[
                    imminent_thickness = imminent_thickness,
                    add_repulsion_force = enable_repulsion_force,
                    cquads = proxy<space>({},cquads),
                    tris = tris.begin("inds",dim_c<3>,int_c),
                    edges = edges.begin("inds",dim_c<2>,int_c),
                    dt = dt,
                    stiffnessOffset = cquads.getPropertyOffset("relative_stiffness"),
                    affiliationOffset = cquads.getPropertyOffset("xpbd_affiliation"),
                    dampingOffset = cquads.getPropertyOffset("damping_coeff"),
                    indsOffset = cquads.getPropertyOffset("inds"),
                    baryOffset = cquads.getPropertyOffset("bary"),
                    typeOffset = cquads.getPropertyOffset("type"),
                    hitPointOffset = cquads.getPropertyOffset("hit_point"),
                    hitVelocityOffset = cquads.getPropertyOffset("hit_velocity"),
                    // weight_sum = proxy<space>(weight_sum),
                    wOffset = verts.getPropertyOffset("w"),
                    ptagOffset = verts.getPropertyOffset(ptag),
                    pptagOffset = verts.getPropertyOffset(pptag),
                    dptagOffset = verts.getPropertyOffset(dptag),
                    minvOffset = verts.getPropertyOffset("minv"),
                    mOffset = verts.getPropertyOffset("m"),
                    verts = view<space>(verts),
                    exec_tag = exec_tag] ZS_LAMBDA(int ci) mutable {
                        // auto w = (T)1.0;
                        auto w = cquads(stiffnessOffset,ci);
                        auto inds = cquads.pack(dim_c<4>,indsOffset,ci,int_c);
                        auto bary = cquads.pack(dim_c<4>,baryOffset,ci);
                        auto type = zs::reinterpret_bits<int>(cquads(typeOffset,ci));   
                        
                        auto hit_point = cquads.pack(dim_c<3>,hitPointOffset,ci);
                        auto hit_velocity = cquads.pack(dim_c<3>,hitVelocityOffset,ci);

                        vec3 dp[4] = {};

                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        for(int i = 0;i != 4;++i) {
                            ps[i] = hit_point;
                            vs[i] = hit_velocity;
                        }

                        if(type == 0) {// csPKT, inds{ktri[0],ktri[1],ktri[2],vi}
                            auto p = verts.pack(dim_c<3>,pptagOffset,inds[3]);
                            auto v = verts.pack(dim_c<3>,ptagOffset,inds[3]) - p;
                            auto m = verts(mOffset,inds[3]);
                            auto minv = verts(minvOffset,inds[3]);

                            if(minv < eps)
                                return;

                            m = minv < 1e-3 ? 1e3 : m;
                            minv = minv < 1e-3 ? 0 : minv;

                            ps[3] = p;
                            vs[3] = v;

                            if(!COLLISION_UTILS::compute_imminent_collision_impulse(
                                    ps,
                                    vs,
                                    bary,
                                    {1e3,1e3,1e3,m},
                                    {0,0,0,minv},
                                    dp,
                                    imminent_thickness,type,
                                    add_repulsion_force)) {
                                return;
                            }

                            for(int i = 3;i != 4;++i) {
                                // if(minvs[i] < eps)
                                //     continue;
                                if(dp[i].norm() > 2) {
                                    printf("too large impulse detected at kinematic dcd[%d] type[%d] : %f %f %f\n",
                                        i,type,(float)dp[i][0],(float)dp[i][1],(float)dp[i][2]);
                                }
                                // atomic_add(exec_tag,&weight_sum[inds[i]],w);
                                atomic_add(exec_tag,&verts(wOffset,inds[i]),w);
                                for(int d = 0;d != 3;++d){
                                    atomic_add(exec_tag,&verts(dptagOffset + d,inds[i]),dp[i][d] * w);
                                }
                            }   
                        } else if(type == 1) {// csKPT, inds{tri[0],tri[1],tri[2],kvi}
                            vec3 ms{};
                            vec3 minvs{};
                            for(int i = 0;i != 3;++i) {
                                ps[i] = verts.pack(dim_c<3>,pptagOffset,inds[i]);
                                vs[i] = verts.pack(dim_c<3>,ptagOffset,inds[i]) - ps[i];
                                ms[i] = verts(mOffset,inds[i]);
                                minvs[i] = verts(minvOffset,inds[i]);

                                ms[i] = minvs[i] < 1e-3 ? 1e3 : ms[i];
                                minvs[i] = minvs[i] < 1e-3 ? 0 : minvs[i];
                            }

                            vec3 dp[4] = {};
                            if(!COLLISION_UTILS::compute_imminent_collision_impulse(
                                    ps,
                                    vs,
                                    bary,
                                    {ms[0],ms[1],ms[2],1e3},
                                    {minvs[0],minvs[1],minvs[2],0},
                                    dp,
                                    imminent_thickness,type,add_repulsion_force)) {
                                return;
                            }

                            for(int i = 0;i != 3;++i) {
                                if(minvs[i] < eps)
                                    continue;
                                if(dp[i].norm() > 2) {
                                    printf("too large impulse detected at kinematic dcd[%d] type[%d] : %f %f %f\n",
                                        i,type,(float)dp[i][0],(float)dp[i][1],(float)dp[i][2]);
                                }
                                // atomic_add(exec_tag,&weight_sum[inds[i]],w);
                                atomic_add(exec_tag,&verts(wOffset,inds[i]),w);
                                for(int d = 0;d != 3;++d){
                                    atomic_add(exec_tag,&verts(dptagOffset + d,inds[i]),dp[i][d] * w);
                                }
                            }
                        }else if(type == 2){ // csEKE e[0],e[1],ke[0],ke[1],
                            vec2 ms{};
                            vec2 minvs{};
                            for(int i = 0;i != 2;++i) {
                                ps[i] = verts.pack(dim_c<3>,pptagOffset,inds[i]);
                                vs[i] = verts.pack(dim_c<3>,ptagOffset,inds[i]) - ps[i];
                                ms[i] = verts(mOffset,inds[i]);
                                minvs[i] = verts(minvOffset,inds[i]);

                                ms[i] = minvs[i] < 1e-3 ? 1e3 : ms[i];
                                minvs[i] = minvs[i] < 1e-3 ? 0 : minvs[i];
                            }
                            // for the mass of kinematic boundary, 1e3 is suggested, too large mass (e.g 1e5) will bring in instablility、dcd bouncing etc,
                            vec3 dp[4] = {};
                            if(!COLLISION_UTILS::compute_imminent_collision_impulse(
                                    ps,
                                    vs,
                                    bary,
                                    {ms[0],ms[1],1e3,1e3},
                                    {minvs[0],minvs[1],0,0},
                                    dp,
                                    imminent_thickness,type,add_repulsion_force)) {
                                return;
                            }


                            for(int i = 0;i != 2;++i) {
                                if(minvs[i] < eps)
                                    continue;
                                if(dp[i].norm() > 2) {
                                    printf("too large impulse detected at kinematic dcd[%d] type[%d] : %f %f %f\n",
                                        i,type,(float)dp[i][0],(float)dp[i][1],(float)dp[i][2]);
                                }
                                // atomic_add(exec_tag,&weight_sum[inds[i]],w);
                                atomic_add(exec_tag,&verts(wOffset,inds[i]),w);
                                for(int d = 0;d != 3;++d){
                                    atomic_add(exec_tag,&verts(dptagOffset + d,inds[i]),dp[i][d] * w);
                                }
                            }
                        }else {
                            printf("unrecognized type of dcd kinematic collision detected\n");
                            return;
                        }
                    });
            }
        }

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            eps = eps,
            dptagOffset = verts.getPropertyOffset(dptag),
            ptagOffset = verts.getPropertyOffset(ptag),
            wOffset = verts.getPropertyOffset("w")] ZS_LAMBDA(int vi) mutable {
                if(verts(wOffset,vi) > eps)
                    verts.tuple(dim_c<3>,dptagOffset,vi) = verts.pack(dim_c<3>,dptagOffset,vi) / verts(wOffset,vi);
                else
                    verts.tuple(dim_c<3>,dptagOffset,vi) = vec3::zeros();
                verts.tuple(dim_c<3>,ptagOffset,vi) = verts.pack(dim_c<3>,ptagOffset,vi) + 2.f * verts.pack(dim_c<3>,dptagOffset,vi);
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
                                {"int","nm_substeps","1"},
                                {"int","substep_id","0"},
                                {"int","iter_id","0"}
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
