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

#include "../geometry/kernel/tiled_vector_ops.hpp"
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/geo_math.hpp"
#include "../geometry/kernel/global_intersection_analysis.hpp"
#include "../fem/collision_energy/evaluate_collision.hpp"
#include "constraint_function_kernel/constraint_types.hpp"

namespace zeno {


struct SDFColliderProject : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto sdfBoundary = get_input<PrimitiveObject>("sdf_boundary");

        // prex
        auto xtag = get_input2<std::string>("xtag");
        // x
        auto ptag = get_input2<std::string>("ptag");
        auto friction = get_input2<T>("friction");

        // auto collider_type = get_input2<std::string>("sdf_collider_type");

        // auto do_stablize = get_input2<bool>("do_stablize");

        auto& verts = zsparticles->getParticles();

        // if(collider_type == "sdf_sphere") {
        auto radius = sdfBoundary->userData().get2<float>("radius");
        auto center = sdfBoundary->userData().get2<zeno::vec3f>("pos");
        auto cv = zeno::vec3f{0,0,0};
        auto w = zeno::vec3f{0.0,0};
        // auto current_transform = sdfBoundary->userData().get("transform");
        // auto previous_transform = sdfBoundary->userData().get("pre_transform");

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            ptag = zs::SmallString(ptag),
            friction,
            radius,
            center,
            cv,w] ZS_LAMBDA(int vi) mutable {
                if(verts("minv",vi) < (T)1e-6)
                    return;

                auto pred = verts.pack(dim_c<3>,ptag,vi);
                auto pos = verts.pack(dim_c<3>,xtag,vi);


                auto center_vel = vec3::from_array(cv);
                auto center_pos = vec3::from_array(center);
                auto angular_velocity = vec3::from_array(w);

                auto disp = pred - center_pos;
                auto dist = radius - disp.norm() + verts("pscale",vi);

                if(dist < 0)
                    return;

                auto nrm = disp.normalized();

                auto dp = dist * nrm;
                if(dp.norm() < (T)1e-6)
                    return;

                pred += dp;

                // if(do_stablize) {
                //     pos += dp;
                //     verts.tuple(dim_c<3>,xtag,vi) = pos; 
                // }

                auto collider_velocity_at_p = center_vel + angular_velocity.cross(pred - center_pos);
                auto rel_vel = pred - pos - collider_velocity_at_p;

                auto tan_vel = rel_vel - nrm * rel_vel.dot(nrm);
                auto tan_len = tan_vel.norm();
                auto max_tan_len = dp.norm() * friction;

                if(tan_len > (T)1e-6) {
                    auto alpha = (T)max_tan_len / (T)tan_len;
                    dp = -tan_vel * zs::min(alpha,(T)1.0);
                    pred += dp;
                }

                // dp = dp * verts("m",vi) * verts("minv",vi);

                verts.tuple(dim_c<3>,ptag,vi) = pred;    
        });
        // }
        set_output("zsparticles",zsparticles);
    }

};

ZENDEFNODE(SDFColliderProject, {{{"zsparticles"},
                                {"sdf_boundary"},
                                // {"float","radius","1"},
                                // {"center"},
                                // {"center_velocity"},
                                // {"angular_velocity"},
                                {"string","xtag","x"},
                                {"string","ptag","x"},
                                {"float","friction","0"}
                                // {"bool","do_stablize","0"}
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});

struct SDFColliderProject2 : INode {
    template <typename LsView, typename TileVecT>
    constexpr void projectBoundary(zs::CudaExecutionPolicy &cudaPol, LsView lsv, const ZenoBoundary &boundary,
                                    const zs::SmallString& xtag,const zs::SmallString& vtag,
                                   TileVecT &verts,const T& friction) {
        using namespace zs;
        using T = typename TileVecT::value_type;
        auto collider = boundary.getBoundary(lsv);
        cudaPol(Collapse{verts.size()},
                [verts = proxy<execspace_e::cuda>({}, verts), boundary = collider, xtag,vtag,friction] __device__(int vi) mutable {
                    using mat3 = zs::vec<double, 3, 3>;
                    if(verts("minv",vi) < 1e-6)
                        return;
                    auto vel = verts.template pack<3>(vtag, vi);
                    auto pred = verts.template pack<3>(xtag, vi);
                    if (boundary.queryInside(pred)) {
                        auto bou_normal = boundary.getNormal(pred);
                        boundary.resolveCollisionWithNormal(pred,vel,bou_normal);
                        verts.tuple(dim_c<3>,xtag,vi) = pred;
                    }
                });
    }

    void apply() override {
        using namespace zs;
        using dtiles_t = zs::TileVector<float,32>;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec().device(0);

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto xtag = get_input2<std::string>("xtag");
        auto ptag = get_input2<std::string>("pxtag");

        auto friction = get_input2<float>("friction");

        auto& verts = zsparticles->getParticles();

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },verts.size()};

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            xtag = zs::SmallString(xtag),
            ptag = zs::SmallString(ptag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,ptag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,ptag,vi) - verts.pack(dim_c<3>,xtag,vi);
                vtemp("minv",vi) = verts("minv",vi);
        });

        if(has_input<ZenoBoundary>("zsboundary")) {
            using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
            using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
            using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;

            auto boundary = get_input<ZenoBoundary>("zsboundary");
            if(boundary->zsls) {
                match(
                    [&](const auto& ls) {
                        if constexpr (is_same_v<RM_CVREF_T(ls),basic_ls_t>) {
                            match([&](const auto& lsPtr) {
                                auto lsv = get_level_set_view<space>(lsPtr);
                                projectBoundary(cudaPol,lsv,*boundary,"x","v",vtemp,friction);
                            })(ls._ls);
                        }
                })(boundary->zsls->getLevelSet());   
            }
        }
        TILEVEC_OPS::copy(cudaPol,vtemp,"x",verts,ptag);

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(SDFColliderProject2, {{{"zsparticles"},
                                {"zsboundary"},
                                {"string","xtag","x"},
                                {"string","ptag","x"},
                                {"float","friction","0"}
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});

struct DetangleImminentCollisionWithBoundary : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using dtiles_t = zs::TileVector<T,32>;

    using bvh_t = ZenoLinearBvh::lbvh_t;
    using bv_t = bvh_t::Box;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        // constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;
        constexpr auto MAX_COLLISION_PAIRS = 100000;
        constexpr auto MAX_IMMINENT_COLLISION_PAIRS = 20000;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");      
        auto repel_strength = get_input2<float>("repeling_strength");
        auto imminent_collision_thickness = get_input2<float>("immc_thickness");
        // apply impulse for imminent collision for previous configuration
        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();
        const auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];


        if(!verts.hasProperty("imminent_fail"))
            verts.append_channels(cudaPol,{{"imminent_fail",1}});
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {verts("imminent_fail",vi) = (T)0;});

        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;

        auto kboundary = get_input2<ZenoParticles>("boundary");
        // auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        // auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto &kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = kboundary->getQuadraturePoints();

        zs::bht<int,2,int> csPT{verts.get_allocator(),MAX_COLLISION_PAIRS};csPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEE{edges.get_allocator(),MAX_COLLISION_PAIRS};csEE.reset(cudaPol,true);

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1},
            {"collision_cancel",1}
        },verts.size() + kverts.size()};
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            pre_x_tag = zs::SmallString(pre_x_tag),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp("minv",vi) = verts("minv",vi);
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3) 
                    vtemp("collision_cancel",vi) = 1;
                else
                    vtemp("collision_cancel",vi) = 0;
        });

        cudaPol(zs::range(kverts.size()),[
            voffset = verts.size(),
            vtemp = proxy<space>({},vtemp),
            kverts = proxy<space>({},kverts),
            nm_substeps = nm_substeps,
            pw = pw] ZS_LAMBDA(int kvi) mutable {
                vtemp.tuple(dim_c<3>,"x",kvi + voffset) = kverts.pack(dim_c<3>,"px",kvi) * (1-pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                vtemp.tuple(dim_c<3>,"v",kvi + voffset) = (kverts.pack(dim_c<3>,"x",kvi) - kverts.pack(dim_c<3>,"px",kvi)) / (T)nm_substeps;
                vtemp("minv",kvi + voffset) = (T)0;
                if(kverts.hasProperty("collision_cancel") && kverts("collision_cancel",kvi) > 1e-3) 
                    vtemp("collision_cancel",kvi + voffset) = 1;
                 else
                    vtemp("collision_cancel",kvi + voffset) = 0;
        });


        dtiles_t etemp{edges.get_allocator(),{
            {"inds",2}
        },edges.size() + kedges.size()};
        TILEVEC_OPS::copy<2>(cudaPol,edges,"inds",etemp,"inds",0);
        cudaPol(zs::range(kedges.size()),[
            eoffset = edges.size(),
            etemp = proxy<space>({},etemp),
            kedges = proxy<space>({},kedges),
            voffset = verts.size()] ZS_LAMBDA(int kei) mutable {
                auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);
                kedge += voffset;
                etemp.tuple(dim_c<2>,"inds",kei + eoffset) = kedge.reinterpret_bits(float_c);
        });

        dtiles_t ttemp{tris.get_allocator(),{
            {"inds",3}
        },tris.size() + ktris.size()};
        TILEVEC_OPS::copy<3>(cudaPol,tris,"inds",ttemp,"inds",0);
        cudaPol(zs::range(ktris.size()),[
            toffset = tris.size(),
            ttemp = proxy<space>({},ttemp),
            ktris = proxy<space>({},ktris),
            voffset = verts.size()] ZS_LAMBDA(int kti) mutable {
                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                ktri += voffset;
                ttemp.tuple(dim_c<3>,"inds",kti + toffset) = ktri.reinterpret_bits(float_c);
        });

        dtiles_t imminent_collision_buffer(verts.get_allocator(),
            {
                {"inds",4},
                {"bary",4},
                {"impulse",3},
                {"collision_normal",3}
            },(size_t)MAX_IMMINENT_COLLISION_PAIRS);


        auto nm_iters = get_input2<int>("nm_imminent_iters");
        auto imminent_restitution_rate = get_input2<float>("imm_restitution");
        auto imminent_relaxation_rate = get_input2<float>("imm_relaxation");


        auto do_pt_detection = get_input2<bool>("use_PT");
        auto do_ee_detection = get_input2<bool>("use_EE");

        zs::Vector<int> nm_imminent_collision{verts.get_allocator(),(size_t)1};

        // std::cout << "do imminent detangle" << std::endl;

        float vn_threshold = 5e-3;
        auto add_repulsion_force = get_input2<bool>("add_repulsion_force");
        

        auto triBvh = bvh_t{};
        auto edgeBvh = bvh_t{};


        for(int it = 0;it != nm_iters;++it) {
            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {verts("imminent_fail",vi) = (T)0;});

            if(do_pt_detection) {
                auto triBvs = retrieve_bounding_volumes(cudaPol,vtemp,ttemp,wrapv<3>{},imminent_collision_thickness / (T)2,"x");
                if(it == 0) {
                    triBvh.build(cudaPol,triBvs);
                }else {
                    triBvh.refit(cudaPol,triBvs);
                }

                COLLISION_UTILS::calc_imminent_self_PT_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    ttemp,
                    halfedges,
                    imminent_collision_thickness,
                    0,
                    triBvh,
                    imminent_collision_buffer,csPT);
                std::cout << "nm_imminent_PT_collision : " << csPT.size() << std::endl;
            }

            if(do_ee_detection) {
                auto edgeBvs = retrieve_bounding_volumes(cudaPol,vtemp,etemp,wrapv<2>{},imminent_collision_thickness / (T)2,"x");
                if(it == 0) {
                    edgeBvh.build(cudaPol,edgeBvs);
                }else {
                    edgeBvh.refit(cudaPol,edgeBvs);
                }
                COLLISION_UTILS::calc_imminent_self_EE_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    etemp,
                    imminent_collision_thickness,
                    csPT.size(),
                    edgeBvh,
                    imminent_collision_buffer,csEE);
                std::cout << "nm_imminent_EE_collision : " << csEE.size() << std::endl;
            }
            // resolve imminent PT collision
            
            // impulse_norm = TILEVEC_OPS::dot<3>(cudaPol,imminent_collision_buffer,"impulse","impulse");
            // std::cout << "EE_PT_impulse_norm : " << impulse_norm << std::endl;
            
            COLLISION_UTILS::apply_impulse(cudaPol,
                vtemp,"v",
                imminent_restitution_rate,
                imminent_relaxation_rate,
                vn_threshold,
                imminent_collision_buffer,
                nm_imminent_collision,csPT.size() + csEE.size());


            std::cout << "nm_kinematic_imminent_collision : " << nm_imminent_collision.getVal(0) << std::endl;
            if(nm_imminent_collision.getVal(0) == 0) 
                break;              

            if(add_repulsion_force) {
                // if(add_repulsion_force) {
                std::cout << "add imminent replering force" << std::endl;
                auto max_repel_distance = get_input2<T>("max_repel_distance");
    
                cudaPol(zs::range(csPT.size() + csEE.size()),[
                    imminent_collision_buffer = proxy<space>({},imminent_collision_buffer)] ZS_LAMBDA(int ci) mutable {
                        imminent_collision_buffer.tuple(dim_c<3>,"impulse",ci) = vec3::zeros();
                });
    
                cudaPol(zs::range(csPT.size() + csEE.size()),[
                    verts = proxy<space>({},verts),
                    vtemp = proxy<space>({},vtemp),
                    eps = eps,
                    exec_tag = wrapv<space>{},
                    k = repel_strength,
                    vn_threshold = vn_threshold,
                    max_repel_distance = max_repel_distance,
                    thickness = imminent_collision_thickness,
                    nm_imminent_collision = proxy<space>(nm_imminent_collision),
                    imminent_collision_buffer = proxy<space>({},imminent_collision_buffer)] ZS_LAMBDA(auto id) mutable {
                        auto inds = imminent_collision_buffer.pack(dim_c<4>,"inds",id,int_c);
                        auto bary = imminent_collision_buffer.pack(dim_c<4>,"bary",id);
    
                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        auto vr = vec3::zeros();
                        auto pr = vec3::zeros();
                        for(int i = 0;i != 4;++i) {
                            ps[i] = vtemp.pack(dim_c<3>,"x",inds[i]);
                            vs[i] = vtemp.pack(dim_c<3>,"v",inds[i]);
                            pr += bary[i] * ps[i];
                            vr += bary[i] * vs[i];
                        }
    
                        auto dist = pr.norm();
                        vec3 collision_normal = imminent_collision_buffer.pack(dim_c<3>,"collision_normal",id);
    
                        if(dist > thickness) 
                            return;
    
                        auto d = thickness - dist;
                        auto vn = vr.dot(collision_normal);
                        if(vn < -vn_threshold) {
                            atomic_add(exec_tag,&nm_imminent_collision[0],1);
                            for(int i = 0;i != 4;++i) {
                                if(vtemp("minv",inds[i]) < eps)
                                    continue;
    
                                verts("imminent_fail",inds[i]) = (T)1.0;
                            }
                        }
                        if(vn > (T)max_repel_distance * d || d < 0) {          
                            // if with current velocity, the collided particles can be repeled by more than 1% of collision depth, no extra repulsion is needed
                            return;
                        } else {
                            // make sure the collided particles is seperated by 1% of collision depth
                            // assume the particles has the same velocity
                            auto I = k * d;
                            auto I_max = (max_repel_distance * d - vn);
                            I = I_max < I ? I_max : I;
                            auto impulse = (T)I * collision_normal; 
    
                            imminent_collision_buffer.tuple(dim_c<3>,"impulse",id) = impulse;
                        }   
                    });
    
    
                auto impulse_norm = TILEVEC_OPS::dot<3>(cudaPol,imminent_collision_buffer,"impulse","impulse");
                // std::cout << "REPEL_impulse_norm : " << impulse_norm << std::endl;
    
                COLLISION_UTILS::apply_impulse(cudaPol,
                    vtemp,"v",
                    imminent_restitution_rate,
                    imminent_relaxation_rate,
                    imminent_collision_buffer,csPT.size() + csEE.size());
            }
        }


        std::cout << "finish imminent collision" << std::endl;

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>,current_x_tag,vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
        });
        
        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(DetangleImminentCollisionWithBoundary, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                {"float","repeling_strength","1.0"},
                                {"float","immc_thickness","0.01"},
                                {"boundary"},
                                // {"string","current_kx_tag","x"},
                                // {"string","previous_kx_tag","px"},
                                {"int","nm_imminent_iters","1"},
                                {"float","imm_restitution","0.1"},
                                {"float","imm_relaxation","0.25"},
                                {"float","max_repel_distance","0.1"},
                                {"bool","add_repulsion_force","0"},
                                {"bool","use_PT","1"},
                                {"bool","use_EE","1"},
                                {"int","nm_substeps","1"},
                                {"int","substep_id","0"}
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});



struct DetangleImminentCollisionWithBoundaryFast : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using dtiles_t = zs::TileVector<T,32>;

    using bvh_t = ZenoLinearBvh::lbvh_t;
    using bv_t = bvh_t::Box;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        // constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;
        constexpr auto MAX_COLLISION_PAIRS = 100000;
        constexpr auto MAX_IMMINENT_COLLISION_PAIRS = 20000;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");      
        auto repel_strength = get_input2<float>("repeling_strength");
        auto imminent_collision_thickness = get_input2<float>("immc_thickness");
        // apply impulse for imminent collision for previous configuration
        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();
        const auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];


        if(!verts.hasProperty("imminent_fail"))
            verts.append_channels(cudaPol,{{"imminent_fail",1}});
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {verts("imminent_fail",vi) = (T)0;});

        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;

        auto kboundary = get_input2<ZenoParticles>("boundary");
        // auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        // auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto &kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = kboundary->getQuadraturePoints();

        zs::bht<int,2,int> csPT{verts.get_allocator(),MAX_COLLISION_PAIRS};csPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEE{edges.get_allocator(),MAX_COLLISION_PAIRS};csEE.reset(cudaPol,true);

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1},
            {"collision_cancel",1}
        },verts.size() + kverts.size()};
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            pre_x_tag = zs::SmallString(pre_x_tag),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp("minv",vi) = verts("minv",vi);
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3) 
                    vtemp("collision_cancel",vi) = 1;
                else
                    vtemp("collision_cancel",vi) = 0;
        });

        cudaPol(zs::range(kverts.size()),[
            voffset = verts.size(),
            vtemp = proxy<space>({},vtemp),
            kverts = proxy<space>({},kverts),
            nm_substeps = nm_substeps,
            pw = pw] ZS_LAMBDA(int kvi) mutable {
                vtemp.tuple(dim_c<3>,"x",kvi + voffset) = kverts.pack(dim_c<3>,"px",kvi) * (1-pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                vtemp.tuple(dim_c<3>,"v",kvi + voffset) = (kverts.pack(dim_c<3>,"x",kvi) - kverts.pack(dim_c<3>,"px",kvi)) / (T)nm_substeps;
                vtemp("minv",kvi + voffset) = (T)0;
                if(kverts.hasProperty("collision_cancel") && kverts("collision_cancel",kvi) > 1e-3) 
                    vtemp("collision_cancel",kvi + voffset) = 1;
                    else
                    vtemp("collision_cancel",kvi + voffset) = 0;
        });


        dtiles_t etemp{edges.get_allocator(),{
            {"inds",2}
        },edges.size() + kedges.size()};
        TILEVEC_OPS::copy<2>(cudaPol,edges,"inds",etemp,"inds",0);
        cudaPol(zs::range(kedges.size()),[
            eoffset = edges.size(),
            etemp = proxy<space>({},etemp),
            kedges = proxy<space>({},kedges),
            voffset = verts.size()] ZS_LAMBDA(int kei) mutable {
                auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);
                kedge += voffset;
                etemp.tuple(dim_c<2>,"inds",kei + eoffset) = kedge.reinterpret_bits(float_c);
        });

        dtiles_t ttemp{tris.get_allocator(),{
            {"inds",3}
        },tris.size() + ktris.size()};
        TILEVEC_OPS::copy<3>(cudaPol,tris,"inds",ttemp,"inds",0);
        cudaPol(zs::range(ktris.size()),[
            toffset = tris.size(),
            ttemp = proxy<space>({},ttemp),
            ktris = proxy<space>({},ktris),
            voffset = verts.size()] ZS_LAMBDA(int kti) mutable {
                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                ktri += voffset;
                ttemp.tuple(dim_c<3>,"inds",kti + toffset) = ktri.reinterpret_bits(float_c);
        });

        dtiles_t imminent_collision_buffer(verts.get_allocator(),
            {
                {"inds",4},
                {"bary",4},
                {"impulse",3},
                {"collision_normal",3}
            },(size_t)MAX_IMMINENT_COLLISION_PAIRS);


        auto nm_iters = get_input2<int>("nm_imminent_iters");
        auto imminent_restitution_rate = get_input2<float>("imm_restitution");
        auto imminent_relaxation_rate = get_input2<float>("imm_relaxation");


        auto do_pt_detection = get_input2<bool>("use_PT");
        auto do_ee_detection = get_input2<bool>("use_EE");

        zs::Vector<int> nm_imminent_collision{verts.get_allocator(),(size_t)1};

        // std::cout << "do imminent detangle" << std::endl;

        float vn_threshold = 5e-3;
        auto add_repulsion_force = get_input2<bool>("add_repulsion_force");
        

        auto triBvh = bvh_t{};
        auto edgeBvh = bvh_t{};


        for(int it = 0;it != nm_iters;++it) {
            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {verts("imminent_fail",vi) = (T)0;});

            if(do_pt_detection) {
                auto triBvs = retrieve_bounding_volumes(cudaPol,vtemp,ttemp,wrapv<3>{},imminent_collision_thickness / (T)2,"x");
                if(it == 0) {
                    triBvh.build(cudaPol,triBvs);
                }else {
                    triBvh.refit(cudaPol,triBvs);
                }

                COLLISION_UTILS::calc_imminent_self_PT_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    ttemp,
                    halfedges,
                    imminent_collision_thickness,
                    0,
                    triBvh,
                    imminent_collision_buffer,csPT);
                std::cout << "nm_imminent_PT_collision : " << csPT.size() << std::endl;
            }

            if(do_ee_detection) {
                auto edgeBvs = retrieve_bounding_volumes(cudaPol,vtemp,etemp,wrapv<2>{},imminent_collision_thickness / (T)2,"x");
                if(it == 0) {
                    edgeBvh.build(cudaPol,edgeBvs);
                }else {
                    edgeBvh.refit(cudaPol,edgeBvs);
                }
                COLLISION_UTILS::calc_imminent_self_EE_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    etemp,
                    imminent_collision_thickness,
                    csPT.size(),
                    edgeBvh,
                    imminent_collision_buffer,csEE);
                std::cout << "nm_imminent_EE_collision : " << csEE.size() << std::endl;
            }
            // resolve imminent PT collision
            
            // impulse_norm = TILEVEC_OPS::dot<3>(cudaPol,imminent_collision_buffer,"impulse","impulse");
            // std::cout << "EE_PT_impulse_norm : " << impulse_norm << std::endl;
            
            COLLISION_UTILS::apply_impulse(cudaPol,
                vtemp,"v",
                imminent_restitution_rate,
                imminent_relaxation_rate,
                vn_threshold,
                imminent_collision_buffer,
                nm_imminent_collision,csPT.size() + csEE.size());


            std::cout << "nm_kinematic_imminent_collision : " << nm_imminent_collision.getVal(0) << std::endl;
            if(nm_imminent_collision.getVal(0) == 0) 
                break;              

            if(add_repulsion_force) {
                // if(add_repulsion_force) {
                std::cout << "add imminent replering force" << std::endl;
                auto max_repel_distance = get_input2<T>("max_repel_distance");
    
                cudaPol(zs::range(csPT.size() + csEE.size()),[
                    imminent_collision_buffer = proxy<space>({},imminent_collision_buffer)] ZS_LAMBDA(int ci) mutable {
                        imminent_collision_buffer.tuple(dim_c<3>,"impulse",ci) = vec3::zeros();
                });
    
                cudaPol(zs::range(csPT.size() + csEE.size()),[
                    verts = proxy<space>({},verts),
                    vtemp = proxy<space>({},vtemp),
                    eps = eps,
                    exec_tag = wrapv<space>{},
                    k = repel_strength,
                    vn_threshold = vn_threshold,
                    max_repel_distance = max_repel_distance,
                    thickness = imminent_collision_thickness,
                    nm_imminent_collision = proxy<space>(nm_imminent_collision),
                    imminent_collision_buffer = proxy<space>({},imminent_collision_buffer)] ZS_LAMBDA(auto id) mutable {
                        auto inds = imminent_collision_buffer.pack(dim_c<4>,"inds",id,int_c);
                        auto bary = imminent_collision_buffer.pack(dim_c<4>,"bary",id);
    
                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        auto vr = vec3::zeros();
                        auto pr = vec3::zeros();
                        for(int i = 0;i != 4;++i) {
                            ps[i] = vtemp.pack(dim_c<3>,"x",inds[i]);
                            vs[i] = vtemp.pack(dim_c<3>,"v",inds[i]);
                            pr += bary[i] * ps[i];
                            vr += bary[i] * vs[i];
                        }
    
                        auto dist = pr.norm();
                        vec3 collision_normal = imminent_collision_buffer.pack(dim_c<3>,"collision_normal",id);
    
                        if(dist > thickness) 
                            return;
    
                        auto d = thickness - dist;
                        auto vn = vr.dot(collision_normal);
                        if(vn < -vn_threshold) {
                            atomic_add(exec_tag,&nm_imminent_collision[0],1);
                            for(int i = 0;i != 4;++i) {
                                if(vtemp("minv",inds[i]) < eps)
                                    continue;
    
                                verts("imminent_fail",inds[i]) = (T)1.0;
                            }
                        }
                        if(vn > (T)max_repel_distance * d || d < 0) {          
                            // if with current velocity, the collided particles can be repeled by more than 1% of collision depth, no extra repulsion is needed
                            return;
                        } else {
                            // make sure the collided particles is seperated by 1% of collision depth
                            // assume the particles has the same velocity
                            auto I = k * d;
                            auto I_max = (max_repel_distance * d - vn);
                            I = I_max < I ? I_max : I;
                            auto impulse = (T)I * collision_normal; 
    
                            imminent_collision_buffer.tuple(dim_c<3>,"impulse",id) = impulse;
                        }   
                    });
    
    
                auto impulse_norm = TILEVEC_OPS::dot<3>(cudaPol,imminent_collision_buffer,"impulse","impulse");
                // std::cout << "REPEL_impulse_norm : " << impulse_norm << std::endl;
    
                COLLISION_UTILS::apply_impulse(cudaPol,
                    vtemp,"v",
                    imminent_restitution_rate,
                    imminent_relaxation_rate,
                    imminent_collision_buffer,csPT.size() + csEE.size());
            }
        }


        std::cout << "finish imminent collision" << std::endl;

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>,current_x_tag,vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
        });
        
        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(DetangleImminentCollisionWithBoundaryFast, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                {"float","repeling_strength","1.0"},
                                {"float","immc_thickness","0.01"},
                                {"boundary"},
                                // {"string","current_kx_tag","x"},
                                // {"string","previous_kx_tag","px"},
                                {"int","nm_imminent_iters","1"},
                                {"float","imm_restitution","0.1"},
                                {"float","imm_relaxation","0.25"},
                                {"float","max_repel_distance","0.1"},
                                {"bool","add_repulsion_force","0"},
                                {"bool","use_PT","1"},
                                {"bool","use_EE","1"},
                                {"int","nm_substeps","1"},
                                {"int","substep_id","0"}
                            },
                            {{"zsparticles"}},
                            {},
                            {"PBD"}});


struct DetangleCCDCollisionWithBoundary : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using vec4i = zs::vec<int,4>;
    using dtiles_t = zs::TileVector<T,32>;

    using bvh_t = ZenoLinearBvh::lbvh_t;
    using bv_t = bvh_t::Box;

    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using lbvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;
        constexpr auto MAX_COLLISION_PAIRS = 200000;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");  
        auto nm_ccd_iters = get_input2<int>("nm_ccd_iters"); 

        auto thickness = get_input2<float>("thickness");    
        auto restitution_rate = get_input2<float>("restitution");
        auto relaxation_rate = get_input2<float>("relaxation");

        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();    
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 

        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;

        auto kboundary = get_input2<ZenoParticles>("boundary");

        
        auto boundary_velocity_scale = get_input2<float>("boundary_velocity_scale");
        // auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        // auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto &kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = kboundary->getQuadraturePoints();

        zs::bht<int,2,int> csPT{verts.get_allocator(),MAX_COLLISION_PAIRS};csPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEE{edges.get_allocator(),MAX_COLLISION_PAIRS};csEE.reset(cudaPol,true);

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1},
            {"m",1},
            {"collision_cancel",1}
        },verts.size() + kverts.size()};
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            pre_x_tag = zs::SmallString(pre_x_tag),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp("minv",vi) = verts("minv",vi);
                vtemp("m",vi) = verts("m",vi);
                if(verts.hasProperty("collision_cancel") && verts("collision_cancel",vi) > 1e-3)
                    vtemp("collision_cancel",vi) = 1;
                else
                    vtemp("collision_cancel",vi) = 0;
        });

        cudaPol(zs::range(kverts.size()),[
            voffset = verts.size(),
            vtemp = proxy<space>({},vtemp),
            kverts = proxy<space>({},kverts),
            pw = pw,
            boundary_velocity_scale = boundary_velocity_scale,
            w = w] ZS_LAMBDA(int kvi) mutable {
                auto cur_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 -  w) + kverts.pack(dim_c<3>,"x",kvi) *  w;
                auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                vtemp.tuple(dim_c<3>,"x",kvi + voffset) = pre_kvert;
                vtemp.tuple(dim_c<3>,"v",kvi + voffset) = (cur_kvert - pre_kvert) * boundary_velocity_scale;
                vtemp("minv",kvi + voffset) = (T)0;
                vtemp("m",kvi + voffset) = (T)1000;
                if(kverts.hasProperty("collision_cancel") && kverts("collision_cancel",kvi) > 1e-3)
                    vtemp("collision_cancel",kvi + voffset) = 1;
                else
                    vtemp("collision_cancel",kvi + voffset) = 0;
        });

        dtiles_t etemp{edges.get_allocator(),{
            {"inds",2}
        },edges.size() + kedges.size()};
        TILEVEC_OPS::copy<2>(cudaPol,edges,"inds",etemp,"inds",0);
        cudaPol(zs::range(kedges.size()),[
            eoffset = edges.size(),
            etemp = proxy<space>({},etemp),
            kedges = proxy<space>({},kedges),
            voffset = verts.size()] ZS_LAMBDA(int kei) mutable {
                auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);
                kedge += voffset;
                etemp.tuple(dim_c<2>,"inds",kei + eoffset) = kedge.reinterpret_bits(float_c);
        });

        dtiles_t ttemp{tris.get_allocator(),{
            {"inds",3}
        },tris.size() + ktris.size()};
        TILEVEC_OPS::copy<3>(cudaPol,tris,"inds",ttemp,"inds",0);
        cudaPol(zs::range(ktris.size()),[
            toffset = tris.size(),
            ttemp = proxy<space>({},ttemp),
            ktris = proxy<space>({},ktris),
            voffset = verts.size()] ZS_LAMBDA(int kti) mutable {
                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                ktri += voffset;
                ttemp.tuple(dim_c<3>,"inds",kti + toffset) = ktri.reinterpret_bits(float_c);
        });

        lbvh_t triBvh{},eBvh{};

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

        auto do_ee_detection = get_input2<bool>("do_ee_detection");
        auto do_pt_detection = get_input2<bool>("do_pt_detection");
    
        zs::Vector<int> nm_ccd_collision{verts.get_allocator(),1};

        auto res_threshold = thickness * 0.01;
        res_threshold = res_threshold < 5e-3 ? 5e-3 : res_threshold;

        zs::Vector<int> ccd_fail_mark{verts.get_allocator(),verts.size()};

        for(int iter = 0;iter != nm_ccd_iters;++iter) {

            cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
            cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

            nm_ccd_collision.setVal(0);

            if(do_pt_detection) {
                std::cout << "do continous self PT cololision impulse" << std::endl;

                auto do_bvh_refit = iter > 0;
                COLLISION_UTILS::calc_continous_self_PT_collision_impulse_with_toc(cudaPol,
                    vtemp,
                    vtemp,
                    vtemp,"x","v",
                    ttemp,
                    // thickness,
                    triBvh,
                    do_bvh_refit,
                    csPT,
                    impulse_buffer,
                    impulse_count,false);
            }

            if(do_ee_detection) {
                std::cout << "do continous self EE cololision impulse" << std::endl;
                auto do_bvh_refit = iter > 0;
                COLLISION_UTILS::calc_continous_self_EE_collision_impulse_with_toc(cudaPol,
                    vtemp,
                    vtemp,
                    vtemp,"x","v",
                    etemp,
                    0,
                    edges.size(),
                    eBvh,
                    do_bvh_refit,
                    csEE,
                    impulse_buffer,
                    impulse_count,false);
            }

            std::cout << "apply CCD impulse" << std::endl;
            cudaPol(zs::range(verts.size()),[
                ccd_fail_mark = proxy<space>(ccd_fail_mark),
                verts = proxy<space>({},verts),
                vtemp = proxy<space>({},vtemp),
                impulse_buffer = proxy<space>(impulse_buffer),
                impulse_count = proxy<space>(impulse_count),
                relaxation_rate = relaxation_rate,
                nm_ccd_collision = proxy<space>(nm_ccd_collision),
                res_threshold = res_threshold,
                eps = eps,
                thickness = thickness,
                exec_tag = exec_tag] ZS_LAMBDA(int vi) mutable {
                ccd_fail_mark[vi] = 0;
                if(impulse_count[vi] == 0)
                    return;
                if(impulse_buffer[vi].norm() < eps)
                    return;

                auto impulse = relaxation_rate * impulse_buffer[vi] / impulse_count[vi];
                // auto impulse = relaxation_rate * impulse_buffer[vi] * 0.25;
                if(impulse.norm() > res_threshold) {
                    ccd_fail_mark[vi] = 1;
                    atomic_add(exec_tag,&nm_ccd_collision[0],1);
                }

                // auto dv = impulse
                vtemp.tuple(dim_c<3>,"v",vi) = vtemp.pack(dim_c<3>,"v",vi) + impulse;
                // for(int i = 0;i != 3;++i)   
                //     atomic_add(exec_tag,&vtemp("v",i,vi),dv[i]);
            });

            std::cout << "nm_kinematic_ccd_collision : " << nm_ccd_collision.getVal() << std::endl;
            if(nm_ccd_collision.getVal() == 0)
                break;
        }   

        std::cout << "finish solving continous collision " << std::endl;
        if(!verts.hasProperty("ccd_fail_mark"))
            verts.append_channels(cudaPol,{{"ccd_fail_mark",1}});
        cudaPol(zs::range(verts.size()),[
            vtemp = proxy<space>({},vtemp),
            verts = proxy<space>({},verts),
            ccd_fail_mark = proxy<space>(ccd_fail_mark),
            xtag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                // if(ccd_fail_mark[vi] == 0)
            verts.tuple(dim_c<3>,xtag,vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
            verts("ccd_fail_mark",vi) = ccd_fail_mark[vi];
        });  

        set_output("zsparticles",zsparticles);   
    }
};

ZENDEFNODE(DetangleCCDCollisionWithBoundary, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                {"int","nm_ccd_iters","1"},
                                {"float","thickness","0.1"},
                                {"float","restitution","0.1"},
                                {"float","relaxation","1"},
                                {"boundary"},
                                {"bool","do_ee_detection","1"},
                                {"bool","do_pt_detection","1"},
                                {"int","substep_id","0"},
                                {"int","nm_substeps","1"},
                                {"float","boundary_velocity_scale","1"},
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});


struct VisualizeCCDCollisionWithBoundary : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using vec4i = zs::vec<int,4>;
    using dtiles_t = zs::TileVector<T,32>;

    using bvh_t = ZenoLinearBvh::lbvh_t;
    using bv_t = bvh_t::Box;   
    
    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;
        
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using lbvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;
        constexpr auto MAX_COLLISION_PAIRS = 200000;
        
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto xtag = get_input2<std::string>("xtag");
        auto vtag = get_input2<std::string>("vtag");
        // auto pre_x_tag = get_input2<std::string>("previous_x_tag");  

        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();    
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 

        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;
        // auto use_barycentric_interpolator = get_input2<bool>("use_barycentric_interpolator");
        auto kboundary = get_input2<ZenoParticles>("boundary");
        // auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        // auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = kboundary->getQuadraturePoints();

        zs::bht<int,2,int> csPT{verts.get_allocator(),MAX_COLLISION_PAIRS};csPT.reset(cudaPol,true);

        dtiles_t kvtemp{kverts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"inds",1}
        },kverts.size()};

        cudaPol(zs::range(kverts.size()),[
            kvtemp = proxy<space>({},kvtemp),
            kverts = proxy<space>({},kverts),
            pw = pw,
            w = w] ZS_LAMBDA(int kvi) mutable {
                auto cur_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 -  w) + kverts.pack(dim_c<3>,"x",kvi) *  w;
                auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                kvtemp.tuple(dim_c<3>,"x",kvi) = pre_kvert;
                kvtemp.tuple(dim_c<3>,"v",kvi) = cur_kvert - pre_kvert;
                kvtemp("inds",kvi) = zs::reinterpret_bits<float>(kvi);
        });


        lbvh_t ktriBvh{};
        auto ktriBvh_bvs = retrieve_bounding_volumes(cudaPol,kvtemp,ktris,kvtemp,wrapv<3>{},(T)1.0,(T)0,"x","v");        
        ktriBvh.build(cudaPol,ktriBvh_bvs);

        zs::Vector<T> tocs{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(tocs),[] ZS_LAMBDA(auto& toc) mutable {toc = std::numeric_limits<T>::max();});
        COLLISION_UTILS::detect_continous_PKT_collision_pairs(cudaPol,
            verts,xtag,vtag,
            kvtemp,"x","v",
            ktris,
            ktriBvh,
            tocs,
            csPT);

        std::cout << "nm_PKT_intersections : " << csPT.size() << std::endl;

        dtiles_t PKT_pairs_vis{csPT.get_allocator(),{
                {"xts",9},
                {"xtc",9},
                {"xte",9},
                {"xp",3}
            },
        csPT.size()};
        
        cudaPol(zip(zs::range(csPT.size()),csPT._activeKeys),[
            PKT_pairs_vis = proxy<space>({},PKT_pairs_vis),
            tocs = proxy<space>(tocs),
            verts = proxy<space>({},verts),xtag = zs::SmallString(xtag),
            kvtemp = proxy<space>({},kvtemp),
            ktris = proxy<space>({},ktris)] ZS_LAMBDA(auto ci,const auto& pair) mutable {
                auto vi = pair[0];
                auto kti = pair[1];
                auto p = verts.pack(dim_c<3>,xtag,vi);
                auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);

                vec3 ktps[3] = {};
                vec3 ktvs[3] = {};
                for(int i = 0;i != 3;++i) {
                    ktps[i] = kvtemp.pack(dim_c<3>,"x",ktri[i]);
                    ktvs[i] = kvtemp.pack(dim_c<3>,"v",ktri[i]);
                }

                auto toc = tocs[vi];

                vec3 tkps_collider[3] = {};
                for(int i = 0;i != 3;++i)
                    tkps_collider[i] = ktps[i] + toc * ktvs[i];

                vec3 tkps_end[3] = {};
                for(int i = 0;i != 3;++i)
                    tkps_end[i] = ktps[i] + ktvs[i];

                
                PKT_pairs_vis.tuple(dim_c<9>,"xts",ci) = zs::vec<T,9>{
                    ktps[0][0],ktps[0][1],ktps[0][2],
                    ktps[1][0],ktps[1][1],ktps[1][2],
                    ktps[2][0],ktps[2][1],ktps[2][2]};

                PKT_pairs_vis.tuple(dim_c<9>,"xtc",ci) = zs::vec<T,9>{
                    tkps_collider[0][0],tkps_collider[0][1],tkps_collider[0][2],
                    tkps_collider[1][0],tkps_collider[1][1],tkps_collider[1][2],
                    tkps_collider[2][0],tkps_collider[2][1],tkps_collider[2][2]};

                PKT_pairs_vis.tuple(dim_c<9>,"xte",ci) = zs::vec<T,9>{
                    tkps_end[0][0],tkps_end[0][1],tkps_end[0][2],
                    tkps_end[1][0],tkps_end[1][1],tkps_end[1][2],
                    tkps_end[2][0],tkps_end[2][1],tkps_end[2][2]};

                PKT_pairs_vis.tuple(dim_c<3>,"xp",ci) = p;
        });

        auto ompPol = omp_exec();
        constexpr auto omp_space = execspace_e::openmp;
        PKT_pairs_vis = PKT_pairs_vis.clone({memsrc_e::host});
        
        auto collided_tri_start = std::make_shared<zeno::PrimitiveObject>();
        auto& cts_verts = collided_tri_start->verts;
        auto& cts_tris = collided_tri_start->tris;
        cts_verts.resize(csPT.size() * 3);
        cts_tris.resize(csPT.size());

        auto collided_tri_middle = std::make_shared<zeno::PrimitiveObject>();
        auto& ctm_verts = collided_tri_middle->verts;
        auto& ctm_tris = collided_tri_middle->tris;
        ctm_verts.resize(csPT.size() * 3);
        ctm_tris.resize(csPT.size());

        auto collided_tri_end = std::make_shared<zeno::PrimitiveObject>();
        auto& cte_verts = collided_tri_end->verts;
        auto& cte_tris = collided_tri_end->tris;
        cte_verts.resize(csPT.size() * 3);
        cte_tris.resize(csPT.size());

        ompPol(zs::range(csPT.size()),[
            &cts_verts,&cts_tris,
            &ctm_verts,&ctm_tris,
            &cte_verts,&cte_tris,
            PKT_pairs_vis = proxy<omp_space>({},PKT_pairs_vis)] (auto ci) mutable {
                auto tps_all = PKT_pairs_vis.pack(dim_c<9>,"xts",ci);
                vec3 tps[3] = {};
                for(int i = 0;i != 3;++i)
                    tps[i] = vec3{tps_all[i * 3 + 0],tps_all[i * 3 + 1],tps_all[i * 3 + 2]};
                for(int i = 0;i != 3;++i)
                    cts_verts[ci * 3 + i] = tps[i].to_array();
                cts_tris[ci] = zeno::vec3i{ci * 3 + 0,ci * 3 + 1,ci * 3 + 2};

                auto tpm_all = PKT_pairs_vis.pack(dim_c<9>,"xtc",ci);
                for(int i = 0;i != 3;++i)
                    tps[i] = vec3{tpm_all[i * 3 + 0],tpm_all[i * 3 + 1],tpm_all[i * 3 + 2]};   
                for(int i = 0;i != 3;++i)
                    ctm_verts[ci * 3 + i] = tps[i].to_array();
                ctm_tris[ci] = zeno::vec3i{ci * 3 + 0,ci * 3 + 1,ci * 3 + 2};       
                
                auto tpe_all = PKT_pairs_vis.pack(dim_c<9>,"xte",ci);
                for(int i = 0;i != 3;++i)
                    tps[i] = vec3{tpe_all[i * 3 + 0],tpe_all[i * 3 + 1],tpe_all[i * 3 + 2]};   
                for(int i = 0;i != 3;++i)
                    cte_verts[ci * 3 + i] = tps[i].to_array();
                cte_tris[ci] = zeno::vec3i{ci * 3 + 0,ci * 3 + 1,ci * 3 + 2};   
        });

        set_output("collided_tri_start",std::move(collided_tri_start));
        set_output("collided_tri_middle",std::move(collided_tri_middle));
        set_output("collided_tri_end",std::move(collided_tri_end));
    }
};


ZENDEFNODE(VisualizeCCDCollisionWithBoundary, {{{"zsparticles"},
                                {"string","xtag","x"},
                                {"string","vtag","v"},
                                {"int","nm_ccd_iters","1"},
                                {"boundary"},
                                {"bool","do_ee_detection","1"},
                                {"bool","do_pt_detection","1"},
                                {"int","substep_id","0"},
                                {"int","nm_substeps","1"},
                            },
							{
                                {"collided_tri_start"},
                                {"collided_tri_middle"},
                                {"collided_tri_end"}
                            },
							{},
							{"PBD"}});

struct DetangleCCDCollisionWithBoundary2 : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using vec4i = zs::vec<int,4>;
    using dtiles_t = zs::TileVector<T,32>;

    using bvh_t = ZenoLinearBvh::lbvh_t;
    using bv_t = bvh_t::Box;

    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using lbvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;
        constexpr auto MAX_COLLISION_PAIRS = 200000;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto xtag = get_input2<std::string>("xtag");
        auto vtag = get_input2<std::string>("vtag");
        // auto pre_x_tag = get_input2<std::string>("previous_x_tag");  
        auto nm_ccd_iters = get_input2<int>("nm_ccd_iters"); 

        auto thickness = get_input2<float>("thickness");    
        auto restitution_rate = get_input2<float>("restitution");
        auto relaxation_rate = get_input2<float>("relaxation");

        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();    
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 

        auto kxtag = get_input2<std::string>("kxtag");
        auto kpxtag = get_input2<std::string>("kpxtag");

        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;
        // auto use_barycentric_interpolator = get_input2<bool>("use_barycentric_interpolator");
        auto kboundary = get_input2<ZenoParticles>("boundary");
        // auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        // auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = kboundary->getQuadraturePoints();

        zs::bht<int,2,int> csPT{verts.get_allocator(),MAX_COLLISION_PAIRS};csPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEE{edges.get_allocator(),MAX_COLLISION_PAIRS};csEE.reset(cudaPol,true);

        dtiles_t kvtemp{kverts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"inds",1}
        },kverts.size()};
 
        cudaPol(zs::range(kverts.size()),[
            kvtemp = proxy<space>({},kvtemp),
            kverts = proxy<space>({},kverts),
            pw = pw,
            kxtag = zs::SmallString(kxtag),
            kpxtag = zs::SmallString(kpxtag),
            w = w] ZS_LAMBDA(int kvi) mutable {
                auto cur_kvert = kverts.pack(dim_c<3>,kpxtag,kvi) * (1 -  w) + kverts.pack(dim_c<3>,kxtag,kvi) *  w;
                auto pre_kvert = kverts.pack(dim_c<3>,kpxtag,kvi) * (1 - pw) + kverts.pack(dim_c<3>,kxtag,kvi) * pw;
                kvtemp.tuple(dim_c<3>,"x",kvi) = pre_kvert;
                kvtemp.tuple(dim_c<3>,"v",kvi) = cur_kvert - pre_kvert;
                kvtemp("inds",kvi) = zs::reinterpret_bits<float>(kvi);
        });

        lbvh_t ktriBvh{},keBvh{},kpBvh{};
        auto ktriBvh_bvs = retrieve_bounding_volumes(cudaPol,kvtemp,ktris,kvtemp,wrapv<3>{},(T)1.0,(T)0,"x","v");
        ktriBvh.build(cudaPol,ktriBvh_bvs);
        auto keBvh_bvs = retrieve_bounding_volumes(cudaPol,kvtemp,kedges,kvtemp,wrapv<2>{},(T)1.0,(T)0,"x","v");
        keBvh.build(cudaPol,keBvh_bvs);
        auto kpBvh_bvs = retrieve_bounding_volumes(cudaPol,kvtemp,kvtemp,kvtemp,wrapv<1>{},(T)1.0,(T)0,"x","v");
        kpBvh.build(cudaPol,kpBvh_bvs);

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

        auto do_ee_detection = get_input2<bool>("do_ee_detection");
        auto do_pt_detection = get_input2<bool>("do_pt_detection");

        // auto do_self_detection = get_input2<bool>("do_self_detection");
    
        // zs::Vector<int> nm_ccd_collision{verts.get_allocator(),1};

        auto res_threshold = thickness * 0.01;
        res_threshold = res_threshold < 5e-3 ? 5e-3 : res_threshold;

        // zs::Vector<int> ccd_fail_mark{verts.get_allocator(),verts.size()};

        auto update_x = get_input2<bool>("update_x");

        for(int iter = 0;iter != nm_ccd_iters;++iter) {
            cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
            cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

            // nm_ccd_collision.setVal(0);

            if(do_pt_detection) {
                std::cout << "do continous PKT cololision impulse" << std::endl;
                COLLISION_UTILS::calc_continous_PKT_collision_impulse(cudaPol,
                    verts,xtag,vtag,
                    kvtemp,"x","v",
                    ktris,
                    ktriBvh,
                    csPT,
                    impulse_buffer,
                    impulse_count);
                std::cout << "do continous KPT cololision impulse" << std::endl;
                COLLISION_UTILS::calc_continous_KPT_collision_impulse(cudaPol,
                    kvtemp,"x","v",
                    kpBvh,
                    verts,xtag,vtag,
                    tris,
                    csPT,
                    impulse_buffer,
                    impulse_count);
            }

            if(do_ee_detection) {
                std::cout << "do continous EKE cololision impulse" << std::endl;
                COLLISION_UTILS::calc_continous_EKE_collision_impulse_with_toc(cudaPol,
                    verts,xtag,vtag,
                    edges,
                    kvtemp,"x","v",
                    kedges,
                    keBvh,
                    csEE,
                    impulse_buffer,
                    impulse_count);
            }

            // if(do_self_detection) {

            // }

            std::cout << "apply CCD impulse" << std::endl;
            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                update_x = update_x,
                xtag = zs::SmallString(xtag),vtag = zs::SmallString(vtag),
                impulse_buffer = proxy<space>(impulse_buffer),
                impulse_count = proxy<space>(impulse_count),
                relaxation_rate = relaxation_rate] ZS_LAMBDA(int vi) mutable {
                    if(impulse_count[vi] == 0)
                        return;
                    auto impulse = relaxation_rate * impulse_buffer[vi] / impulse_count[vi];
                    if(update_x)
                        verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + impulse;
                    else
                        verts.tuple(dim_c<3>,vtag,vi) = verts.pack(dim_c<3>,vtag,vi) + impulse;
            });
        }   
        set_output("zsparticles",zsparticles);   
    }
};

ZENDEFNODE(DetangleCCDCollisionWithBoundary2, {{{"zsparticles"},
                                {"string","xtag","x"},
                                {"string","vtag","v"},
                                {"int","nm_ccd_iters","1"},
                                {"float","thickness","0.1"},
                                {"float","restitution","0.1"},
                                {"float","relaxation","1"},
                                {"boundary"},
                                {"string","kxtag","x"},
                                {"string","kpxtag","px"},
                                {"bool","do_ee_detection","1"},
                                {"bool","do_pt_detection","1"},
                                {"int","substep_id","0"},
                                {"int","nm_substeps","1"},
                                {"bool","update_x","1"}
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});

// struct CCDProjectToGeometry : INode {
//     using T = float;
//     using vec3 = zs::vec<T,3>;
//     using vec4 = zs::vec<T,4>;
//     using vec4i = zs::vec<int,4>;
//     using dtiles_t = zs::TileVector<T,32>;

//     using bvh_t = ZenoLinearBvh::lbvh_t;
//     using bv_t = bvh_t::Box;    

//     virtual void apply() override {
//         using namespace zs;
//         using namespace PBD_CONSTRAINT;

//         constexpr auto space = execspace_e::cuda;
//         auto cudaPol = cuda_exec();
//         using lbvh_t = typename ZenoLinearBvh::lbvh_t;
//         using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
//         constexpr auto exec_tag = wrapv<space>{};
//         constexpr auto eps = (T)1e-7;
//         constexpr auto MAX_COLLISION_PAIRS = 200000;
        
//         auto zsparticles = get_input<ZenoParticles>("zsparticles");
//         auto xtag = get_input2<std::string>("xtag");
//         auto dxtag = get_input2<std::string>("dxtag");  
        
//         auto nm_ccd_iters = get_input2<int>("nm_ccd_iters"); 

//         auto& verts = zsparticles->getParticles();
//         const auto& tris = zsparticles->getQuadraturePoints();    
//         const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        
        
//     }
// };

};