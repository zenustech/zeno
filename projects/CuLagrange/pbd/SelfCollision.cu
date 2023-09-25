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

// #include "constraint_function_kernel/constraint.cuh"
#include "../geometry/kernel/tiled_vector_ops.hpp"
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/geo_math.hpp"
#include "../fem/collision_energy/evaluate_collision.hpp"
// #include "constraint_function_kernel/constraint_types.hpp"

namespace zeno {

struct DetangleImminentCollision : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using dtiles_t = zs::TileVector<T,32>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        // constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;

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

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },verts.size()};
        TILEVEC_OPS::copy(cudaPol,verts,pre_x_tag,vtemp,"x");
        TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv");
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - vtemp.pack(dim_c<3>,"x",vi);
        });
        // TILEVEC_OPS::copy(cudaPol,verts,current_x_tag,vtemp,"v");
        // TILEVEC_OPS::add(cudaPol,vtemp,"v",1,"x",-1,"v");
        dtiles_t imminent_collision_buffer(verts.get_allocator(),
            {
                {"inds",4},
                {"bary",4},
                {"impulse",3},
                {"collision_normal",3}
            },(size_t)0);

        auto imminent_restitution_rate = get_input2<float>("imm_restitution");
        auto imminent_relaxation_rate = get_input2<float>("imm_relaxation");

        auto nm_iters = get_input2<int>("nm_imminent_iters");

        auto do_pt_detection = get_input2<bool>("use_PT");
        auto do_ee_detection = get_input2<bool>("use_EE");

        zs::Vector<int> nm_imminent_collision{verts.get_allocator(),(size_t)1};

        std::cout << "do imminent detangle" << std::endl;

        auto vn_threshold = 5e-3;

        for(int it = 0;it != nm_iters;++it) {

            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {verts("imminent_fail",vi) = (T)0;});

        // we use collision cell as the collision volume, PT collision is enough prevent penertation?
            if(do_pt_detection) {
                COLLISION_UTILS::calc_imminent_self_PT_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    tris,
                    halfedges,
                    imminent_collision_thickness,
                    0,
                    imminent_collision_buffer);
                // std::cout << "nm_imminent_PT_collision : " << imminent_collision_buffer.size() << std::endl;
            }

            if(do_ee_detection) {
                COLLISION_UTILS::calc_imminent_self_EE_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    edges,
                    imminent_collision_thickness,
                    imminent_collision_buffer.size(),
                    imminent_collision_buffer);
                // std::cout << "nm_imminent_EE_collision : " << imminent_collision_buffer.size() << std::endl;
            }
            // resolve imminent PT collision
            
            // impulse_norm = TILEVEC_OPS::dot<3>(cudaPol,imminent_collision_buffer,"impulse","impulse");
            // std::cout << "EE_PT_impulse_norm : " << impulse_norm << std::endl;

            COLLISION_UTILS::apply_impulse(cudaPol,
                vtemp,"v",
                imminent_restitution_rate,
                imminent_relaxation_rate,
                imminent_collision_buffer);

 
            auto add_repulsion_force = get_input2<bool>("add_repulsion_force");
            // add an impulse to repel the pair further

            nm_imminent_collision.setVal(0);
            // if(add_repulsion_force) {
                // std::cout << "add imminent replering force" << std::endl;
                auto max_repel_distance = get_input2<T>("max_repel_distance");

                cudaPol(zs::range(imminent_collision_buffer.size()),[
                    imminent_collision_buffer = proxy<space>({},imminent_collision_buffer)] ZS_LAMBDA(int ci) mutable {
                        imminent_collision_buffer.tuple(dim_c<3>,"impulse",ci) = vec3::zeros();
                });

                cudaPol(zs::range(imminent_collision_buffer.size()),[
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
                            for(int i = 0;i != 4;++i)
                                verts("imminent_fail",inds[i]) = (T)1.0;
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
                    std::cout << "nm_imminent_collision : " << nm_imminent_collision.getVal(0) << std::endl;
                    if(nm_imminent_collision.getVal(0) == 0) 
                        break;              


                // auto impulse_norm = TILEVEC_OPS::dot<3>(cudaPol,imminent_collision_buffer,"impulse","impulse");
                // std::cout << "REPEL_impulse_norm : " << impulse_norm << std::endl;

                COLLISION_UTILS::apply_impulse(cudaPol,
                    vtemp,"v",
                    imminent_restitution_rate,
                    imminent_relaxation_rate,
                    imminent_collision_buffer);
            // }
        }

        std::cout << "finish imminent collision" << std::endl;

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>,current_x_tag,vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
        });
        
        // std::cout << "finish apply imminent impulse" << std::endl;

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(DetangleImminentCollision, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                // {"string","pscaleTag","pscale"},
                                {"float","repeling_strength","1.0"},
                                {"float","immc_thickness","0.01"},
                                {"int","nm_imminent_iters","1"},
                                {"float","imm_restitution","0.1"},
                                {"float","imm_relaxation","0.25"},
                                {"float","max_repel_distance","0.1"},
                                {"bool","add_repulsion_force","0"},
                                {"bool","use_PT","1"},
                                {"bool","use_EE","1"},
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});


struct VisualizeImminentCollision : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using dtiles_t = zs::TileVector<T,32>;
    
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        auto ompPol = omp_exec();  
        constexpr auto omp_space = execspace_e::openmp;  

        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;
        constexpr auto MAX_PT_COLLISION_PAIRS = 1000000;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");
        auto imminent_collision_thickness = get_input2<float>("immc_thickness");

        // apply impulse for imminent collision for previous configuration
        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();
        const auto& edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
        const auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];

        // auto nm_imminent_iters = get_input2<int>("nm_imminent_iters");
        dtiles_t imminent_PT_collision_buffer(verts.get_allocator(),
            {
                {"inds",4},
                {"bary",4},
                {"impulse",3},
                {"collision_normal",3}
            },(size_t)0);

        dtiles_t imminent_EE_collision_buffer(verts.get_allocator(),
        {
            {"inds",4},
            {"bary",4},
            {"impulse",3},
            {"collision_normal",3}
        },(size_t)0);        
        // zs::bht<int,2,int> csPT{verts.get_allocator(),MAX_PT_COLLISION_PAIRS};
        // csPT.reset(cudaPol,true);

        dtiles_t vtemp(verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },verts.size());
        TILEVEC_OPS::copy(cudaPol,verts,pre_x_tag,vtemp,"x");
        TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv");
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - vtemp.pack(dim_c<3>,"x",vi);
        });
        // we use collision cell as the collision volume, PT collision is enough prevent penertation?
        COLLISION_UTILS::calc_imminent_self_PT_collision_impulse(cudaPol,
            vtemp,"x","v",
            tris,
            halfedges,
            imminent_collision_thickness,
            0,
            imminent_PT_collision_buffer);
        std::cout << "nm_PT_collision : " << imminent_PT_collision_buffer.size() << std::endl;

        COLLISION_UTILS::calc_imminent_self_EE_collision_impulse(cudaPol,
            vtemp,"x","v",
            edges,
            imminent_collision_thickness,
            0,
            imminent_EE_collision_buffer);
        std::cout << "nm_EE_collision : " << imminent_EE_collision_buffer.size() << std::endl;
        // resolve imminent PT collision
        
        dtiles_t imminent_PT_collision_vis(verts.get_allocator(),{
                {"collision_normal",3},
                {"pt",3},
                {"pp",3}},imminent_PT_collision_buffer.size());

        cudaPol(zs::range(imminent_PT_collision_buffer.size()),[
            vtemp = proxy<space>({},vtemp),
            imminent_PT_collision_vis = proxy<space>({},imminent_PT_collision_vis),
            imminent_PT_collision_buffer = proxy<space>({},imminent_PT_collision_buffer)] ZS_LAMBDA(auto ci) mutable {
                auto inds = imminent_PT_collision_buffer.pack(dim_c<4>,"inds",ci,int_c);
                auto bary = imminent_PT_collision_buffer.pack(dim_c<4>,"bary",ci);

                vec3 ps[4] = {};
                for(int i = 0;i != 4;++i)
                    ps[i] = vtemp.pack(dim_c<3>,"x",inds[i]);

                auto proj_t = vec3::zeros();
                auto pr = vec3::zeros();
                for(int i = 0;i != 4;++i)
                    pr += bary[i] * ps[i];

                for(int i = 0;i != 3;++i)
                    proj_t -= bary[i] * ps[i];

                auto collision_normal = pr.normalized();

                imminent_PT_collision_vis.tuple(dim_c<3>,"collision_normal",ci) = collision_normal;

                imminent_PT_collision_vis.tuple(dim_c<3>,"pt",ci) = proj_t;
                imminent_PT_collision_vis.tuple(dim_c<3>,"pp",ci) = vtemp.pack(dim_c<3>,"x",inds[3]);
        });
        imminent_PT_collision_vis = imminent_PT_collision_vis.clone({zs::memsrc_e::host});

        auto prim_PT = std::make_shared<PrimitiveObject>();
        auto& vis_verts_PT = prim_PT->verts;
        auto& vis_lines_PT = prim_PT->lines;
        vis_verts_PT.resize(imminent_PT_collision_buffer.size() * 2);  
        vis_lines_PT.resize(imminent_PT_collision_buffer.size());

        auto nrm_prim_PT = std::make_shared<PrimitiveObject>();
        auto& nrm_verts_PT = nrm_prim_PT->verts;
        auto& nrm_lines_PT = nrm_prim_PT->lines;
        nrm_verts_PT.resize(imminent_PT_collision_buffer.size() * 2);  
        nrm_lines_PT.resize(imminent_PT_collision_buffer.size());

        ompPol(zs::range(imminent_PT_collision_buffer.size()),[
            &vis_verts_PT,&vis_lines_PT,
            &nrm_verts_PT,&nrm_lines_PT,
            imminent_PT_collision_vis = proxy<omp_space>({},imminent_PT_collision_vis)] (int ci) mutable {
                auto cnrm = imminent_PT_collision_vis.pack(dim_c<3>,"collision_normal",ci);
                auto pt = imminent_PT_collision_vis.pack(dim_c<3>,"pt",ci);
                auto pp = imminent_PT_collision_vis.pack(dim_c<3>,"pp",ci);
                auto pn = pp + cnrm;

                vis_verts_PT[ci * 2 + 0] = pp.to_array();
                vis_verts_PT[ci * 2 + 1] = pt.to_array();
                vis_lines_PT[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};

                nrm_verts_PT[ci * 2 + 0] = pp.to_array();
                nrm_verts_PT[ci * 2 + 1] = pn.to_array();
                nrm_lines_PT[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });

        set_output("prim_PT",std::move(prim_PT));
        set_output("cnrm_PT",std::move(nrm_prim_PT));

        std::cout << "output PT VIS" << std::endl;


        dtiles_t imminent_EE_collision_vis(verts.get_allocator(),{
            {"collision_normal",3},
            {"pt",3},
            {"pp",3}},imminent_EE_collision_buffer.size());     

        cudaPol(zs::range(imminent_EE_collision_buffer.size()),[
            vtemp = proxy<space>({},vtemp),
            imminent_EE_collision_vis = proxy<space>({},imminent_EE_collision_vis),
            imminent_EE_collision_buffer = proxy<space>({},imminent_EE_collision_buffer)] ZS_LAMBDA(auto ci) mutable {
                auto inds = imminent_EE_collision_buffer.pack(dim_c<4>,"inds",ci,int_c);
                auto bary = imminent_EE_collision_buffer.pack(dim_c<4>,"bary",ci);

                vec3 ps[4] = {};
                for(int i = 0;i != 4;++i)
                    ps[i] = vtemp.pack(dim_c<3>,"x",inds[i]);

                auto pr = vec3::zeros();
                for(int i = 0;i != 4;++i)
                    pr += bary[i] * ps[i];

                auto proja = vec3::zeros();
                auto projb = vec3::zeros();
                for(int i = 0;i != 2;++i) {
                    proja -= bary[i] * ps[i];
                    projb += bary[i + 2] * ps[i + 2];
                }
                
                auto collision_normal = pr.normalized();   
                
                imminent_EE_collision_vis.tuple(dim_c<3>,"collision_normal",ci) = collision_normal;
                imminent_EE_collision_vis.tuple(dim_c<3>,"pt",ci) = proja;
                imminent_EE_collision_vis.tuple(dim_c<3>,"pp",ci) = projb;
        });   
        imminent_EE_collision_vis = imminent_EE_collision_vis.clone({zs::memsrc_e::host});

        std::cout << "output EE VIS" << std::endl;

        auto prim_EE = std::make_shared<PrimitiveObject>();
        auto& vis_verts_EE = prim_EE->verts;
        auto& vis_lines_EE = prim_EE->lines;
        vis_verts_EE.resize(imminent_EE_collision_buffer.size() * 2);  
        vis_lines_EE.resize(imminent_EE_collision_buffer.size());

        auto nrm_prim_EE = std::make_shared<PrimitiveObject>();
        auto& nrm_verts_EE = nrm_prim_EE->verts;
        auto& nrm_lines_EE = nrm_prim_EE->lines;
        nrm_verts_EE.resize(imminent_EE_collision_buffer.size() * 2);  
        nrm_lines_EE.resize(imminent_EE_collision_buffer.size());

        ompPol(zs::range(imminent_EE_collision_buffer.size()),[
            &vis_verts_EE,&vis_lines_EE,
            &nrm_verts_EE,&nrm_lines_EE,
            imminent_EE_collision_vis = proxy<omp_space>({},imminent_EE_collision_vis)] (int ci) mutable {
                auto cnrm = imminent_EE_collision_vis.pack(dim_c<3>,"collision_normal",ci);
                auto pt = imminent_EE_collision_vis.pack(dim_c<3>,"pt",ci);
                auto pp = imminent_EE_collision_vis.pack(dim_c<3>,"pp",ci);
                auto pn = pp + cnrm;

                vis_verts_EE[ci * 2 + 0] = pp.to_array();
                vis_verts_EE[ci * 2 + 1] = pt.to_array();
                vis_lines_EE[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};

                nrm_verts_EE[ci * 2 + 0] = pp.to_array();
                nrm_verts_EE[ci * 2 + 1] = pn.to_array();
                nrm_lines_EE[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
        });

        set_output("prim_EE",std::move(prim_EE));
        set_output("cnrm_EE",std::move(nrm_prim_EE));

    }    
};

ZENDEFNODE(VisualizeImminentCollision, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                // {"string","pscaleTag","pscale"},
                                // {"float","repeling_strength","1.0"},
                                {"float","immc_thickness","0.01"},
                                // {"int","nm_imminent_iters","1"},
                                // {"float","imm_restitution","0.1"},
                                // {"float","imm_relaxation","0.25"},
                                // {"bool","add_repulsion_force","0"},
                            },
							{{"prim_PT"},{"cnrm_PT"},{"prim_EE"},{"cnrm_EE"}},
							{},
							{"PBD"}});


struct DetangleCCDCollision : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using vec4 = zs::vec<T,4>;
    using vec4i = zs::vec<int,4>;
    using dtiles_t = zs::TileVector<T,32>;
    
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using lbvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;
        // constexpr auto MAX_PT_COLLISION_PAIRS = 1000000;
        
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");

        auto restitution_rate = get_input2<float>("restitution");
        auto relaxation_rate = get_input2<float>("relaxation");

        auto thickness = get_input2<float>("thickness");

        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();    
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 

        dtiles_t vtemp(verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },(size_t)verts.size());

        auto nm_ccd_iters = get_input2<int>("nm_ccd_iters");

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            current_x_tag = zs::SmallString(current_x_tag),
            pre_x_tag = zs::SmallString(pre_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp("minv",vi) = verts("minv",vi);
        });


        lbvh_t triBvh{},eBvh{};

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

        auto do_ee_detection = get_input2<bool>("do_ee_detection");
        auto do_pt_detection = get_input2<bool>("do_pt_detection");
    
        zs::Vector<int> nm_ccd_collision{verts.get_allocator(),1};

        std::cout << "resolve continous collision " << std::endl;

        auto res_threshold = thickness * 0.01;
        res_threshold = res_threshold < 1e-3 ? 1e-3 : res_threshold;

        for(int iter = 0;iter != nm_ccd_iters;++iter) {

            cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
            cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

            nm_ccd_collision.setVal(0);

            if(do_pt_detection) {
                std::cout << "do continous self PT cololision impulse" << std::endl;

                auto do_bvh_refit = iter > 0;
                COLLISION_UTILS::calc_continous_self_PT_collision_impulse(cudaPol,
                    verts,
                    vtemp,"x","v",
                    tris,
                    // thickness,
                    triBvh,
                    do_bvh_refit,
                    impulse_buffer,
                    impulse_count);
            }

            if(do_ee_detection) {
                std::cout << "do continous self EE cololision impulse" << std::endl;
                auto do_bvh_refit = iter > 0;
                COLLISION_UTILS::calc_continous_self_EE_collision_impulse(cudaPol,
                    verts,
                    vtemp,"x","v",
                    edges,
                    eBvh,
                    do_bvh_refit,
                    impulse_buffer,
                    impulse_count);
            }

            std::cout << "apply CCD impulse" << std::endl;
            cudaPol(zs::range(verts.size()),[
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
                if(impulse_count[vi] == 0)
                    return;
                if(impulse_buffer[vi].norm() < eps)
                    return;

                auto impulse = relaxation_rate * impulse_buffer[vi] / impulse_count[vi];
                if(impulse.norm() > res_threshold)
                    atomic_add(exec_tag,&nm_ccd_collision[0],1);

                // auto dv = impulse
                vtemp.tuple(dim_c<3>,"v",vi) = vtemp.pack(dim_c<3>,"v",vi) + impulse;
                // for(int i = 0;i != 3;++i)   
                //     atomic_add(exec_tag,&vtemp("v",i,vi),dv[i]);
            });

            std::cout << "nm_ccd_collision : " << nm_ccd_collision.getVal() << std::endl;
            if(nm_ccd_collision.getVal() == 0)
                break;
        }
        std::cout << "finish solving continous collision " << std::endl;
        cudaPol(zs::range(verts.size()),[
            vtemp = proxy<space>({},vtemp),
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>,xtag,vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
        });

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(DetangleCCDCollision, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                {"int","nm_ccd_iters","1"},
                                // {"string","pscaleTag","pscale"},
                                {"float","thickness","0.1"},
                                // {"float","immc_thickness","0.01"},
                                // {"int","nm_imminent_iters","1"},
                                {"float","restitution","0.1"},
                                {"float","relaxation","1"},
                                {"bool","do_ee_detection","1"},
                                {"bool","do_pt_detection","1"},
                                // {"bool","add_repulsion_force","0"},
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});


struct VisualizeContinousCollision : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using dtiles_t = zs::TileVector<T,32>;

    virtual void apply() override {
        using namespace zs;
        using lbvh_t = typename ZenoLinearBvh::lbvh_t;
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        constexpr auto omp_space = execspace_e::openmp;  
        auto ompPol = omp_exec();  
        
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");

        auto thickness = get_input2<float>("thickness");

        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();    
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag]; 

        dtiles_t vtemp(verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },(size_t)verts.size());

        lbvh_t triBvh{},eBvh{};
        // auto nm_ccd_iters = get_input2<int>("nm_ccd_iters");

        cudaPol(zs::range(verts.size()),[
            verts = proxy<cuda_space>({},verts),
            vtemp = proxy<cuda_space>({},vtemp),
            current_x_tag = zs::SmallString(current_x_tag),
            pre_x_tag = zs::SmallString(pre_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp("minv",vi) = verts("minv",vi);
        });
 
        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

        cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
        cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

        COLLISION_UTILS::calc_continous_self_PT_collision_impulse(cudaPol,
            verts,
            vtemp,"x","v",
            tris,
            // thickness,
            triBvh,
            false,
            impulse_buffer,
            impulse_count);
        
        cudaPol(zs::range(impulse_buffer.size()),[
            impulse_buffer = proxy<cuda_space>(impulse_buffer),
            impulse_count = proxy<cuda_space>(impulse_count)] ZS_LAMBDA(int vi) mutable {
                if(impulse_count[vi] > 0) {
                    impulse_buffer[vi] = impulse_buffer[vi] / (T)impulse_count[vi];
                    // printf("impulse[%d] : %f %f %f\n",vi
                    //     ,(float)impulse_buffer[vi][0]
                    //     ,(float)impulse_buffer[vi][1]
                    //     ,(float)impulse_buffer[vi][2]);
                }
        });

        dtiles_t ccd_PT_collision_vis(verts.get_allocator(),{
            {"impulse",3},
            {"p0",3},
            {"p1",3},
            {"pp",3}
        },verts.size());

        cudaPol(zs::range(verts.size()),[
            ccd_PT_collision_vis = proxy<cuda_space>({},ccd_PT_collision_vis),
            impulse_buffer = proxy<cuda_space>(impulse_buffer),
            vtemp = proxy<cuda_space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                ccd_PT_collision_vis.tuple(dim_c<3>,"impulse",vi) = impulse_buffer[vi];
                ccd_PT_collision_vis.tuple(dim_c<3>,"p0",vi) = vtemp.pack(dim_c<3>,"x",vi);
                ccd_PT_collision_vis.tuple(dim_c<3>,"p1",vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
                ccd_PT_collision_vis.tuple(dim_c<3>,"pp",vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi) + impulse_buffer[vi];
        });

        ccd_PT_collision_vis = ccd_PT_collision_vis.clone({zs::memsrc_e::host});
        auto prim_PT = std::make_shared<PrimitiveObject>();
        auto& vis_verts_PT = prim_PT->verts;
        auto& vis_lines_PT = prim_PT->lines;
        vis_verts_PT.resize(verts.size() * 2);
        vis_lines_PT.resize(verts.size());

        auto scale = get_input2<float>("scale");

        ompPol(zs::range(verts.size()),[
            &vis_verts_PT,&vis_lines_PT,
            scale = scale,
            ccd_PT_collision_vis = proxy<omp_space>({},ccd_PT_collision_vis)] (int vi) mutable {
                auto p = ccd_PT_collision_vis.pack(dim_c<3>,"p0",vi);
                auto impulse = ccd_PT_collision_vis.pack(dim_c<3>,"impulse",vi);
                auto pc = p + impulse * scale;
                vis_verts_PT[vi * 2 + 0] = p.to_array();
                vis_verts_PT[vi * 2 + 1] = pc.to_array();
                vis_lines_PT[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
        });

        set_output("impulse_PT",std::move(prim_PT));

        cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
        cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});   
    
        COLLISION_UTILS::calc_continous_self_EE_collision_impulse(cudaPol,
            verts,
            vtemp,"x","v",
            edges,
            eBvh,
            false,
            impulse_buffer,
            impulse_count);

        cudaPol(zs::range(impulse_buffer.size()),[
            impulse_buffer = proxy<cuda_space>(impulse_buffer),
            impulse_count = proxy<cuda_space>(impulse_count)] ZS_LAMBDA(int vi) mutable {
                if(impulse_count[vi] > 0) {
                    impulse_buffer[vi] = impulse_buffer[vi] / (T)impulse_count[vi];
                    // printf("impulse[%d] : %f %f %f\n",vi
                    //     ,(float)impulse_buffer[vi][0]
                    //     ,(float)impulse_buffer[vi][1]
                    //     ,(float)impulse_buffer[vi][2]);
                }
        });
        
        dtiles_t ccd_EE_collision_vis(verts.get_allocator(),{
            {"impulse",3},
            {"p0",3},
            {"p1",3},
            {"pp",3}
        },verts.size());

        cudaPol(zs::range(verts.size()),[
            ccd_EE_collision_vis = proxy<cuda_space>({},ccd_EE_collision_vis),
            impulse_buffer = proxy<cuda_space>(impulse_buffer),
            vtemp = proxy<cuda_space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                ccd_EE_collision_vis.tuple(dim_c<3>,"impulse",vi) = impulse_buffer[vi];
                ccd_EE_collision_vis.tuple(dim_c<3>,"p0",vi) = vtemp.pack(dim_c<3>,"x",vi);
                ccd_EE_collision_vis.tuple(dim_c<3>,"p1",vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
                ccd_EE_collision_vis.tuple(dim_c<3>,"pp",vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi) + impulse_buffer[vi];
        });

        ccd_EE_collision_vis = ccd_EE_collision_vis.clone({zs::memsrc_e::host});
        auto prim_EE = std::make_shared<PrimitiveObject>();
        auto& vis_verts_EE = prim_EE->verts;
        auto& vis_lines_EE = prim_EE->lines;
        vis_verts_EE.resize(verts.size() * 2);
        vis_lines_EE.resize(verts.size());      
        
        ompPol(zs::range(verts.size()),[
            &vis_verts_EE,&vis_lines_EE,
            scale = scale,
            ccd_EE_collision_vis = proxy<omp_space>({},ccd_EE_collision_vis)] (int vi) mutable {
                auto p = ccd_EE_collision_vis.pack(dim_c<3>,"p0",vi);
                auto impulse = ccd_EE_collision_vis.pack(dim_c<3>,"impulse",vi);
                auto pc = p + impulse * scale;
                vis_verts_EE[vi * 2 + 0] = p.to_array();
                vis_verts_EE[vi * 2 + 1] = pc.to_array();
                vis_lines_EE[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
        });

        set_output("impulse_EE",std::move(prim_EE));  
    }
};

ZENDEFNODE(VisualizeContinousCollision, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                // {"string","pscaleTag","pscale"},
                                // {"float","repeling_strength","1.0"},
                                {"float","thickness","0.01"},
                                // {"int","nm_imminent_iters","1"},
                                // {"float","imm_restitution","0.1"},
                                {"float","scale","0.25"},
                                // {"bool","add_repulsion_force","0"},
                            },
							{{"impulse_PT"},{"impulse_EE"}},
							{},
							{"PBD"}});

};