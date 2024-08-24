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

        
        auto among_same_group = get_input2<bool>("among_same_group");
        auto among_different_groups = get_input2<bool>("among_different_groups");

        int group_strategy = 0;
        group_strategy |= (among_same_group ? 1 : 0);
        group_strategy |= (among_different_groups ? 2 : 0);

        auto boundary_velocity_scale = get_input2<float>("boundary_velocity_scale");
        // auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        // auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto &kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = kboundary->getQuadraturePoints();

        zs::bht<int,2,int> csPT{verts.get_allocator(),MAX_COLLISION_PAIRS};csPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEE{edges.get_allocator(),MAX_COLLISION_PAIRS};csEE.reset(cudaPol,true);

        auto collision_group_name = get_input2<std::string>("group_name");

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"X",3},
            {"v",3},
            {"minv",1},
            {"m",1},
            {"collision_cancel",1},
            {"collision_group",1}
        },verts.size() + kverts.size()};

        // TILEVEC_OPS::copy<3>(cudaPol,verts,"X",vtemp,"X");
        TILEVEC_OPS::copy(cudaPol,verts,collision_group_name,vtemp,"collision_group");
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            pre_x_tag = zs::SmallString(pre_x_tag),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"X",vi) = verts.pack(dim_c<3>,"X",vi);
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
            collision_group_name = zs::SmallString(collision_group_name),
            boundary_velocity_scale = boundary_velocity_scale,
            w = w] ZS_LAMBDA(int kvi) mutable {
                auto cur_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 -  w) + kverts.pack(dim_c<3>,"x",kvi) *  w;
                auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                vtemp("collision_group",kvi + voffset) = kverts(collision_group_name,kvi);
                // for alignment, we directly assign the current boundary as reference shape
                vtemp.tuple(dim_c<3>,"X",kvi + voffset) = kverts.pack(dim_c<3>,"x",kvi);
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

        auto do_jacobi_iter = get_input2<bool>("do_jacobi_iter");

        auto nm_accept_collisions = get_input2<int>("nm_accept_collisions");

        for(int iter = 0;iter != nm_ccd_iters;++iter) {

            cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
            cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

            nm_ccd_collision.setVal(0);

            if(do_pt_detection) {
                // std::cout << "do continous self PT cololision impulse" << std::endl;

                auto do_bvh_refit = iter > 0;
                if(do_jacobi_iter) {
                    COLLISION_UTILS::calc_continous_self_PT_collision_impulse(cudaPol,
                        vtemp,
                        vtemp,
                        vtemp,"x","v",
                        ttemp,
                        thickness,
                        triBvh,
                        do_bvh_refit,
                        csPT,
                        impulse_buffer,
                        impulse_count,true,true);
                }else {
                    COLLISION_UTILS::calc_continous_self_PT_collision_impulse_with_toc(cudaPol,
                        vtemp,
                        vtemp,
                        vtemp,"x","v",
                        ttemp,
                        (T)0.0,
                        thickness,
                        triBvh,
                        do_bvh_refit,
                        csPT,
                        impulse_buffer,
                        impulse_count,true,true,false,group_strategy);
                    // std::cout << "nm_PT_continuous_collisions : " << csPT.size() << std::endl;
                }
            }

            if(do_ee_detection) {
                // std::cout << "do continous self EE cololision impulse" << std::endl;
                auto do_bvh_refit = iter > 0;
                if(do_jacobi_iter) {
                    COLLISION_UTILS::calc_continous_self_EE_collision_impulse(cudaPol,
                        vtemp,
                        vtemp,
                        vtemp,"x","v",
                        etemp,
                        thickness,
                        0,
                        edges.size(),
                        eBvh,
                        do_bvh_refit,
                        csEE,
                        impulse_buffer,
                        impulse_count,true,true);
                } else {
                    COLLISION_UTILS::calc_continous_self_EE_collision_impulse_with_toc(cudaPol,
                        vtemp,
                        vtemp,
                        vtemp,"x","v",
                        etemp,
                        (T)0.0,
                        thickness,
                        (size_t)0,
                        edges.size(),
                        eBvh,
                        do_bvh_refit,
                        csEE,
                        impulse_buffer,
                        impulse_count,true,true,false,group_strategy);
                    // std::cout << "nm_EE_continuous_collisions : " << csPT.size() << std::endl;
                }
            }

            // std::cout << "apply CCD impulse" << std::endl;
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

            // std::cout << "nm_kinematic_ccd_collision : " << nm_ccd_collision.getVal() << std::endl;
            if(nm_ccd_collision.getVal() <= nm_accept_collisions)
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
                                {gParamType_String,"current_x_tag","x"},
                                {gParamType_String,"previous_x_tag","px"},
                                {gParamType_Int,"nm_ccd_iters","1"},
                                {gParamType_Int,"nm_accept_collisions","0"},
                                {gParamType_Float,"thickness","0.1"},
                                {gParamType_Float,"restitution","0.1"},
                                {gParamType_Float,"relaxation","1"},
                                {gParamType_String,"group_name","groupName"},
                                {"boundary"},
                                {gParamType_Bool,"do_jacobi_iter","0"},
                                {gParamType_Bool,"do_ee_detection","1"},
                                {gParamType_Bool,"do_pt_detection","1"},
                                {gParamType_Int,"substep_id","0"},
                                {gParamType_Int,"nm_substeps","1"},
                                {gParamType_Float,"boundary_velocity_scale","1"},
                                {gParamType_Bool,"among_same_group","1"},
                                {gParamType_Bool,"among_different_groups","1"}
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});


};