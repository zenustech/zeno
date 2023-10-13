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
#include "../fem/collision_energy/evaluate_collision.hpp"

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

struct DetangleKineImminentCollision : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using dtiles_t = zs::TileVector<T,32>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto execTag = wrapv<space>{};

        auto cudaPol = cuda_exec();
        using lbvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        // constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;    
        
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");      
        // auto repel_strength = get_input2<float>("repeling_strength");
        auto imminent_collision_thickness = get_input2<float>("thickness");
        auto res_threshold = imminent_collision_thickness * 0.01;

        auto relaxation_rate = get_input2<float>("relaxation");

        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];

        auto boundary = get_input2<ZenoParticles>("boundary");
        auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        auto& kverts = boundary->getParticles();
        const auto& ktris = boundary->getQuadraturePoints();
        const auto &kedges = (*boundary)[ZenoParticles::s_surfEdgeTag];

        dtiles_t vtemp(verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },(size_t)verts.size());

        dtiles_t kvtemp(kverts.get_allocator(),{
            {"x",3},
            {"v",3}},(size_t)kverts.size());    
            
        TILEVEC_OPS::copy<3>(cudaPol,verts,pre_x_tag,vtemp,"x",0);
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            current_x_tag = zs::SmallString(current_x_tag),
            pre_x_tag = zs::SmallString(pre_x_tag),
            vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
        });
        TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv",0);     
        
        TILEVEC_OPS::copy<3>(cudaPol,kverts,pre_kx_tag,vtemp,"x",0);
        cudaPol(zs::range(kverts.size()),[
            kverts = proxy<space>({},kverts),
            current_kx_tag = zs::SmallString(current_kx_tag),
            pre_kx_tag = zs::SmallString(pre_kx_tag),
            kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
                kvtemp.tuple(dim_c<3>,"v",kvi) = kverts.pack(dim_c<3>,current_kx_tag,kvi) - kverts.pack(dim_c<3>,pre_kx_tag,kvi);
        });

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

        auto nm_dcd_iters = get_input2<int>("nm_dcd_iters");
        auto do_pt_detection = get_input2<bool>("use_PT");
        auto do_ee_detection = get_input2<bool>("use_EE");       

        zs::Vector<int> nm_DCD_collision{verts.get_allocator(),(size_t)1};

        lbvh_t triBvh{},ktriBvh{},kedgeBvh{};


        for(int it = 0;it != nm_dcd_iters;++it) {
            auto do_refit = it > 0;
            if(!do_pt_detection && !do_ee_detection)
                break;

            cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::zeros();});
            cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& cnt) mutable {cnt = 0;});
            nm_DCD_collision.setVal(0);

            if(do_pt_detection) {
                COLLISION_UTILS::calc_imminent_kinematic_PT_collision_impulse(cudaPol,
                    vtemp,
                    vtemp,"x","v",
                    kvtemp,"x","v",
                    tris,ktris,
                    imminent_collision_thickness,
                    triBvh,ktriBvh,
                    do_refit,
                    impulse_buffer,
                    impulse_count);
            }

            if(do_ee_detection) {
                COLLISION_UTILS::calc_imminent_kinematic_EE_collision_impulse(cudaPol,
                    vtemp,
                    vtemp,"x","v",
                    kvtemp,"x","v",
                    edges,kedges,
                    imminent_collision_thickness,
                    kedgeBvh,
                    do_refit,
                    impulse_buffer,
                    impulse_count);
            }

            std::cout << "apply kinematic DCD impulse" << std::endl;
            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                vtemp = proxy<space>({},vtemp),
                impulse_buffer = proxy<space>(impulse_buffer),
                impulse_count = proxy<space>(impulse_count),
                relaxation_rate = relaxation_rate,
                res_threshold = res_threshold,
                nm_DCD_collision = proxy<space>(nm_DCD_collision),
                eps = eps,
                execTag = execTag] ZS_LAMBDA(int vi) mutable {
                if(impulse_count[vi] == 0)
                    return;
                if(impulse_buffer[vi].norm() < eps)
                    return;

                auto impulse = relaxation_rate * impulse_buffer[vi] / impulse_count[vi];
                if(impulse.norm() > res_threshold)
                    atomic_add(execTag,&nm_DCD_collision[0],1);

                vtemp.tuple(dim_c<3>,"v",vi) = vtemp.pack(dim_c<3>,"v",vi) + impulse;
            });

            auto dcd_count = nm_DCD_collision.getVal(0);
            if(dcd_count == 0)
                break;
            else 
                std::cout << "nm DCD colisions : " << dcd_count << std::endl;
        }

        std::cout << "finish solving DCD collision " << std::endl;
        cudaPol(zs::range(verts.size()),[
            vtemp = proxy<space>({},vtemp),
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                verts.tuple(dim_c<3>,xtag,vi) = vtemp.pack(dim_c<3>,"x",vi) + vtemp.pack(dim_c<3>,"v",vi);
        });   

        set_output("zsparticles",zsparticles);
        set_output("boundary",boundary);
    }
};

ZENDEFNODE(DetangleKineImminentCollision, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                {"float","thickness","0.1"},
                                {"float","relaxation","1"},
                                {"boundary"},
                                {"string","current_kx_tag","x"},
                                {"string","previous_kx_tag","px"},
                                {"int","nm_dcd_iters","1"},
                                {"bool","use_EE","1"},
                                {"bool","use_PT","1"},
                                // {"bool","add_repulsion_force","0"},
                            },
							{{"zsparticles"},{"boundary"}},
							{},
							{"PBD"}});

// struct VisualizeKineImminentCollision : INode {
//     using T = float;
//     using vec3 = zs::vec<T,3>;
//     using dtiles_t = zs::TileVector<T,32>;
//     using lbvh_t = typename ZenoLinearBvh::lbvh_t;
//     using bv_t = typename ZenoLinearBvh::lbvh_t::Box;

//     virtual void apply() override {
//         using namespace zs;
//         constexpr auto cuda_space = execspace_e::cuda;
//         auto cuda_exec_tag = wrapv<cuda_space>{};
//         auto cudaPol = cuda_exec();

//         constexpr auto omp_space = execspace_e::omp;
//         auto omp_exc_tag = wrapv<omp_space>{};
//         auto ompPol = omp_exec();

//         auto zsparticles = get_input<ZenoParticles>("zsparticles");
//         auto current_x_tag = get_input2<std::string>("current_x_tag");
//         auto pre_x_tag = get_input2<std::string>("previous_x_tag");    
//         auto imminent_collision_thickness = get_input2<float>("thickness");
//         auto res_threshold = imminent_collision_thickness * 0.01;

//         auto& verts = zsparticles->getParticles();
//         const auto& tris = zsparticles->getQuadraturePoints();
//         const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];

//         auto boundary = get_input2<ZenoParticles>("boundary");
//         auto current_kx_tag = get_input2<std::string>("current_kx_tag");
//         auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
//         auto& kverts = boundary->getParticles();
//         const auto& ktris = boundary->getQuadraturePoints();
//         const auto &kedges = (*boundary)[ZenoParticles::s_surfEdgeTag];

//         dtiles_t vtemp(verts.get_allocator(),{
//             {"x",3},
//             {"v",3},
//             {"ori_v",3},
//             {"minv",1}
//         },(size_t)verts.size());

//         dtiles_t kvtemp(kverts.get_allocator(),{
//             {"x",3},
//             {"v",3}},(size_t)kverts.size());    

//         TILEVEC_OPS::copy<3>(cudaPol,verts,pre_x_tag,vtemp,"x",0);
//         cudaPol(zs::range(verts.size()),[
//             verts = proxy<cuda_space>({},verts),
//             current_x_tag = zs::SmallString(current_x_tag),
//             pre_x_tag = zs::SmallString(pre_x_tag),
//             vtemp = proxy<cuda_space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
//                 vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
//                 vtemp.tuple(dim_c<3>,"ori_v",vi) = vtemp.pack(dim_c<3>,"v",vi);
//         });
//         TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv",0);     
        
//         TILEVEC_OPS::copy<3>(cudaPol,kverts,pre_kx_tag,vtemp,"x",0);
//         cudaPol(zs::range(kverts.size()),[
//             kverts = proxy<cuda_space>({},kverts),
//             current_kx_tag = zs::SmallString(current_kx_tag),
//             pre_kx_tag = zs::SmallString(pre_kx_tag),
//             kvtemp = proxy<cuda_space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
//                 kvtemp.tuple(dim_c<3>,"v",kvi) = kverts.pack(dim_c<3>,current_kx_tag,kvi) - kverts.pack(dim_c<3>,pre_kx_tag,kvi);
//         });

//         zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
//         zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

//         auto nm_dcd_iters = get_input2<int>("nm_dcd_iters");
//         lbvh_t triBvh{},ktriBvh{},kedgeBvh{};
    
//     }
// };

struct DetangleKineCCDCollision : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    using dtiles_t = zs::TileVector<T,32>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        using lbvh_t = ZenoLinearBvh::lbvh_t;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;     
        
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto current_x_tag = get_input2<std::string>("current_x_tag");
        auto pre_x_tag = get_input2<std::string>("previous_x_tag");      
        
        auto thickness = get_input2<float>("thickness");
        auto relaxation_rate = get_input2<float>("relaxation");

        auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->getQuadraturePoints();
        const auto &edges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];

        auto kboundary = get_input<ZenoParticles>("boundary");
        auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto& ktris = kboundary->getQuadraturePoints();
        const auto& kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];

        dtiles_t vtemp(verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },(size_t)verts.size());

        dtiles_t kvtemp(kverts.get_allocator(),{
            {"x",3},
            {"v",3}},(size_t)kverts.size());

        TILEVEC_OPS::copy<3>(cudaPol,verts,pre_x_tag,vtemp,"x",0);
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            current_x_tag = zs::SmallString(current_x_tag),
            pre_x_tag = zs::SmallString(pre_x_tag),
            vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - verts.pack(dim_c<3>,pre_x_tag,vi);
        });
        TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv",0);     
        
        TILEVEC_OPS::copy<3>(cudaPol,kverts,pre_kx_tag,vtemp,"x",0);
        cudaPol(zs::range(kverts.size()),[
            kverts = proxy<space>({},kverts),
            current_kx_tag = zs::SmallString(current_kx_tag),
            pre_kx_tag = zs::SmallString(pre_kx_tag),
            kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
                kvtemp.tuple(dim_c<3>,"v",kvi) = kverts.pack(dim_c<3>,current_kx_tag,kvi) - kverts.pack(dim_c<3>,pre_kx_tag,kvi);
        });
        
        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

        auto nm_ccd_iters = get_input2<int>("nm_ccd_iters");
        auto do_pt_detection = get_input2<bool>("use_PT");
        auto do_ee_detection = get_input2<bool>("use_EE");       

        zs::Vector<int> nm_CCD_collision{verts.get_allocator(),(size_t)1};
        auto vn_threshold = 5e-3;

        lbvh_t triBvh{},ktriBvh{},kedgeBvh{};

        auto res_threshold = thickness * 0.01;

        for(int it = 0;it != nm_ccd_iters;++it) {
            cudaPol(zs::range(impulse_buffer),[] ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
            cudaPol(zs::range(impulse_count),[] ZS_LAMBDA(auto& cnt) mutable {cnt = 0;});

            nm_CCD_collision.setVal(0);

            if(do_pt_detection) {
                auto do_refit = it > 0;
                COLLISION_UTILS::calc_continous_kinematic_PT_collision_impulse(cudaPol,
                    vtemp,
                    vtemp,"x","v",
                    kvtemp,"x","v",
                    tris,
                    ktris,
                    triBvh,
                    ktriBvh,
                    do_refit,
                    impulse_buffer,
                    impulse_count);
            }    
            if(do_ee_detection) {
                auto do_refit = it > 0;
                COLLISION_UTILS::calc_continous_kinematic_EE_collision_impulse(cudaPol,
                    vtemp,
                    vtemp,"x","v",
                    kvtemp,"x","v",
                    edges,
                    kedges,
                    kedgeBvh,
                    do_refit,
                    impulse_buffer,
                    impulse_count);
            }   
            
            if(do_ee_detection || do_pt_detection) {
                std::cout << "apply kinematic CCD impulse" << std::endl;
                cudaPol(zs::range(verts.size()),[
                    verts = proxy<space>({},verts),
                    vtemp = proxy<space>({},vtemp),
                    impulse_buffer = proxy<space>(impulse_buffer),
                    impulse_count = proxy<space>(impulse_count),
                    relaxation_rate = relaxation_rate,
                    nm_CCD_collision = proxy<space>(nm_CCD_collision),
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
                        atomic_add(exec_tag,&nm_CCD_collision[0],1);
    
                    vtemp.tuple(dim_c<3>,"v",vi) = vtemp.pack(dim_c<3>,"v",vi) + impulse;
                });
            }

            auto nm_ccd_collision = nm_CCD_collision.getVal(0);
            if(nm_ccd_collision == 0)
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
        set_output("boundary",kboundary);
    }
};

ZENDEFNODE(DetangleKineCCDCollision, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                {"float","thickness","0.1"},
                                {"float","relaxation","1"},
                                {"boundary"},
                                {"string","current_kx_tag","x"},
                                {"string","previous_kx_tag","px"},
                                {"int","nm_ccd_iters","1"},
                                {"bool","use_EE","1"},
                                {"bool","use_PT","1"},
                                // {"bool","add_repulsion_force","0"},
                            },
							{{"zsparticles"},{"boundary"}},
							{},
							{"PBD"}});




struct DetangleImminentCollisionWithBoundary : INode {
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


        auto kboundary = get_input2<ZenoParticles>("boundary");
        auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto &kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = kboundary->getQuadraturePoints();

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },verts.size() + kverts.size()};
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            pre_x_tag = zs::SmallString(pre_x_tag),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - vtemp.pack(dim_c<3>,"x",vi);
                vtemp("minv",vi) = verts("minv",vi);
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

        cudaPol(zs::range(kverts.size()),[
            voffset = verts.size(),
            vtemp = proxy<space>({},vtemp),
            kverts = proxy<space>({},kverts),
            kxtag = zs::SmallString(current_kx_tag),
            pkxtag = zs::SmallString(pre_kx_tag)] ZS_LAMBDA(int kvi) mutable {
                vtemp.tuple(dim_c<3>,"x",kvi + voffset) = kverts.pack(dim_c<3>,pkxtag,kvi);
                vtemp.tuple(dim_c<3>,"v",kvi + voffset) = kverts.pack(dim_c<3>,kxtag,kvi) - vtemp.pack(dim_c<3>,"x",kvi + voffset);
                vtemp("minv",kvi + voffset) = (T)0;
        });
        // use only PP and PT?
        // TILEVEC_OPS::copy(cudaPol,verts,current_x_tag,vtemp,"v");
        // TILEVEC_OPS::add(cudaPol,vtemp,"v",1,"x",-1,"v");
        dtiles_t imminent_collision_buffer(verts.get_allocator(),
            {
                {"inds",4},
                {"bary",4},
                {"impulse",3},
                {"collision_normal",3}
            },(size_t)0);


        auto nm_iters = get_input2<int>("nm_imminent_iters");
        auto imminent_restitution_rate = get_input2<float>("imm_restitution");
        auto imminent_relaxation_rate = get_input2<float>("imm_relaxation");


        auto do_pt_detection = get_input2<bool>("use_PT");
        auto do_ee_detection = get_input2<bool>("use_EE");

        zs::Vector<int> nm_imminent_collision{verts.get_allocator(),(size_t)1};

        // std::cout << "do imminent detangle" << std::endl;

        float vn_threshold = 5e-3;
        auto add_repulsion_force = get_input2<bool>("add_repulsion_force");
        
        for(int it = 0;it != nm_iters;++it) {

            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {verts("imminent_fail",vi) = (T)0;});

        // we use collision cell as the collision volume, PT collision is enough prevent penertation?
            if(do_pt_detection) {
                COLLISION_UTILS::calc_imminent_self_PT_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    ttemp,
                    halfedges,
                    imminent_collision_thickness,
                    0,
                    imminent_collision_buffer);
                // std::cout << "nm_imminent_PT_collision : " << imminent_collision_buffer.size() << std::endl;
            }

            if(do_ee_detection) {
                COLLISION_UTILS::calc_imminent_self_EE_collision_impulse(cudaPol,
                    vtemp,"x","v",
                    etemp,
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
                vn_threshold,
                imminent_collision_buffer,
                nm_imminent_collision);

            std::cout << "nm_self_imminent_collision : " << nm_imminent_collision.getVal(0) << std::endl;
            if(nm_imminent_collision.getVal(0) == 0) 
                break;              

            // }
        }

        if(add_repulsion_force) {
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


            // auto impulse_norm = TILEVEC_OPS::dot<3>(cudaPol,imminent_collision_buffer,"impulse","impulse");
            // std::cout << "REPEL_impulse_norm : " << impulse_norm << std::endl;

            COLLISION_UTILS::apply_impulse(cudaPol,
                vtemp,"v",
                imminent_restitution_rate,
                imminent_relaxation_rate,
                imminent_collision_buffer);
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
                                {"string","current_kx_tag","x"},
                                {"string","previous_kx_tag","px"},
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


struct DetangleCCDCollisionWithBoundary : INode {
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


        auto kboundary = get_input2<ZenoParticles>("boundary");
        auto current_kx_tag = get_input2<std::string>("current_kx_tag");
        auto pre_kx_tag = get_input2<std::string>("previous_kx_tag");
        const auto& kverts = kboundary->getParticles();
        const auto &kedges = (*kboundary)[ZenoParticles::s_surfEdgeTag];

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"v",3},
            {"minv",1}
        },verts.size() + kverts.size()};
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp),
            pre_x_tag = zs::SmallString(pre_x_tag),
            current_x_tag = zs::SmallString(current_x_tag)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,pre_x_tag,vi);
                vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,current_x_tag,vi) - vtemp.pack(dim_c<3>,"x",vi);
                vtemp("minv",vi) = verts("minv",vi);
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


        cudaPol(zs::range(kverts.size()),[
            voffset = verts.size(),
            vtemp = proxy<space>({},vtemp),
            kverts = proxy<space>({},kverts),
            kxtag = zs::SmallString(current_kx_tag),
            pkxtag = zs::SmallString(pre_kx_tag)] ZS_LAMBDA(int kvi) mutable {
                vtemp.tuple(dim_c<3>,"x",kvi + voffset) = kverts.pack(dim_c<3>,pkxtag,kvi);
                vtemp.tuple(dim_c<3>,"v",kvi + voffset) = kverts.pack(dim_c<3>,kxtag,kvi) - vtemp.pack(dim_c<3>,"x",kvi + voffset);
                vtemp("minv",kvi + voffset) = (T)0;
        });

        lbvh_t triBvh{},eBvh{};

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};

        auto do_ee_detection = get_input2<bool>("do_ee_detection");
        auto do_pt_detection = get_input2<bool>("do_pt_detection");
    
        zs::Vector<int> nm_ccd_collision{verts.get_allocator(),1};

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

ZENDEFNODE(DetangleCCDCollisionWithBoundary, {{{"zsparticles"},
                                {"string","current_x_tag","x"},
                                {"string","previous_x_tag","px"},
                                {"int","nm_ccd_iters","1"},
                                {"float","thickness","0.1"},
                                {"float","restitution","0.1"},
                                {"float","relaxation","1"},
                                {"boundary"},
                                {"string","current_kx_tag","x"},
                                {"string","previous_kx_tag","px"},
                                {"bool","do_ee_detection","1"},
                                {"bool","do_pt_detection","1"},
                                // {"bool","add_repulsion_force","0"},
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});


};