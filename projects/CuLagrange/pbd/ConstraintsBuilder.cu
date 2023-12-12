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
#include "constraint_function_kernel/constraint_types.hpp"
#include "../fem/collision_energy/evaluate_collision.hpp"


namespace zeno {

// we only need to record the topo here
// serve triangulate mesh or strands only currently
struct MakeSurfaceConstraintTopology : INode {

using bvh_t = ZenoLinearBvh::lbvh_t;
using bv_t = bvh_t::Box;
using dtiles_t = zs::TileVector<T,32>;

template <typename TileVecT>
void buildBvh(zs::CudaExecutionPolicy &pol, 
        TileVecT &verts, 
        const zs::SmallString& srcTag,
        const zs::SmallString& dstTag,
        const zs::SmallString& pscaleTag,
            bvh_t &bvh) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<bv_t> bvs{verts.get_allocator(), verts.size()};
    pol(range(verts.size()),
        [verts = proxy<space>({}, verts),
         bvs = proxy<space>(bvs),
         pscaleTag,srcTag,dstTag] ZS_LAMBDA(int vi) mutable {
            auto src = verts.template pack<3>(srcTag, vi);
            auto dst = verts.template pack<3>(dstTag, vi);
            auto pscale = verts(pscaleTag,vi);

            bv_t bv{src,dst};
            bv._min -= pscale;
            bv._max += pscale;
            bvs[vi] = bv;
        });
    bvh.build(pol, bvs);
}

virtual void apply() override {
    using namespace zs;
    using namespace PBD_CONSTRAINT;

    using vec2 = zs::vec<float,2>;
    using vec3 = zs::vec<float,3>;
    using vec4 = zs::vec<float,4>;
    using vec9 = zs::vec<float,9>;
    using vec2i = zs::vec<int,2>;
    using vec3i = zs::vec<int,3>;
    using vec4i = zs::vec<int,4>;
    using mat4 = zs::vec<int,4,4>;

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = zs::cuda_exec();

    auto source = get_input<ZenoParticles>("source");
    auto constraint = std::make_shared<ZenoParticles>();

    auto type = get_input2<std::string>("topo_type");

    if(source->category != ZenoParticles::surface)
        throw std::runtime_error("Try adding Constraint topology to non-surface ZenoParticles");

    auto& verts = source->getParticles();
    const auto& quads = source->getQuadraturePoints();

    auto relative_stiffness = get_input2<float>("relative_stiffness");
    auto uniform_xpbd_affiliation = get_input2<float>("xpbd_affiliation");
    auto damping_coeff = get_input2<float>("damping_coeff");

    auto make_empty = get_input2<bool>("make_empty_constraint");

    if(!make_empty) {

    zs::Vector<float> colors{quads.get_allocator(),0};
    zs::Vector<int> reordered_map{quads.get_allocator(),0};
    zs::Vector<int> color_offset{quads.get_allocator(),0};

    constraint->sprayedOffset = 0;
    constraint->elements = typename ZenoParticles::particles_t({
        {"relative_stiffness",1},
        {"xpbd_affiliation",1},
        {"lambda",1},
        {"damping_coeff",1},
        {"tclr",1}
    }, 0, zs::memsrc_e::device,0);
    auto &eles = constraint->getQuadraturePoints();
    constraint->setMeta(CONSTRAINT_TARGET,source.get());

    auto do_constraint_topological_coloring = get_input2<bool>("do_constraint_topological_coloring");

    if(type == "stretch") {
        constraint->setMeta(CONSTRAINT_KEY,category_c::edge_length_constraint);
        auto quads_vec = tilevec_topo_to_zsvec_topo(cudaPol,quads,wrapv<3>{});
        zs::Vector<zs::vec<int,2>> edge_topos{quads.get_allocator(),0};
        retrieve_edges_topology(cudaPol,quads_vec,edge_topos);
        eles.resize(edge_topos.size());

        if(do_constraint_topological_coloring) {
            topological_coloring(cudaPol,edge_topos,colors);
            sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
        }
        // std::cout << "quads.size() = " << quads.size() << "\t" << "edge_topos.size() = " << edge_topos.size() << std::endl;
        eles.append_channels(cudaPol,{{"inds",2},{"r",1}});

        auto rest_scale = get_input2<float>("rest_scale");

        cudaPol(zs::range(eles.size()),[
            verts = proxy<space>({},verts),
            eles = proxy<space>({},eles),
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map),
            // uniform_stiffness = uniform_stiffness,
            colors = proxy<space>(colors),
            rest_scale = rest_scale,
            // do_constraint_topological_coloring = do_constraint_topological_coloring,
            edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                eles.tuple(dim_c<2>,"inds",oei) = edge_topos[ei].reinterpret_bits(float_c);
                vec3 x[2] = {};
                for(int i = 0;i != 2;++i)
                    x[i] = verts.pack(dim_c<3>,"x",edge_topos[ei][i]);
                eles("r",oei) = (x[0] - x[1]).norm() * rest_scale;
                // printf("r[%d] : %f\n",oei,(float)eles("r",oei));
        });            

    }

    if(type == "follow_animation_constraint") {
        constexpr auto eps = 1e-6;
        constraint->setMeta(CONSTRAINT_KEY,category_c::follow_animation_constraint);
        if(!has_input<ZenoParticles>("target")) {
            std::cout << "no target specify while adding follow animation constraint" << std::endl;
            throw std::runtime_error("no target specify while adding follow animation constraint");
        }
        auto target = get_input<ZenoParticles>("target");
        if(target->getParticles().size() != verts.size()) {
            std::cout << "the size of target and the cloth not match : " << target->getParticles().size() << "\t" << source->getParticles().size() << std::endl;
            throw std::runtime_error("the size of the target and source not matched");
        }
        const auto& kverts = target->getParticles();
        if(!kverts.hasProperty("ani_mask")) {
            std::cout << "the animation target should has \'ani_mask\' nodal attribute" << std::endl;
            throw std::runtime_error("the animation target should has \'ani_mask\' nodal attribute");
        }

        zs::Vector<zs::vec<int,1>> point_topos{quads.get_allocator(),0};
        point_topos.resize(verts.size());
        cudaPol(zip(zs::range(point_topos.size()),point_topos),[] ZS_LAMBDA(const auto& id,auto& pi) mutable {pi = id;});
        // std::cout << "nm binder point : " << point_topos.size() << std::endl;
        if(do_constraint_topological_coloring) {
            topological_coloring(cudaPol,point_topos,colors,false);
            sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
        }

        eles.resize(verts.size());
        // we need an extra 'inds' tag, in case the source and animation has different topo
        eles.append_channels(cudaPol,{{"inds",1},{"follow_weight",1}});
        // cudaPol(zs::range(eles.size()),[
        //     eles = proxy<space>({},eles)] ZS_LAMBDA(int ei) mutable {eles("inds",1) = zs::reinterpret_bits<float>(ei);});
        
        // TILEVEC_OPS::copy(cudaPol,kverts,"ani_mask",eles,"follow_weight");
        cudaPol(zs::range(eles.size()),[
            kverts = proxy<space>({},kverts),
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map),
            point_topos = proxy<space>(point_topos),
            eles = proxy<space>({},eles)] ZS_LAMBDA(int oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                auto pi = point_topos[ei][0];

                auto am = kverts("ani_mask",pi);
                am = am > 1 ? 1 : am;
                am = am < 0 ? 0 : am;
                eles("follow_weight",oei) = 1 - am;
                eles("inds",oei) = zs::reinterpret_bits<float>(pi);
        });
        // not sure about effect by increasing the nodal mass
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            kverts = proxy<space>({},kverts)] ZS_LAMBDA(int vi) mutable {
                verts("minv",vi) = (1 - kverts("ani_mask",vi)) * verts("minv",vi);
        });
        constraint->setMeta(CONSTRAINT_TARGET,target.get());
    }   

    if(type == "reference_dcd_collision_constraint") {
        constexpr auto eps = 1e-6;
        constexpr auto MAX_IMMINENT_COLLISION_PAIRS = 2000000;
        auto dcd_source_xtag = get_input2<std::string>("dcd_source_xtag");
        constraint->setMeta(CONSTRAINT_KEY,category_c::dcd_collision_constraint);
        eles.append_channels(cudaPol,{{"inds",4},{"bary",4},{"type",1}});
        eles.resize(MAX_IMMINENT_COLLISION_PAIRS);

        const auto &edges = (*source)[ZenoParticles::s_surfEdgeTag];
        auto has_input_collider = has_input<ZenoParticles>("target");

        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;
        
        auto nm_verts = verts.size();
        auto nm_tris = quads.size();
        auto nm_edges = edges.size();

        if(has_input_collider) {
            auto collider = get_input<ZenoParticles>("target");
            constraint->setMeta(CONSTRAINT_TARGET,collider.get());
            const auto& kverts = collider->getParticles();
            nm_verts += collider->getParticles().size();
            nm_tris += collider->getQuadraturePoints().size();
            nm_edges += (*collider)[ZenoParticles::s_surfEdgeTag].size();
        }

        dtiles_t vtemp{verts.get_allocator(),{
            {"x",3},
            {"X",3},
            {"minv",1},
            {"dcd_collision_tag",1},
            {"collision_cancel",1}
        },nm_verts};
        TILEVEC_OPS::fill(cudaPol,vtemp,"dcd_collision_tag",0);
        TILEVEC_OPS::copy<3>(cudaPol,verts,dcd_source_xtag,vtemp,"x");
        TILEVEC_OPS::copy<3>(cudaPol,verts,"X",vtemp,"X");
        TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv");
        if(verts.hasProperty("collision_cancel"))
            TILEVEC_OPS::copy(cudaPol,verts,"collision_cancel",vtemp,"collision_cancel");
        else
            TILEVEC_OPS::fill(cudaPol,vtemp,"collision_cancel",0);

        dtiles_t etemp{edges.get_allocator(),{
            {"inds",2}
        },nm_edges};
        TILEVEC_OPS::copy<2>(cudaPol,edges,"inds",etemp,"inds");

        dtiles_t ttemp{quads.get_allocator(),{
            {"inds",3}
        },nm_tris};
        TILEVEC_OPS::copy<3>(cudaPol,quads,"inds",ttemp,"inds");

        auto imminent_collision_thickness = get_input2<float>("thickness");
        if(has_input_collider) {
            auto collider = get_input<ZenoParticles>("target");
            const auto& kverts = collider->getParticles();
            const auto& kedges = (*collider)[ZenoParticles::s_surfEdgeTag];
            const auto& ktris = collider->getQuadraturePoints();

            auto voffset = verts.size();
            auto eoffset = edges.size();
            auto toffset = quads.size();

            cudaPol(zs::range(kverts.size()),[
                kverts = proxy<space>({},kverts),
                voffset = voffset,
                pw = pw,
                // kverts_pre = proxy<space>({},kverts_pre),
                vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int kvi) mutable {
                    auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                    vtemp.tuple(dim_c<3>,"x",voffset + kvi) = pre_kvert;
                    vtemp("minv",voffset + kvi) = 0;
                    if(kverts.hasProperty("collision_cancel")) 
                        vtemp("collision_cancel",voffset + kvi) = kverts("collision_cancel",kvi);
                    else
                        vtemp("collision_cancel",voffset + kvi) = 0;
            });

            cudaPol(zs::range(kedges.size()),[
                kedges = proxy<space>({},kedges),
                etemp = proxy<space>({},etemp),
                eoffset = eoffset,
                voffset = voffset] ZS_LAMBDA(int kei) mutable {
                    auto kedge = kedges.pack(dim_c<2>,"inds",kei,int_c);
                    kedge += voffset;
                    etemp.tuple(dim_c<2>,"inds",eoffset + kei) = kedge.reinterpret_bits(float_c);
            });

            cudaPol(zs::range(ktris.size()),[
                ktris = proxy<space>({},ktris),
                ttemp = proxy<space>({},ttemp),
                toffset = toffset,
                voffset = voffset] ZS_LAMBDA(int kti) mutable {
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                    ktri += voffset;
                    ttemp.tuple(dim_c<3>,"inds",toffset + kti) = ktri.reinterpret_bits(float_c);
            });
        }

        zs::bht<int,2,int> csPT{verts.get_allocator(),(size_t)MAX_IMMINENT_COLLISION_PAIRS};csPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEE{edges.get_allocator(),(size_t)MAX_IMMINENT_COLLISION_PAIRS};csEE.reset(cudaPol,true);

    
        auto triBvh = bvh_t{};
        auto triBvs = retrieve_bounding_volumes(cudaPol,vtemp,ttemp,wrapv<3>{},imminent_collision_thickness/(float)2.0,"x");
        triBvh.build(cudaPol,triBvs);
        COLLISION_UTILS::detect_self_imminent_PT_close_proximity(cudaPol,vtemp,"x",ttemp,imminent_collision_thickness,0,triBvh,eles,csPT);

        std::cout << "nm_imminent_csPT : " << csPT.size() << std::endl;

        auto edgeBvh = bvh_t{};
        auto edgeBvs = retrieve_bounding_volumes(cudaPol,vtemp,etemp,wrapv<2>{},imminent_collision_thickness/(float)2.0,"x");
        edgeBvh.build(cudaPol,edgeBvs);  
        COLLISION_UTILS::detect_self_imminent_EE_close_proximity(cudaPol,vtemp,"x",etemp,imminent_collision_thickness,csPT.size(),edgeBvh,eles,csEE);

        std::cout << "nm_imminent_csEE : " << csEE.size() << std::endl;
        // std::cout << "csEE + csPT = " << csPT.size() + csEE.size() << std::endl;
        if(!verts.hasProperty("dcd_collision_tag"))
            verts.append_channels(cudaPol,{{"dcd_collision_tag",1}});
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                verts("dcd_collision_tag",vi) = vtemp("dcd_collision_tag",vi);
        });

        if(has_input_collider) {
            auto collider = get_input<ZenoParticles>("target");
            auto& kverts = collider->getParticles();
            if(!kverts.hasProperty("dcd_collision_tag"))
                kverts.append_channels(cudaPol,{{"dcd_collision_tag",1}});
            cudaPol(zs::range(kverts.size()),[
                kverts = proxy<space>({},kverts),
                voffset = verts.size(),
                vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int kvi) mutable {
                    kverts("dcd_collision_tag",kvi) = vtemp("dcd_collision_tag",kvi + voffset);
            });                
        }

        constraint->setMeta<size_t>(NM_DCD_COLLISIONS,csEE.size() + csPT.size());
        constraint->setMeta(GLOBAL_DCD_THICKNESS,imminent_collision_thickness);
    }

    if(type == "kinematic_dcd_collision_constraint") {
        constexpr auto eps = 1e-6;
        constexpr auto MAX_KINEMATIC_IMMINENT_COLLISION_PAIRS = 200000;
        auto dcd_source_xtag = get_input2<std::string>("dcd_source_xtag");
        const auto &edges = (*source)[ZenoParticles::s_surfEdgeTag];

        auto collider = get_input<ZenoParticles>("target");
        auto dcd_collider_xtag = get_input2<std::string>("dcd_collider_xtag");
        auto dcd_collider_pxtag = get_input2<std::string>("dcd_collider_pxtag");
        auto toc = get_input2<float>("toc");
        const auto& kverts = collider->getParticles();
        const auto& kedges = (*collider)[ZenoParticles::s_surfEdgeTag];
        const auto& ktris = collider->getQuadraturePoints();

        constraint->setMeta(CONSTRAINT_KEY,category_c::kinematic_dcd_collision_constraint);
        eles.append_channels(cudaPol,{{"inds",4},{"bary",4},{"type",1},{"hit_point",3},{"hit_velocity",3}});        

        auto imminent_collision_thickness = get_input2<float>("thickness");

        zs::bht<int,2,int> csPKT{verts.get_allocator(),(size_t)MAX_KINEMATIC_IMMINENT_COLLISION_PAIRS};csPKT.reset(cudaPol,true);
        zs::bht<int,2,int> csKPT{verts.get_allocator(),(size_t)MAX_KINEMATIC_IMMINENT_COLLISION_PAIRS};csKPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEKE{edges.get_allocator(),(size_t)MAX_KINEMATIC_IMMINENT_COLLISION_PAIRS};csEKE.reset(cudaPol,true);

        auto substep_id = get_input2<int>("substep_id");
        auto nm_substeps = get_input2<int>("nm_substeps");
        auto w = (float)(substep_id + 1) / (float)nm_substeps;
        auto pw = (float)(substep_id) / (float)nm_substeps;  

        dtiles_t kvtemp{kverts.get_allocator(),{
            {"x",3},
            {"v",3}
        },kverts.size()};

        cudaPol(zs::range(kverts.size()),[
            kxOffset = kverts.getPropertyOffset(dcd_collider_xtag),
            kpxOffset = kverts.getPropertyOffset(dcd_collider_pxtag),
            kverts = proxy<space>({},kverts),
            toc = toc,
            w = w,
            pw = pw,
            tmpXOffset = kvtemp.getPropertyOffset("x"),
            tmpVOffset = kvtemp.getPropertyOffset("v"),
            kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(int kvi) mutable {
                auto kpreVert = kverts.pack(dim_c<3>,kpxOffset,kvi) * (1 - pw) + kverts.pack(dim_c<3>,kxOffset,kvi) * pw;
                auto kcurVert = kverts.pack(dim_c<3>,kpxOffset,kvi) * (1 - w) + kverts.pack(dim_c<3>,kxOffset,kvi) * w;
                kpreVert = kpreVert * (static_cast<T>(1.0) - toc) + kcurVert * toc;
                auto kvel = kcurVert - kpreVert;
                
                kvtemp.tuple(dim_c<3>,tmpXOffset,kvi) = kpreVert;
                kvtemp.tuple(dim_c<3>,tmpVOffset,kvi) = kvel;
        });

        auto triBvh = bvh_t{};
        auto triBvs = retrieve_bounding_volumes(cudaPol,verts,quads,wrapv<3>{},imminent_collision_thickness/static_cast<float>(2.0),dcd_source_xtag);
        triBvh.build(cudaPol,triBvs);

        auto ktriBvh = bvh_t{};
        auto ktriBvs = retrieve_bounding_volumes(cudaPol,kvtemp,ktris,wrapv<3>{},imminent_collision_thickness/static_cast<float>(2.0),"x");
        ktriBvh.build(cudaPol,ktriBvs);

        auto kedgeBvh = bvh_t{};
        auto kedgeBvs = retrieve_bounding_volumes(cudaPol,kvtemp,kedges,wrapv<2>{},imminent_collision_thickness/static_cast<float>(2.0),"x");
        kedgeBvh.build(cudaPol,kedgeBvs);

        COLLISION_UTILS::detect_imminent_PKT_close_proximity(cudaPol,
            verts,dcd_source_xtag,
            kvtemp,"x",
            ktris,
            imminent_collision_thickness,
            ktriBvh,
            csPKT);

        COLLISION_UTILS::detect_imminent_PKT_close_proximity(cudaPol,
            kvtemp,"x",
            verts,dcd_source_xtag,
            quads,
            imminent_collision_thickness,
            triBvh,
            csKPT);

        COLLISION_UTILS::detect_imminent_EKE_close_proximity(cudaPol,
            verts,dcd_source_xtag,
            edges,
            kvtemp,"x",
            kedges,
            imminent_collision_thickness,
            kedgeBvh,
            csEKE);

        eles.resize(csPKT.size() + csKPT.size() + csEKE.size());

        cudaPol(zip(zs::range(csPKT.size()),csPKT._activeKeys),[
            indsOffset = eles.getPropertyOffset("inds"),
            baryOffset = eles.getPropertyOffset("bary"),
            typeOffset = eles.getPropertyOffset("type"),
            hitPointOffset = eles.getPropertyOffset("hit_point"),
            hitVelocityOffset = eles.getPropertyOffset("hit_velocity"),
            proximity_buffer = proxy<space>({},eles),
            xoffset = verts.getPropertyOffset(dcd_source_xtag),
            verts = proxy<space>({},verts),
            kxOffset = kvtemp.getPropertyOffset("x"),
            kvOffset = kvtemp.getPropertyOffset("v"),
            kvtemp = proxy<space>({},kvtemp),
            ktris = ktris.begin("inds",dim_c<3>,int_c)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto vi = pair[0];
                auto kti = pair[1];
                auto ktri = ktris[kti];

                auto p = verts.pack(dim_c<3>,xoffset,vi);
                vec3 kps[3] = {};
                vec3 kvs[3] = {};
                for(int i = 0;i != 3;++i) {
                    kps[i] = kvtemp.pack(dim_c<3>,kxOffset,ktri[i]);
                    kvs[i] = kvtemp.pack(dim_c<3>,kvOffset,ktri[i]);
                }

                vec3 tri_bary{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(kps[0],kps[1],kps[2],p,tri_bary);
                vec4 bary{-tri_bary[0],-tri_bary[1],-tri_bary[2],1};
                vec4i inds{ktri[0],ktri[1],ktri[2],vi};

                auto hit_point = vec3::zeros();
                auto hit_velocity = vec3::zeros();
                for(int i = 0;i != 3;++i) {
                    hit_point += kps[i] * tri_bary[i];
                    hit_velocity += kvs[i] * tri_bary[i];
                }

                proximity_buffer.tuple(dim_c<4>,indsOffset,id) = inds.reinterpret_bits(float_c);
                proximity_buffer.tuple(dim_c<4>,baryOffset,id) = bary;
                proximity_buffer(typeOffset,id) = zs::reinterpret_bits<float>((int)0);
                proximity_buffer.tuple(dim_c<3>,hitPointOffset,id) = hit_point;
                proximity_buffer.tuple(dim_c<3>,hitVelocityOffset,id) = hit_velocity;
        });

        cudaPol(zip(zs::range(csKPT.size()),csKPT._activeKeys),[
            buffer_offset = csPKT.size(),
            indsOffset = eles.getPropertyOffset("inds"),
            baryOffset = eles.getPropertyOffset("bary"),
            typeOffset = eles.getPropertyOffset("type"), 
            hitPointOffset = eles.getPropertyOffset("hit_point"),
            hitVelocityOffset = eles.getPropertyOffset("hit_velocity"),
            // hitNormalOffset = eles.getPropertyOffset("hit_normal"),
            proximity_buffer = proxy<space>({},eles),     
            xoffset = verts.getPropertyOffset(dcd_source_xtag),   
            verts = proxy<space>({},verts),
            tris = quads.begin("inds",dim_c<3>,int_c),
            kxoffset = kvtemp.getPropertyOffset("x"),
            kvoffset = kvtemp.getPropertyOffset("v"),
            kvtemp = proxy<space>({},kvtemp)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto kvi = pair[0];
                auto ti = pair[1];
                auto tri = tris[ti];

                auto kp = kvtemp.pack(dim_c<3>,kxoffset,kvi);
                auto kv = kvtemp.pack(dim_c<3>,kvoffset,kvi);

                vec3 ps[3] = {};
                for(int i = 0;i != 3;++i)
                    ps[i] = verts.pack(dim_c<3>,xoffset,tri[i]);

                vec3 tri_bary{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(ps[0],ps[1],ps[2],kp,tri_bary);
                vec4 bary{-tri_bary[0],-tri_bary[1],-tri_bary[2],1};
                vec4i inds{tri[0],tri[1],tri[2],kvi};

                proximity_buffer.tuple(dim_c<4>,indsOffset,id + buffer_offset) = inds.reinterpret_bits(float_c);
                proximity_buffer.tuple(dim_c<4>,baryOffset,id + buffer_offset) = bary;
                proximity_buffer(typeOffset,id + buffer_offset) = zs::reinterpret_bits<float>((int)1);
                proximity_buffer.tuple(dim_c<3>,hitPointOffset,id + buffer_offset) = kp;
                proximity_buffer.tuple(dim_c<3>,hitVelocityOffset,id + buffer_offset) = kv;
                // proximity_buffer.tuple(dim_c<3>,hitNormalOffset,id) = hit_normal;
        });


        cudaPol(zip(zs::range(csEKE.size()),csEKE._activeKeys),[
            eps = eps,
            buffer_offset = csPKT.size() + csKPT.size(),
            indsOffset = eles.getPropertyOffset("inds"),
            baryOffset = eles.getPropertyOffset("bary"),
            typeOffset = eles.getPropertyOffset("type"),
            hitPointOffset = eles.getPropertyOffset("hit_point"),
            hitVelocityOffset = eles.getPropertyOffset("hit_velocity"),
            proximity_buffer = proxy<space>({},eles),
            xoffset = verts.getPropertyOffset(dcd_source_xtag),
            verts = proxy<space>({},verts),
            edges = edges.begin("inds",dim_c<2>,int_c),
            kxoffset = kvtemp.getPropertyOffset("x"),
            kvoffset = kvtemp.getPropertyOffset("v"),
            kvtemp = proxy<space>({},kvtemp),
            kedges = kedges.begin("inds",dim_c<2>,int_c)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto ei = pair[0];
                auto kei = pair[1];

                auto edge = edges[ei];
                auto kedge = kedges[kei];

                vec4i inds{edge[0],edge[1],kedge[0],kedge[1]};

                vec3 ps[2] = {};
                for(int i = 0;i != 2;++i)
                    ps[i] = verts.pack(dim_c<3>,xoffset,edge[i]);

                vec3 kps[2] = {};
                vec3 kvs[2] = {};
                for(int i = 0;i != 2;++i) {
                    kps[i] = kvtemp.pack(dim_c<3>,kxoffset,kedge[i]);
                    kvs[i] = kvtemp.pack(dim_c<3>,kvoffset,kedge[i]);
                }

                auto hit_point = vec3::zeros();
                auto hit_velocity = vec3::zeros();

                vec2 edge_bary{};
                LSL_GEO::get_edge_edge_barycentric_coordinates(ps[0],ps[1],kps[0],kps[1],edge_bary);
                vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

                hit_point = bary[2] * kps[0] + bary[3] * kps[1];
                hit_velocity = bary[2] * kvs[0] + bary[3] * kvs[1];

                // auto hit_normal = bary[0] * ps[0] + bary[1] * ps[1] + bary[2] * kps[0] + bary[3] * kps[1];
                // if(hit_normal.norm() > eps)
                //     hit_normal = hit_normal.normalized();
                // else
                //     hit_normal = (ps[1] - ps[0]).cross(kps[1] - kps[0]).normalized();

                proximity_buffer.tuple(dim_c<4>,baryOffset,id + buffer_offset) = bary;
                proximity_buffer.tuple(dim_c<4>,indsOffset,id + buffer_offset) = inds.reinterpret_bits(float_c);
                proximity_buffer(typeOffset,id + buffer_offset) = zs::reinterpret_bits<float>((int)2);
                proximity_buffer.tuple(dim_c<3>,hitPointOffset,id + buffer_offset) = hit_point;
                proximity_buffer.tuple(dim_c<3>,hitVelocityOffset,id + buffer_offset) = hit_velocity;
                // proximity_buffer.tuple(dim_c<3>,hitNormalOffset,id) = hit_normal;
        });

        constraint->setMeta<size_t>(NM_DCD_COLLISIONS,csEKE.size() + csPKT.size() + csKPT.size());
        constraint->setMeta(GLOBAL_DCD_THICKNESS,imminent_collision_thickness);
        constraint->setMeta<bool>(ENABLE_DCD_REPULSION_FORCE,get_input2<bool>("add_dcd_repulsion_force"));
    }

    if(type == "self_dcd_collision_constraint") {
        constexpr auto eps = 1e-6;
        constexpr auto MAX_SELF_IMMINENT_COLLISION_PAIRS = 200000;
        auto dcd_source_xtag = get_input2<std::string>("dcd_source_xtag");
        constraint->setMeta(CONSTRAINT_KEY,category_c::self_dcd_collision_constraint);
        eles.append_channels(cudaPol,{{"inds",4},{"bary",4},{"type",1}});

        const auto &edges = (*source)[ZenoParticles::s_surfEdgeTag];

        auto imminent_collision_thickness = get_input2<float>("thickness");

        zs::bht<int,2,int> csPT{verts.get_allocator(),(size_t)MAX_SELF_IMMINENT_COLLISION_PAIRS};csPT.reset(cudaPol,true);
        zs::bht<int,2,int> csEE{edges.get_allocator(),(size_t)MAX_SELF_IMMINENT_COLLISION_PAIRS};csEE.reset(cudaPol,true);


        auto triBvh = bvh_t{};
        auto triBvs = retrieve_bounding_volumes(cudaPol,verts,quads,wrapv<3>{},imminent_collision_thickness/static_cast<float>(2.0),dcd_source_xtag);
        triBvh.build(cudaPol,triBvs);
        COLLISION_UTILS::detect_self_imminent_PT_close_proximity(cudaPol,verts,dcd_source_xtag,quads,imminent_collision_thickness,triBvh,csPT);

        std::cout << "nm_imminent_csPT : " << csPT.size() << std::endl;
        auto edgeBvh = bvh_t{};
        auto edgeBvs = retrieve_bounding_volumes(cudaPol,verts,edges,wrapv<2>{},imminent_collision_thickness/static_cast<float>(2.0),dcd_source_xtag);
        edgeBvh.build(cudaPol,edgeBvs);  
        COLLISION_UTILS::detect_self_imminent_EE_close_proximity(cudaPol,verts,dcd_source_xtag,edges,imminent_collision_thickness,edgeBvh,csEE);

        std::cout << "nm_imminent_csEE : " << csEE.size() << std::endl;

        eles.resize(csPT.size() + csEE.size());

        // initialize self imminent PT collision data
        cudaPol(zip(zs::range(csPT.size()),csPT._activeKeys),[
            indsOffset = eles.getPropertyOffset("inds"),
            baryOffset = eles.getPropertyOffset("bary"),
            typeOffset = eles.getPropertyOffset("type"),
            proximity_buffer = proxy<space>({},eles),
            xoffset = verts.getPropertyOffset(dcd_source_xtag),
            verts = proxy<space>({},verts),
            quads = quads.begin("inds",dim_c<3>,int_c)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto vi = pair[0];
                auto ti = pair[1];
                auto tri = quads[ti];

                auto p = verts.pack(dim_c<3>,xoffset,vi);
                vec3 ts[3] = {};

                for(int i = 0;i != 3;++i)
                    ts[i] = verts.pack(dim_c<3>,xoffset,tri[i]);

                vec3 tri_bary{};
                LSL_GEO::get_triangle_vertex_barycentric_coordinates(ts[0],ts[1],ts[2],p,tri_bary);

                vec4 bary{-tri_bary[0],-tri_bary[1],-tri_bary[2],1};
                vec4i inds{tri[0],tri[1],tri[2],vi};
                proximity_buffer.tuple(dim_c<4>,indsOffset,id) = inds.reinterpret_bits(float_c);
                proximity_buffer.tuple(dim_c<4>,baryOffset,id) = bary;
                proximity_buffer(typeOffset,id) = zs::reinterpret_bits<float>((int)0);                
        });

        cudaPol(zip(zs::range(csEE.size()),csEE._activeKeys),[
            buffer_offset = csPT.size(),
            indsOffset = eles.getPropertyOffset("inds"),
            baryOffset = eles.getPropertyOffset("bary"),
            typeOffset = eles.getPropertyOffset("type"),
            proximity_buffer = proxy<space>({},eles),
            xoffset = verts.getPropertyOffset(dcd_source_xtag),
            verts = proxy<space>({},verts),
            edges = edges.begin("inds",dim_c<2>,int_c)] ZS_LAMBDA(auto id,const auto& pair) mutable {
                auto ei = pair[0];
                auto nei = pair[1];

                auto ea = edges[ei];
                auto eb = edges[nei];
                vec4i inds{ea[0],ea[1],eb[0],eb[1]};

                vec3 ps[4] = {};
                for(int i = 0;i != 4;++i)
                    ps[i] = verts.pack(dim_c<3>,xoffset,inds[i]);

                vec2 edge_bary{};
                LSL_GEO::get_edge_edge_barycentric_coordinates(ps[0],ps[1],ps[2],ps[3],edge_bary);
                vec4 bary{edge_bary[0] - 1,-edge_bary[0],1 - edge_bary[1],edge_bary[1]};

                proximity_buffer.tuple(dim_c<4>,baryOffset,id + buffer_offset) = bary;
                proximity_buffer.tuple(dim_c<4>,indsOffset,id + buffer_offset) = inds.reinterpret_bits(float_c);
                proximity_buffer(typeOffset,id + buffer_offset) = zs::reinterpret_bits<float>((int)1);
        });

        constraint->setMeta<size_t>(NM_DCD_COLLISIONS,csEE.size() + csPT.size());
        constraint->setMeta(GLOBAL_DCD_THICKNESS,imminent_collision_thickness);
        constraint->setMeta<bool>(ENABLE_DCD_REPULSION_FORCE,get_input2<bool>("add_dcd_repulsion_force"));
    }

    if(type == "volume_pin") {
        constexpr auto eps = 1e-6;
        constraint->setMeta(CONSTRAINT_KEY,category_c::volume_pin_constraint);

        auto volume = get_input<ZenoParticles>("target");
        const auto& kverts = volume->getParticles();
        const auto& ktets = volume->getQuadraturePoints();

        constraint->setMeta(CONSTRAINT_TARGET,volume.get());

        auto pin_group_name = get_input2<std::string>("group_name");
        auto binder_max_length = get_input2<float>("thickness");

        zs::Vector<zs::vec<int,1>> point_topos{quads.get_allocator(),0};
        if(verts.hasProperty(pin_group_name)) {
            std::cout << "binder name : " << pin_group_name << std::endl;
            zs::bht<int,1,int> pin_point_set{verts.get_allocator(),verts.size()};
            pin_point_set.reset(cudaPol,true);

            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                eps = eps,
                gname = zs::SmallString(pin_group_name),
                pin_point_set = proxy<space>(pin_point_set)] ZS_LAMBDA(int vi) mutable {
                    auto gtag = verts(gname,vi);
                    if(gtag > eps)
                        pin_point_set.insert(vi);
            });
            point_topos.resize(pin_point_set.size());
            cudaPol(zip(zs::range(pin_point_set.size()),pin_point_set._activeKeys),[
                point_topos = proxy<space>(point_topos)] ZS_LAMBDA(auto id,const auto& pvec) mutable {
                    point_topos[id] = pvec[0];
            });
        }else {
            point_topos.resize(verts.size());
            cudaPol(zip(zs::range(point_topos.size()),point_topos),[] ZS_LAMBDA(const auto& id,auto& pi) mutable {pi = id;});
        }
        std::cout << "nm binder point : " << point_topos.size() << std::endl;
        if(do_constraint_topological_coloring) {
            topological_coloring(cudaPol,point_topos,colors,false);
            sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
        }
        
        eles.append_channels(cudaPol,{{"inds",2},{"bary",4}});
        eles.resize(point_topos.size());
        
        auto thickness = binder_max_length / (T)2.0;

        auto ktetBvh = bvh_t{};
        auto ktetBvs = retrieve_bounding_volumes(cudaPol,kverts,ktets,wrapv<4>{},thickness,"x");
        ktetBvh.build(cudaPol,ktetBvs);
        
        cudaPol(zs::range(point_topos.size()),[
            point_topos = proxy<space>(point_topos),
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map),
            verts = proxy<space>({},verts),
            ktetBvh = proxy<space>(ktetBvh),
            thickness = thickness,
            eles = proxy<space>({},eles),
            kverts = proxy<space>({},kverts),
            ktets = proxy<space>({},ktets)] ZS_LAMBDA(int oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                auto pi = point_topos[ei][0];
                auto p = verts.pack(dim_c<3>,"x",pi);
                auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};

                bool found = false;
                int embed_kti = -1;
                vec4 bary{};
                auto find_embeded_tet = [&](int kti) {
                    if(found)
                        return;
                    auto inds = ktets.pack(dim_c<4>,"inds",kti,int_c);
                    vec3 ktps[4] = {};
                    for(int i = 0;i != 4;++i)
                        ktps[i] = kverts.pack(dim_c<3>,"x",inds[i]);
                    auto ws = compute_vertex_tetrahedron_barycentric_weights(p,ktps[0],ktps[1],ktps[2],ktps[3]);

                    T epsilon = zs::limits<float>::epsilon();
                    if(ws[0] > epsilon && ws[1] > epsilon && ws[2] > epsilon && ws[3] > epsilon){
                        embed_kti = kti;
                        bary = ws;
                        found = true;
                        return;
                    }                        
                };  
                ktetBvh.iter_neighbors(bv,find_embeded_tet);
                if(embed_kti >= 0)
                    verts("minv",pi) = 0;
                eles.tuple(dim_c<2>,"inds",oei) = vec2i{pi,embed_kti}.reinterpret_bits(float_c);
                eles.tuple(dim_c<4>,"bary",oei) = bary;
        });
    }

    if(type == "point_triangle_pin") {
        constexpr auto eps = 1e-6;
        constraint->setMeta(CONSTRAINT_KEY,category_c::pt_pin_constraint);

        auto target = get_input<ZenoParticles>("target");
        const auto& kverts = target->getParticles();
        const auto& ktris = target->getQuadraturePoints();

        constraint->setMeta(CONSTRAINT_TARGET,target.get());

        auto pin_point_group_name = get_input2<std::string>("group_name");
        auto binder_max_length = get_input2<float>("thickness");
        // we might further need a pin_triangle_group_name
        zs::bht<int,1,int> pin_point_set{verts.get_allocator(),verts.size()};
        pin_point_set.reset(cudaPol,true);

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            eps = eps,
            gname = zs::SmallString(pin_point_group_name),
            pin_point_set = proxy<space>(pin_point_set)] ZS_LAMBDA(int vi) mutable {
                auto gtag = verts(gname,vi);
                if(gtag > eps)
                    pin_point_set.insert(vi);
        });
        zs::Vector<zs::vec<int,1>> point_topos{quads.get_allocator(),pin_point_set.size()};
        cudaPol(zip(zs::range(pin_point_set.size()),pin_point_set._activeKeys),[
            point_topos = proxy<space>(point_topos)] ZS_LAMBDA(auto id,const auto& pvec) mutable {
                point_topos[id] = pvec[0];
        });

        std::cout << "binder name : " << pin_point_group_name << std::endl;
        std::cout << "nm binder point : " << point_topos.size() << std::endl;

        if(do_constraint_topological_coloring) {
            topological_coloring(cudaPol,point_topos,colors,false);
            sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
        }

        eles.append_channels(cudaPol,{{"inds",2},{"bary",3},{"rd",1}});
        eles.resize(point_topos.size());  

        auto ktriBvh = bvh_t{};
        auto thickness = binder_max_length / (T)2.0;

        auto ktriBvs = retrieve_bounding_volumes(cudaPol,kverts,ktris,wrapv<3>{},thickness,"x");
        ktriBvh.build(cudaPol,ktriBvs);

        cudaPol(zs::range(point_topos.size()),[
            verts = proxy<space>({},verts),
            point_topos = proxy<space>(point_topos),
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            thickness = thickness,
            eles = proxy<space>({},eles),
            ktriBvh = proxy<space>(ktriBvh),
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map),
            quads = proxy<space>({},quads)] ZS_LAMBDA(auto oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                auto pi = point_topos[ei][0];
                auto p = verts.pack(dim_c<3>,"x",pi);
                auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};

                int min_kti = -1;
                T min_dist = std::numeric_limits<float>::max();
                vec3 min_bary_centric{};
                auto find_closest_triangles = [&](int kti) {
                    // printf("check binder pair[%d %d]\n",pi,kti);
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);
                    vec3 ts[3] = {};
                    for(int i = 0;i != 3;++i)
                        ts[i] = kverts.pack(dim_c<3>,"x",ktri[i]);

                    vec3 bary_centric{};
                    auto pt_dist = LSL_GEO::get_vertex_triangle_distance(ts[0],ts[1],ts[2],p,bary_centric);  
                    for(int i = 0;i != 3;++i)
                        bary_centric[i] = bary_centric[i] < 0 ? 0 : bary_centric[i];
                    // if(pt_dist > thickness * 2)
                    //     return;
                    
                    auto bary_sum = zs::abs(bary_centric[0]) + zs::abs(bary_centric[1]) + zs::abs(bary_centric[2]);
                    bary_centric /= bary_sum;
                    // if(bary_sum > 1.0 + eps * 100)
                    //     return;

                    if(pt_dist < min_dist) {
                        min_dist = pt_dist;
                        min_kti = kti;
                        min_bary_centric = bary_centric;
                    }
                };
                ktriBvh.iter_neighbors(bv,find_closest_triangles);

                if(min_kti >= 0) {
                    auto ktri = ktris.pack(dim_c<3>,"inds",min_kti,int_c);
                    vec3 kps[3] = {};
                    for(int i = 0;i != 3;++i)
                        kps[i] = kverts.pack(dim_c<3>,"x",ktri[i]);
                    auto knrm = LSL_GEO::facet_normal(kps[0],kps[1],kps[2]);
                    auto seg = p - kps[0];
                    if(seg.dot(knrm) < 0)
                        min_dist *= -1;
                    verts("minv",pi) = 0;
                }

                eles.tuple(dim_c<2>,"inds",oei) = vec2i{pi,min_kti}.reinterpret_bits(float_c);
                eles.tuple(dim_c<3>,"bary",oei) = min_bary_centric;
                eles("rd",oei) = min_dist;
        });
    }

    // angle on (p2, p3) between triangles (p0, p2, p3) and (p1, p3, p2)
    if(type == "bending") {
        constraint->setMeta(CONSTRAINT_KEY,category_c::isometric_bending_constraint);
        // constraint->category = ZenoParticles::tri_bending_spring;
        // constraint->sprayedOffset = 0;

        const auto& halfedges = (*source)[ZenoParticles::s_surfHalfEdgeTag];

        zs::Vector<zs::vec<int,4>> bd_topos{quads.get_allocator(),0};
        retrieve_tri_bending_topology(cudaPol,quads,halfedges,bd_topos);

        eles.resize(bd_topos.size());
        if(do_constraint_topological_coloring) {
            topological_coloring(cudaPol,bd_topos,colors);
            sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
        }
        // std::cout << "quads.size() = " << quads.size() << "\t" << "edge_topos.size() = " << edge_topos.size() << std::endl;

        eles.append_channels(cudaPol,{{"inds",4},{"Q",4 * 4},{"C0",1}});

        // std::cout << "halfedges.size() = " << halfedges.size() << "\t" << "bd_topos.size() = " << bd_topos.size() << std::endl;

        cudaPol(zs::range(eles.size()),[
            eles = proxy<space>({},eles),
            bd_topos = proxy<space>(bd_topos),
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map),
            verts = proxy<space>({},verts)] ZS_LAMBDA(int oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                // printf("bd_topos[%d] : %d %d %d %d\n",ei,bd_topos[ei][0],bd_topos[ei][1],bd_topos[ei][2],bd_topos[ei][3]);
                eles.tuple(dim_c<4>,"inds",oei) = bd_topos[ei].reinterpret_bits(float_c);
                vec3 x[4] = {};
                for(int i = 0;i != 4;++i)
                    x[i] = verts.pack(dim_c<3>,"x",bd_topos[ei][i]);

                mat4 Q = mat4::uniform(0);
                float C0{};
                CONSTRAINT::init_IsometricBendingConstraint(x[0],x[1],x[2],x[3],Q,C0);
                eles.tuple(dim_c<16>,"Q",oei) = Q;
                eles("C0",oei) = C0;
        });
    }
    // angle on (p2, p3) between triangles (p0, p2, p3) and (p1, p3, p2)
    if(type == "dihedral") {
        constraint->setMeta(CONSTRAINT_KEY,category_c::dihedral_bending_constraint);

        const auto& halfedges = (*source)[ZenoParticles::s_surfHalfEdgeTag];

        zs::Vector<zs::vec<int,4>> bd_topos{quads.get_allocator(),0};
        retrieve_tri_bending_topology(cudaPol,quads,halfedges,bd_topos);

        auto rest_scale = get_input2<float>("rest_scale");

        eles.resize(bd_topos.size());

        if(do_constraint_topological_coloring) {
            topological_coloring(cudaPol,bd_topos,colors);
            sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
        }
        // std::cout << "quads.size() = " << quads.size() << "\t" << "edge_topos.size() = " << edge_topos.size() << std::endl;

        eles.append_channels(cudaPol,{{"inds",4},{"ra",1},{"sign",1}});      

        cudaPol(zs::range(eles.size()),[
            eles = proxy<space>({},eles),
            bd_topos = proxy<space>(bd_topos),
            rest_scale = rest_scale,
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map),
            verts = proxy<space>({},verts)] ZS_LAMBDA(int oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                eles.tuple(dim_c<4>,"inds",oei) = bd_topos[ei].reinterpret_bits(float_c);

                vec3 x[4] = {};
                for(int i = 0;i != 4;++i)
                    x[i] = verts.pack(dim_c<3>,"x",bd_topos[ei][i]);

                float alpha{};
                float alpha_sign{};
                CONSTRAINT::init_DihedralBendingConstraint(x[0],x[1],x[2],x[3],rest_scale,alpha,alpha_sign);
                eles("ra",oei) = alpha;
                eles("sign",oei) = alpha_sign;
        });      
    }

    if(type == "dihedral_spring") {
        constraint->setMeta(CONSTRAINT_KEY,category_c::dihedral_spring_constraint);
        const auto& halfedges = (*source)[ZenoParticles::s_surfHalfEdgeTag];
        zs::Vector<zs::vec<int,2>> ds_topos{quads.get_allocator(),0};

        retrieve_dihedral_spring_topology(cudaPol,quads,halfedges,ds_topos);

        if(do_constraint_topological_coloring) {
            topological_coloring(cudaPol,ds_topos,colors);
            sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
        }

        eles.resize(ds_topos.size());
        eles.append_channels(cudaPol,{{"inds",2},{"r",1}}); 

        cudaPol(zs::range(eles.size()),[
            verts = proxy<space>({},verts),
            eles = proxy<space>({},eles),
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map),
            // uniform_stiffness = uniform_stiffness,
            colors = proxy<space>(colors),
            edge_topos = proxy<space>(ds_topos)] ZS_LAMBDA(int oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                eles.tuple(dim_c<2>,"inds",oei) = edge_topos[ei].reinterpret_bits(float_c);
                vec3 x[2] = {};
                for(int i = 0;i != 2;++i)
                    x[i] = verts.pack(dim_c<3>,"x",edge_topos[ei][i]);
                eles("r",oei) = (x[0] - x[1]).norm();
        }); 
    }

    if(type != "self_dcd_collision_constraint" && type != "kinematic_dcd_collison_constraint") {
        cudaPol(zs::range(eles.size()),[
            eles = proxy<space>({},eles),
            relative_stiffness = relative_stiffness,
            xpbd_affiliation = uniform_xpbd_affiliation,
            damping_coeff = damping_coeff,
            colors = proxy<space>(colors),
            do_constraint_topological_coloring = do_constraint_topological_coloring,
            reordered_map = proxy<space>(reordered_map)] ZS_LAMBDA(int oei) mutable {
                auto ei = do_constraint_topological_coloring ? reordered_map[oei] : oei;
                if(do_constraint_topological_coloring)
                    eles("tclr",oei) = colors[ei];
                eles("lambda",oei) = 0.0;
                eles("relative_stiffness",oei) = relative_stiffness;
                eles("xpbd_affiliation",oei) = xpbd_affiliation;
                eles("damping_coeff",oei) = damping_coeff;
        });

        constraint->setMeta(CONSTRAINT_COLOR_OFFSET,color_offset);
    }

    }else {
        constraint->setMeta(CONSTRAINT_KEY,category_c::empty_constraint);
    }

    // set_output("source",source);
    set_output("constraint",constraint);
}
};

ZENDEFNODE(MakeSurfaceConstraintTopology, {{
                            {"source"},
                            {"target"},
                            {"string","dcd_source_xtag","px"},
                            {"string","dcd_collider_xtag","x"},
                            {"string","dcd_collider_pxtag","px"},
                            {"float","toc","0"},
                            {"bool","add_dcd_repulsion_force","1"},
                            {"float","relative_stiffness","1.0"},
                            {"float","xpbd_affiliation","1.0"},
                            {"string","topo_type","stretch"},
                            {"float","rest_scale","1.0"},
                            {"string","group_name","groupName"},
                            {"float","thickness","0.1"},
                            {"int","substep_id","0"},
                            {"int","nm_substeps","1"},
                            {"bool","make_empty_constraint","0"},
                            {"bool","do_constraint_topological_coloring","1"},
                            {"float","damping_coeff","0.0"}
                        },
                        {{"constraint"}},
                        { 
                            // {"string","groupID",""},
                        },
                        {"PBD"}});


};