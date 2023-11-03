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

        using vec3 = zs::vec<float,3>;
        using vec4 = zs::vec<float,4>;
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

        auto uniform_stiffness = get_input2<float>("stiffness");

        auto make_empty = get_input2<bool>("make_empty_constraint");

        if(!make_empty) {

        zs::Vector<float> colors{quads.get_allocator(),0};
        zs::Vector<int> reordered_map{quads.get_allocator(),0};
        zs::Vector<int> color_offset{quads.get_allocator(),0};

        constraint->sprayedOffset = 0;
        constraint->elements = typename ZenoParticles::particles_t({{"stiffness",1},{"lambda",1},{"tclr",1}}, 0, zs::memsrc_e::device,0);
        auto &eles = constraint->getQuadraturePoints();
        constraint->setMeta(CONSTRAINT_TARGET,source.get());

        if(type == "stretch") {
            constraint->setMeta(CONSTRAINT_KEY,category_c::edge_length_constraint);
            auto quads_vec = tilevec_topo_to_zsvec_topo(cudaPol,quads,wrapv<3>{});
            zs::Vector<zs::vec<int,2>> edge_topos{quads.get_allocator(),0};
            retrieve_edges_topology(cudaPol,quads_vec,edge_topos);
            eles.resize(edge_topos.size());

            topological_coloring(cudaPol,edge_topos,colors);
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
            // std::cout << "quads.size() = " << quads.size() << "\t" << "edge_topos.size() = " << edge_topos.size() << std::endl;
            eles.append_channels(cudaPol,{{"inds",2},{"r",1}});

            auto rest_scale = get_input2<float>("rest_scale");

            cudaPol(zs::range(eles.size()),[
                verts = proxy<space>({},verts),
                eles = proxy<space>({},eles),
                reordered_map = proxy<space>(reordered_map),
                uniform_stiffness = uniform_stiffness,
                colors = proxy<space>(colors),
                rest_scale = rest_scale,
                edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
                    eles.tuple(dim_c<2>,"inds",oei) = edge_topos[ei].reinterpret_bits(float_c);
                    vec3 x[2] = {};
                    for(int i = 0;i != 2;++i)
                        x[i] = verts.pack(dim_c<3>,"x",edge_topos[ei][i]);
                    eles("r",oei) = (x[0] - x[1]).norm() * rest_scale;
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
            topological_coloring(cudaPol,point_topos,colors,false);
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);

            eles.resize(verts.size());
            // we need an extra 'inds' tag, in case the source and animation has different topo
            eles.append_channels(cudaPol,{{"inds",1},{"follow_weight",1}});
            // cudaPol(zs::range(eles.size()),[
            //     eles = proxy<space>({},eles)] ZS_LAMBDA(int ei) mutable {eles("inds",1) = zs::reinterpret_bits<float>(ei);});
            
            // TILEVEC_OPS::copy(cudaPol,kverts,"ani_mask",eles,"follow_weight");
            cudaPol(zs::range(eles.size()),[
                kverts = proxy<space>({},kverts),
                reordered_map = proxy<space>(reordered_map),
                point_topos = proxy<space>(point_topos),
                eles = proxy<space>({},eles)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
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

        if(type == "dcd_collision_constraint") {
            constexpr auto eps = 1e-6;
            constexpr auto MAX_IMMINENT_COLLISION_PAIRS = 2000000;
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

                // if(!source->hasAuxData(PREVIOUS_COLLISION_TARGET)) {
                //     (*source)[PREVIOUS_COLLISION_TARGET] = dtiles_t{kverts.get_allocator(),{{"x",3}},kverts.size()};
                //     auto& kverts_pre = (*source)[PREVIOUS_COLLISION_TARGET];
                //     cudaPol(zs::range(kverts.size()),[
                //         kverts = proxy<space>({},kverts),
                //         kverts_pre = proxy<space>({},kverts_pre),
                //         pw = pw] ZS_LAMBDA(int kvi) mutable {
                //             auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1-pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                //             kverts_pre.tuple(dim_c<3>,"x",kvi) = pre_kvert;
                //     });            
                // }
            }

            dtiles_t vtemp{verts.get_allocator(),{
                {"x",3},
                {"X",3},
                {"minv",1},
                {"dcd_collision_tag",1},
                {"collision_cancel",1}
            },nm_verts};
            TILEVEC_OPS::fill(cudaPol,vtemp,"dcd_collision_tag",0);
            TILEVEC_OPS::copy<3>(cudaPol,verts,"px",vtemp,"x");
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

                // auto& kverts_pre = (*source)[PREVIOUS_COLLISION_TARGET];

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
            COLLISION_UTILS::detect_imminent_PT_close_proximity(cudaPol,vtemp,"x",ttemp,imminent_collision_thickness,0,triBvh,eles,csPT);
    
            std::cout << "nm_imminent_csPT : " << csPT.size() << std::endl;

            auto edgeBvh = bvh_t{};
            auto edgeBvs = retrieve_bounding_volumes(cudaPol,vtemp,etemp,wrapv<2>{},imminent_collision_thickness/(float)2.0,"x");
            edgeBvh.build(cudaPol,edgeBvs);  
            COLLISION_UTILS::detect_imminent_EE_close_proximity(cudaPol,vtemp,"x",etemp,imminent_collision_thickness,csPT.size(),edgeBvh,eles,csEE);

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
            topological_coloring(cudaPol,point_topos,colors,false);
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
            
            eles.append_channels(cudaPol,{{"inds",2},{"bary",4}});
            eles.resize(point_topos.size());
            
            auto thickness = binder_max_length / (T)2.0;

            auto ktetBvh = bvh_t{};
            auto ktetBvs = retrieve_bounding_volumes(cudaPol,kverts,ktets,wrapv<4>{},thickness,"x");
            ktetBvh.build(cudaPol,ktetBvs);
            
            cudaPol(zs::range(point_topos.size()),[
                point_topos = proxy<space>(point_topos),
                reordered_map = proxy<space>(reordered_map),
                verts = proxy<space>({},verts),
                ktetBvh = proxy<space>(ktetBvh),
                thickness = thickness,
                eles = proxy<space>({},eles),
                kverts = proxy<space>({},kverts),
                ktets = proxy<space>({},ktets)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
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
                        auto ws = compute_barycentric_weights(p,ktps[0],ktps[1],ktps[2],ktps[3]);

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
            topological_coloring(cudaPol,point_topos,colors,false);
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);

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
                reordered_map = proxy<space>(reordered_map),
                quads = proxy<space>({},quads)] ZS_LAMBDA(auto oei) mutable {
                    auto ei = reordered_map[oei];
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
                        auto pt_dist = LSL_GEO::pointTriangleDistance(ts[0],ts[1],ts[2],p,bary_centric);  
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

            topological_coloring(cudaPol,bd_topos,colors);
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
            // std::cout << "quads.size() = " << quads.size() << "\t" << "edge_topos.size() = " << edge_topos.size() << std::endl;

            eles.append_channels(cudaPol,{{"inds",4},{"Q",4 * 4},{"C0",1}});

            // std::cout << "halfedges.size() = " << halfedges.size() << "\t" << "bd_topos.size() = " << bd_topos.size() << std::endl;

            cudaPol(zs::range(eles.size()),[
                eles = proxy<space>({},eles),
                bd_topos = proxy<space>(bd_topos),
                reordered_map = proxy<space>(reordered_map),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
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

            eles.resize(bd_topos.size());

            topological_coloring(cudaPol,bd_topos,colors);
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);
            // std::cout << "quads.size() = " << quads.size() << "\t" << "edge_topos.size() = " << edge_topos.size() << std::endl;

            eles.append_channels(cudaPol,{{"inds",4},{"ra",1},{"sign",1}});      

            cudaPol(zs::range(eles.size()),[
                eles = proxy<space>({},eles),
                bd_topos = proxy<space>(bd_topos),
                reordered_map = proxy<space>(reordered_map),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
                    eles.tuple(dim_c<4>,"inds",oei) = bd_topos[ei].reinterpret_bits(float_c);

                    // printf("topos[%d] : %d %d %d %d\n",oei
                        // ,bd_topos[ei][0]
                        // ,bd_topos[ei][1]
                        // ,bd_topos[ei][2]
                        // ,bd_topos[ei][3]);

                    vec3 x[4] = {};
                    for(int i = 0;i != 4;++i)
                        x[i] = verts.pack(dim_c<3>,"x",bd_topos[ei][i]);

                    float alpha{};
                    float alpha_sign{};
                    CONSTRAINT::init_DihedralBendingConstraint(x[0],x[1],x[2],x[3],alpha,alpha_sign);
                    eles("ra",oei) = alpha;
                    eles("sign",oei) = alpha_sign;
            });      
        }

        if(type == "dihedral_spring") {
            constraint->setMeta(CONSTRAINT_KEY,category_c::dihedral_spring_constraint);
            const auto& halfedges = (*source)[ZenoParticles::s_surfHalfEdgeTag];
            zs::Vector<zs::vec<int,2>> ds_topos{quads.get_allocator(),0};

            retrieve_dihedral_spring_topology(cudaPol,quads,halfedges,ds_topos);

            topological_coloring(cudaPol,ds_topos,colors);
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);

            eles.resize(ds_topos.size());
            eles.append_channels(cudaPol,{{"inds",2},{"r",1}}); 

            cudaPol(zs::range(eles.size()),[
                verts = proxy<space>({},verts),
                eles = proxy<space>({},eles),
                reordered_map = proxy<space>(reordered_map),
                uniform_stiffness = uniform_stiffness,
                colors = proxy<space>(colors),
                edge_topos = proxy<space>(ds_topos)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
                    eles.tuple(dim_c<2>,"inds",oei) = edge_topos[ei].reinterpret_bits(float_c);
                    vec3 x[2] = {};
                    for(int i = 0;i != 2;++i)
                        x[i] = verts.pack(dim_c<3>,"x",edge_topos[ei][i]);
                    eles("r",oei) = (x[0] - x[1]).norm();
            }); 
        }

        if(type != "dcd_collision_constraint") {
            cudaPol(zs::range(eles.size()),[
                eles = proxy<space>({},eles),
                uniform_stiffness = uniform_stiffness,
                colors = proxy<space>(colors),
                // exec_tag,
                reordered_map = proxy<space>(reordered_map)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
                    eles("lambda",oei) = 0.0;
                    eles("stiffness",oei) = uniform_stiffness;
                    eles("tclr",oei) = colors[ei];
                    // auto 
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
                                {"float","stiffness","0.5"},
                                {"string","topo_type","stretch"},
                                {"float","rest_scale","1.0"},
                                {"string","group_name","groupName"},
                                {"float","thickness","0.1"},
                                {"int","substep_id","0"},
                                {"int","nm_substeps","1"},
                                {"bool","make_empty_constraint","0"}
                            },
							{{"constraint"}},
							{ 
                                // {"string","groupID",""},
                            },
							{"PBD"}});


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
        auto ptag = get_param<std::string>("ptag");

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
                kverts = proxy<space>({},kverts),
                kcells = proxy<space>({},kcells),
                cquads = proxy<space>({},cquads)] ZS_LAMBDA(int gi) mutable {
                    float s = cquads("stiffness",coffset + gi);
                    float lambda = cquads("lambda",coffset + gi);

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
                        CONSTRAINT::solve_DistanceConstraint(
                            p,1.f,
                            tp,(float)kminv * 10.f,
                            0.f,
                            s,
                            // dt,
                            // lambda,
                            dp,dkp);
                        // should we update kp here?
                        // use original pbd
                        // lambda = ori_lambda;
                        // printf("solve following animation constraint[%d] : %f %f %f\n",
                        //     vi,(float)dp[0],(float)dp[1],(float)dp[2]);
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
                        auto edge = cquads.pack(dim_c<2>,"inds",coffset + gi,int_c);
                        vec3 p0{},p1{};
                        p0 = verts.pack(dim_c<3>,ptag,edge[0]);
                        p1 = verts.pack(dim_c<3>,ptag,edge[1]);
                        float minv0 = verts("minv",edge[0]);
                        float minv1 = verts("minv",edge[1]);
                        float r = cquads("r",coffset + gi);

                        vec3 dp0{},dp1{};
                        if(CONSTRAINT::solve_DistanceConstraint(
                            p0,minv0,
                            p1,minv1,
                            r,
                            s,
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
                            s,
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
                        float minv[4] = {};
                        for(int i = 0;i != 4;++i) {
                            p[i] = verts.pack(dim_c<3>,ptag,quad[i]);
                            minv[i] = verts("minv",quad[i]);
                        }

                        auto ra = cquads("ra",coffset + gi);
                        auto ras = cquads("sign",coffset + gi);
                        vec3 dp[4] = {};
                        if(!CONSTRAINT::solve_DihedralConstraint(
                            p[0],minv[0],
                            p[1],minv[1],
                            p[2],minv[2],
                            p[3],minv[3],
                            ra,
                            ras,
                            s,
                            dt,
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
        // set_output("target",target);
    };
};

ZENDEFNODE(XPBDSolve, {{{"zsparticles"},
                            {"constraints"},
                            {"int","substep_id","0"},
                            {"int","nm_substeps","1"},
                            // {"bool","make_empty"},
                            // {"string","kptag","x"},
                            {"float","dt","0.5"}},
							{{"zsparticles"},{"constraints"}},
							{{"string","ptag","X"}},
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
            // std::cout << "nm_DCD_COLLISIONS PROXY : " << nm_dcd_collisions << std::endl;

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
                {"minv",1}
            },nm_verts};

            TILEVEC_OPS::copy<3>(cudaPol,verts,"px",vtemp,"x");
            TILEVEC_OPS::copy(cudaPol,verts,"minv",vtemp,"minv");
            cudaPol(zs::range(verts.size()),[
                vtemp = proxy<space>({},vtemp),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                    vtemp.tuple(dim_c<3>,"v",vi) = verts.pack(dim_c<3>,"x",vi) - verts.pack(dim_c<3>,"px",vi);
            });  

            if(has_input_collider) {
                auto collider = constraints->readMeta(CONSTRAINT_TARGET,zs::wrapt<ZenoParticles*>{});
                const auto& kverts = collider->getParticles();
                const auto& kedges = (*collider)[ZenoParticles::s_surfEdgeTag];
                const auto& ktris = collider->getQuadraturePoints();  

                // auto& kverts_pre = (*zsparticles)[PREVIOUS_COLLISION_TARGET];

                auto voffset = verts.size();
                cudaPol(zs::range(kverts.size()),[
                    // kverts = proxy<space>({},kverts),kverts_pre = proxy<space>({},kverts_pre),
                    kverts = proxy<space>({},kverts),
                    voffset = voffset,
                    pw = pw,
                    w = w,
                    nm_substeps = nm_substeps,
                    vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int kvi) mutable {
                        auto pre_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - pw) + kverts.pack(dim_c<3>,"x",kvi) * pw;
                        auto cur_kvert = kverts.pack(dim_c<3>,"px",kvi) * (1 - w) + kverts.pack(dim_c<3>,"x",kvi) * w;
                        vtemp.tuple(dim_c<3>,"x",voffset + kvi) = pre_kvert;
                        vtemp("minv",voffset + kvi) = 0;  
                        vtemp.tuple(dim_c<3>,"v",voffset + kvi) = cur_kvert - pre_kvert;
                });            
            }

            // std::cout << "nm_dcd_collision : " << nm_dcd_collisions << "\t" << cquads.size() << "\t" << dp_buffer.size() << "\t" << dp_count.size() << "\t" << vtemp.size() <<  std::endl;

            auto add_repulsion_force = get_input2<bool>("add_repulsion_force");

            for(auto iter = 0;iter != nm_smooth_iters;++iter) {
                cudaPol(zs::range(verts.size()),[
                    dp_buffer = proxy<space>(dp_buffer),
                    dp_count = proxy<space>(dp_count)] ZS_LAMBDA(int vi) mutable {
                        for(int d = 0;d != 3;++d)
                            dp_buffer[vi * 3 + d] = 0;
                        // dp_buffer[vi] = vec3::zeros();
                        dp_count[vi] = 0;
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

                        vec3 ps[4] = {};
                        vec3 vs[4] = {};
                        vec4 minvs{};

                        for(int i = 0;i != 4;++i) {
                            ps[i] = vtemp.pack(dim_c<3>,"x",inds[i]);
                            vs[i] = vtemp.pack(dim_c<3>,"v",inds[i]);
                            minvs[i] = vtemp("minv",inds[i]);
                        }

                        vec3 imps[4] = {};
                        if(!COLLISION_UTILS::compute_imminent_collision_impulse(ps,vs,bary,minvs,imps,imminent_thickness,add_repulsion_force))
                            return;
                        for(int i = 0;i != 4;++i) {
                            if(minvs[i] < eps)
                                continue;

                            if(isnan(imps[i].norm())) {
                                printf("nan imps detected : %f %f %f %f %f %f %f\n",
                                    (float)imps[i][0],(float)imps[i][1],(float)imps[i][2],
                                    (float)bary[0],(float)bary[1],(float)bary[2],(float)bary[3]);
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

        // set_output("all_constraints",all_constraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolveSmooth, {{{"zsparticles"},
                                {"constraints"},
                                {"float","relaxation_strength","1"},
                                {"int","nm_smooth_iters","1"},
                                {"int","nm_substeps","1"},
                                {"int","substep_id","0"},
                                {"bool","add_repulsion_force","0"}
                            }, 
							{{"zsparticles"}},
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
