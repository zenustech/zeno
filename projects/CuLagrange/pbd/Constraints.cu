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
// #include "../fem/collision_energy/evaluate_collision.hpp"
#include "constraint_function_kernel/constraint_types.hpp"

namespace zeno {


// we only need to record the topo here
// serve triangulate mesh or strands only currently
struct MakeSurfaceConstraintTopology : INode {
    template <typename TileVecT>
    void buildBvh(zs::CudaExecutionPolicy &pol, 
            TileVecT &verts, 
            const zs::SmallString& srcTag,
            const zs::SmallString& dstTag,
            const zs::SmallString& pscaleTag,
                  ZenoLinearBvh::lbvh_t &bvh) {
        using namespace zs;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
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

        const auto& verts = source->getParticles();
        const auto& quads = source->getQuadraturePoints();

        auto uniform_stiffness = get_input2<float>("stiffness");

        zs::Vector<float> colors{quads.get_allocator(),0};
        zs::Vector<int> reordered_map{quads.get_allocator(),0};
        zs::Vector<int> color_offset{quads.get_allocator(),0};

        constraint->sprayedOffset = 0;
        constraint->elements = typename ZenoParticles::particles_t({{"stiffness",1},{"lambda",1},{"tclr",1}}, 0, zs::memsrc_e::device,0);
        auto &eles = constraint->getQuadraturePoints();

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

            eles.append_channels(cudaPol,{{"inds",4},{"ra",1}});      

            cudaPol(zs::range(eles.size()),[
                eles = proxy<space>({},eles),
                bd_topos = proxy<space>(bd_topos),
                reordered_map = proxy<space>(reordered_map),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int oei) mutable {
                    auto ei = reordered_map[oei];
                    eles.tuple(dim_c<4>,"inds",oei) = bd_topos[ei].reinterpret_bits(float_c);

                    vec3 x[4] = {};
                    for(int i = 0;i != 4;++i)
                        x[i] = verts.pack(dim_c<3>,"x",bd_topos[ei][i]);

                    float alpha{};
                    CONSTRAINT::init_DihedralBendingConstraint(x[0],x[1],x[2],x[3],alpha);
                    eles("ra",oei) = alpha;
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

        if(type == "kcollision") {
            using bv_t = typename ZenoLinearBvh::lbvh_t::Box;

            constraint->setMeta(CONSTRAINT_KEY,category_c::p_kp_collision_constraint);
            auto target = get_input<ZenoParticles>("target");

            const auto& kverts = target->getParticles();
            ZenoLinearBvh::lbvh_t kbvh{};
            buildBvh(cudaPol,kverts,"px","x","pscale",kbvh);

            zs::bht<int,2,int> csPP{verts.get_allocator(),verts.size()};
            csPP.reset(cudaPol,true);

            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                kverts = proxy<space>({},kverts),
                kbvh = proxy<space>(kbvh),
                csPP = proxy<space>(csPP)] ZS_LAMBDA(int vi) mutable {
                    auto x = verts.pack(dim_c<3>,"x",vi);
                    auto px = verts.pack(dim_c<3>,"px",vi);
                    auto mx = (x + px) / (T)2.0;
                    auto pscale = verts("pscale",vi);

                    auto radius = (mx - px).norm() + pscale * (T)2.0;
                    auto bv = bv_t{mx - radius,mx + radius};

                    int contact_kvi = -1;
                    T min_contact_time = std::numeric_limits<T>::max();

                    auto process_ccd_collision = [&](int kvi) {
                        auto kpscale = kverts("pscale",kvi);
                        auto kx = kverts.pack(dim_c<3>,"x",kvi);
                        auto pkx = kx;
                        if(kverts.hasProperty("px"))
                            pkx = kverts.pack(dim_c<3>,"px",kvi);

                        auto t = LSL_GEO::ray_ray_intersect(px,x - px,pkx,kx - pkx,(pscale + kpscale) * (T)2);  
                        if(t < min_contact_time) {
                            min_contact_time = t;
                            contact_kvi = kvi;
                        }                      
                    };
                    kbvh.iter_neighbors(bv,process_ccd_collision);

                    if(contact_kvi >= 0) {
                        csPP.insert(vec2i{vi,contact_kvi});
                    }
            });

            eles.resize(csPP.size());
            colors.resize(csPP.size());
            reordered_map.resize(csPP.size());

            eles.append_channels(cudaPol,{{"inds",2}});
            cudaPol(zip(zs::range(csPP.size()),csPP._activeKeys),[
                    eles = proxy<space>({},eles),
                    colors = proxy<space>(colors),
                    reordered_map = proxy<space>(reordered_map)] ZS_LAMBDA(auto ei,const auto& pair) mutable {
                eles.tuple(dim_c<2>,"inds",ei) = pair.reinterpret_bits(float_c);
                colors[ei] = (T)0;
                reordered_map[ei] = ei;
            });

            color_offset.resize(1);
            color_offset.setVal(0);
        }

        // attach to the closest vertex
        if(type == "vertex_attachment") {
            using bv_t = typename ZenoLinearBvh::lbvh_t::Box;

        }

        // attach to the closest point on the surface
        if(type == "surface_point_attachment") {

        }

        // attach to the tetmesh
        if(type == "tetrahedra_attachment") {

        }


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

        constraint->setMeta("color_offset",color_offset);

        // set_output("source",source);
        set_output("constraint",constraint);
    };
};

ZENDEFNODE(MakeSurfaceConstraintTopology, {{
                                {"source"},
                                {"target"},
                                {"float","stiffness","0.5"},
                                {"string","topo_type","stretch"},
                                {"float","rest_scale","1.0"}
                            },
							{{"constraint"}},
							{ 
                                // {"string","groupID",""},
                            },
							{"PBD"}});


struct VisualizePBDConstraint : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;
    // using tiles_t = typename ZenoParticles::particles_t;
    // using dtiles_t = zs::TileVector<T,32>;

    virtual void apply() override {
        using namespace zs;
        using namespace PBD_CONSTRAINT;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto constraints_ptr = get_input<ZenoParticles>("constraints");

        const auto& geo_verts = zsparticles->getParticles();
        const auto& constraints = constraints_ptr->getQuadraturePoints();

        auto tclr_tag = get_param<std::string>("tclrTag");

        zs::Vector<vec3> cvis{geo_verts.get_allocator(),constraints.getChannelSize("inds") * constraints.size()};
        zs::Vector<int> cclrs{constraints.get_allocator(),constraints.size()};
        int cdim = constraints.getChannelSize("inds");
        cudaPol(zs::range(constraints.size()),[
            constraints = proxy<space>({},constraints),
            geo_verts = proxy<space>({},geo_verts),
            cclrs = proxy<space>(cclrs),
            tclr_tag = zs::SmallString(tclr_tag),
            cdim = cdim,
            cvis = proxy<space>(cvis)] ZS_LAMBDA(int ci) mutable {
                // auto cdim = constraints.propertySize("inds");
                for(int i = 0;i != cdim;++i) {
                    auto vi = zs::reinterpret_bits<int>(constraints("inds",i,ci));
                    cvis[ci * cdim + i] = geo_verts.pack(dim_c<3>,"x",vi);
                }
                cclrs[ci] = (int)constraints(tclr_tag,ci);
        });

        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();

        cvis = cvis.clone({zs::memsrc_e::host});
        cclrs = cclrs.clone({zs::memsrc_e::host});
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto& pverts = prim->verts;

        auto constraint_type = constraints_ptr->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});

        if(constraint_type == category_c::edge_length_constraint || constraint_type == category_c::dihedral_spring_constraint) {
            pverts.resize(constraints.size() * 2);
            auto& plines = prim->lines;
            plines.resize(constraints.size());
            auto& tclrs = pverts.add_attr<int>(tclr_tag);
            auto& ltclrs = plines.add_attr<int>(tclr_tag);

            ompPol(zs::range(constraints.size()),[
                &ltclrs,&pverts,&plines,&tclrs,cvis = proxy<omp_space>(cvis),cclrs = proxy<omp_space>(cclrs)] (int ci) mutable {
                    pverts[ci * 2 + 0] = cvis[ci * 2 + 0].to_array();
                    pverts[ci * 2 + 1] = cvis[ci * 2 + 1].to_array();
                    tclrs[ci * 2 + 0] = cclrs[ci];
                    tclrs[ci * 2 + 1] = cclrs[ci];
                    plines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};
                    ltclrs[ci] = cclrs[ci];
            });
        }else if(constraint_type == category_c::isometric_bending_constraint || constraint_type == category_c::dihedral_bending_constraint) {
            pverts.resize(constraints.size() * 2);
            auto& plines = prim->lines;
            plines.resize(constraints.size());
            auto& tclrs = pverts.add_attr<int>(tclr_tag);
            auto& ltclrs = plines.add_attr<int>(tclr_tag);

            ompPol(zs::range(constraints.size()),[
                    &ltclrs,&pverts,&plines,&tclrs,cvis = proxy<omp_space>(cvis),cclrs = proxy<omp_space>(cclrs)] (int ci) mutable {
                zeno::vec3f cverts[4] = {};
                for(int i = 0;i != 4;++i)
                    cverts[i] = cvis[ci * 4 + i].to_array();

                pverts[ci * 2 + 0] = (cverts[0] + cverts[2] + cverts[3]) / (T)3.0;
                pverts[ci * 2 + 1] = (cverts[1] + cverts[2] + cverts[3]) / (T)3.0;
                tclrs[ci * 2 + 0] = cclrs[ci];
                tclrs[ci * 2 + 1] = cclrs[ci];
                ltclrs[ci] = cclrs[ci];

                plines[ci] = zeno::vec2i{ci * 2 + 0,ci * 2 + 1};  
            });
        }

        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(VisualizePBDConstraint, {{{"zsparticles"},{"constraints"}},
							{{"prim"}},
							{
                                {"string","tclrTag","tclrTag"},
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

        auto dt = get_input2<float>("dt");   
        auto ptag = get_param<std::string>("ptag");

        auto coffsets = constraints->readMeta("color_offset",zs::wrapt<zs::Vector<int>>{});  
        int nm_group = coffsets.size();

        auto& verts = zsparticles->getParticles();
        auto& cquads = constraints->getQuadraturePoints();
        auto category = constraints->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});
        auto target = get_input<ZenoParticles>("target");
        const auto& kverts = target->getParticles();
        // zs::Vector<vec3> pv_buffer{verts.get_allocator(),verts.size()};
        // zs::Vector<float> total_ghost_impulse_X{verts.get_allocator(),1};
        // zs::Vector<float> total_ghost_impulse_Y{verts.get_allocator(),1};
        // zs::Vector<float> total_ghost_impulse_Z{verts.get_allocator(),1};

        // std::cout << "SOVLE CONSTRAINT WITH GROUP : " << nm_group << "\t" << cquads.size() << std::endl;



        for(int g = 0;g != nm_group;++g) {

            auto coffset = coffsets.getVal(g);
            int group_size = 0;
            if(g == nm_group - 1)
                group_size = cquads.size() - coffsets.getVal(g);
            else
                group_size = coffsets.getVal(g + 1) - coffsets.getVal(g);

            cudaPol(zs::range(group_size),[
                coffset,
                verts = proxy<space>({},verts),
                category,
                dt,
                ptag = zs::SmallString(ptag),
                kverts = proxy<space>({},kverts),
                cquads = proxy<space>({},cquads)] ZS_LAMBDA(int gi) mutable {
                    float s = cquads("stiffness",coffset + gi);
                    float lambda = cquads("lambda",coffset + gi);

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
                        vec3 dp[4] = {};
                        if(!CONSTRAINT::solve_DihedralConstraint(
                            p[0],minv[0],
                            p[1],minv[1],
                            p[2],minv[2],
                            p[3],minv[3],
                            ra,
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

                    if(category == category_c::p_kp_collision_constraint) {
                        auto quad = cquads.pack(dim_c<2>,"inds",coffset + gi,int_c);

                        vec3 p = verts.pack(dim_c<3>,ptag,quad[0]);
                        vec3 kp = kverts.pack(dim_c<3>,"x",quad[1]);
                        vec3 knrm = kverts.pack(dim_c<3>,"nrm",quad[1]);

                        auto pscale = verts("pscale",quad[0]);
                        auto kpscale = kverts("pscale",quad[1]);

                        T minv = verts("minv",quad[0]);
                        // T kminv = kverts("minv",quad[1]);

                        vec3 dp = {};
                        auto thickness = pscale + kpscale;
                        thickness *= (float)1.01;
                        CONSTRAINT::solve_PlaneConstraint(p,minv,kp,knrm,thickness,s,dt,lambda,dp);
                        verts.tuple(dim_c<3>,ptag,quad[0]) = p + dp;
                    }

                    cquads("lambda",coffset + gi) = lambda;
            });

        }      

        // cudaPol(zs::range(verts.size()))

        set_output("constraints",constraints);
        set_output("zsparticles",zsparticles);
        set_output("target",target);
    };
};

ZENDEFNODE(XPBDSolve, {{{"zsparticles"},{"constraints"},{"target"},{"float","dt","0.5"}},
							{{"zsparticles"},{"constraints"},{"target"}},
							{{"string","ptag","X"}},
							{"PBD"}});

struct XPBDSolveSmooth : INode {

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

        auto all_constraints = RETRIEVE_OBJECT_PTRS(ZenoParticles, "all_constraints");
        auto ptag = get_param<std::string>("ptag");
        auto w = get_input2<float>("relaxation_strength");

        auto& verts = zsparticles->getParticles();

        zs::Vector<float> dp_buffer{verts.get_allocator(),verts.size() * 3};
        cudaPol(zs::range(dp_buffer),[]ZS_LAMBDA(auto& v) {v = 0;});
        zs::Vector<int> dp_count{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(dp_count),[]ZS_LAMBDA(auto& c) {c = 0;});

        for(auto &&constraints : all_constraints) {
            const auto& cquads = constraints->getQuadraturePoints();
            auto category = constraints->readMeta(CONSTRAINT_KEY,wrapt<category_c>{});

            cudaPol(zs::range(cquads.size()),[
                verts = proxy<space>({},verts),
                category,
                // dt,
                // w,
                exec_tag,
                dp_buffer = proxy<space>(dp_buffer),
                dp_count = proxy<space>(dp_count),
                ptag = zs::SmallString(ptag),
                cquads = proxy<space>({},cquads)] ZS_LAMBDA(int ci) mutable {
                    float s = cquads("stiffness",ci);
                    float lambda = cquads("lambda",ci);

                    if(category == category_c::dihedral_bending_constraint) {
                        auto quad = cquads.pack(dim_c<4>,"inds",ci,int_c);
                        vec3 p[4] = {};
                        float minv[4] = {};
                        for(int i = 0;i != 4;++i) {
                            p[i] = verts.pack(dim_c<3>,ptag,quad[i]);
                            minv[i] = verts("minv",quad[i]);
                        }

                        auto ra = cquads("ra",ci);
                        vec3 dp[4] = {};
                        if(!CONSTRAINT::solve_DihedralConstraint(
                            p[0],minv[0],
                            p[1],minv[1],
                            p[2],minv[2],
                            p[3],minv[3],
                            ra,
                            (float)1,
                            dp[0],dp[1],dp[2],dp[3]))
                                return;
                        for(int i = 0;i != 4;++i)
                            for(int j = 0;j != 3;++j)
                                atomic_add(exec_tag,&dp_buffer[quad[i] * 3 + j],dp[i][j]);
                        for(int i = 0;i != 4;++i)
                            atomic_add(exec_tag,&dp_count[quad[i]],(int)1);                      
                    }

                    if(category == category_c::edge_length_constraint) {
                        auto edge = cquads.pack(dim_c<2>,"inds",ci,int_c);
                        vec3 p0{},p1{};
                        p0 = verts.pack(dim_c<3>,ptag,edge[0]);
                        p1 = verts.pack(dim_c<3>,ptag,edge[1]);
                        float minv0 = verts("minv",edge[0]);
                        float minv1 = verts("minv",edge[1]);
                        float r = cquads("r",ci);

                        vec3 dp0{},dp1{};
                        if(!CONSTRAINT::solve_DistanceConstraint(
                            p0,minv0,
                            p1,minv1,
                            r,
                            (float)1,
                            dp0,dp1)) {
                        
                            for(int i = 0;i != 3;++i)
                                atomic_add(exec_tag,&dp_buffer[edge[0] * 3 + i],dp0[i]);
                            for(int i = 0;i != 3;++i)
                                atomic_add(exec_tag,&dp_buffer[edge[1] * 3 + i],dp1[i]);

                            atomic_add(exec_tag,&dp_count[edge[0]],(int)1);
                            atomic_add(exec_tag,&dp_count[edge[1]],(int)1);
                        }
                    }
                    // if(category == category_c::isometric_bending_constraint) {
                    //     return;
                    //     auto quad = cquads.pack(dim_c<4>,"inds",ci,int_c);
                    //     vec3 p[4] = {};
                    //     float minv[4] = {};
                    //     for(int i = 0;i != 4;++i) {
                    //         p[i] = verts.pack(dim_c<3>,ptag,quad[i]);
                    //         minv[i] = verts("minv",quad[i]);
                    //     }

                    //     auto Q = cquads.pack(dim_c<4,4>,"Q",ci);

                    //     vec3 dp[4] = {};
                    //     float lambda = 0;
                    //     CONSTRAINT::solve_IsometricBendingConstraint(
                    //         p[0],minv[0],
                    //         p[1],minv[1],
                    //         p[2],minv[2],
                    //         p[3],minv[3],
                    //         Q,
                    //         (float)1,
                    //         dp[0],
                    //         dp[1],
                    //         dp[2],
                    //         dp[3]);

                    //     auto has_nan = false;
                    //     for(int i = 0;i != 4;++i)
                    //         if(zs::isnan(dp[i].norm()))
                    //             has_nan = true;
                    //     if(has_nan) {
                    //         printf("nan dp detected : %f %f %f %f %f %f %f %f\n",
                    //             (float)p[0].norm(),
                    //             (float)p[1].norm(),
                    //             (float)p[2].norm(),
                    //             (float)p[3].norm(),
                    //             (float)dp[0].norm(),
                    //             (float)dp[1].norm(),
                    //             (float)dp[2].norm(),
                    //             (float)dp[3].norm());
                    //         return;
                    //     }
                    //     for(int i = 0;i != 4;++i)
                    //         for(int j = 0;j != 3;++j)
                    //             atomic_add(exec_tag,&dp_buffer[quad[i] * 3 + j],dp[i][j]);
                    //     for(int i = 0;i != 4;++i)
                    //         atomic_add(exec_tag,&dp_count[quad[i]],(int)1);
                    // }
            });
        }      

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            ptag = zs::SmallString(ptag),w,
            dp_count = proxy<space>(dp_count),
            dp_buffer = proxy<space>(dp_buffer)] ZS_LAMBDA(int vi) mutable {
                if(dp_count[vi] > 0) {
                    auto dp = w * vec3{dp_buffer[vi * 3 + 0],dp_buffer[vi * 3 + 1],dp_buffer[vi * 3 + 2]};
                    verts.tuple(dim_c<3>,ptag,vi) = verts.pack(dim_c<3>,ptag,vi) + dp / (T)dp_count[vi];
                }
        });

        // set_output("all_constraints",all_constraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolveSmooth, {{{"zsparticles"},{"all_constraints"},{"float","relaxation_strength","1"}},
							{{"zsparticles"}},
							{{"string","ptag","x"}},
							{"PBD"}});





};
