#include "Structures.hpp"
// #include "../fem/SolverUtils.cuh"
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
#include "../fem/collision_energy/evaluate_collision.hpp"
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

            eles.append_channels(cudaPol,{{"inds",4},{"Q",4 * 4}});

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
                    CONSTRAINT::init_IsometricBendingConstraint(x[0],x[1],x[2],x[3],Q);
                    eles.tuple(dim_c<16>,"Q",oei) = Q;
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

        if(type == "attachment") {
            using bv_t = typename ZenoLinearBvh::lbvh_t::Box;

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

        if(constraint_type == category_c::edge_length_constraint) {
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
        }else if(constraint_type == category_c::isometric_bending_constraint) {
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

            // if(category == category_c::isometric_bending_constraint)
            //     break;

            auto coffset = coffsets.getVal(g);
            int group_size = 0;
            if(g == nm_group - 1)
                group_size = cquads.size() - coffsets.getVal(g);
            else
                group_size = coffsets.getVal(g + 1) - coffsets.getVal(g);

            // cudaPol(zs::range(verts.size()),[
            //     ptag = zs::SmallString(ptag),
            //     verts = proxy<space>({},verts),
            //     pv_buffer = proxy<space>(pv_buffer)] ZS_LAMBDA(int vi) mutable {
            //         pv_buffer[vi] = verts.pack(dim_c<3>,ptag,vi);
            // });



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

                    if(category == category_c::edge_length_constraint) {
                        auto edge = cquads.pack(dim_c<2>,"inds",coffset + gi,int_c);
                        vec3 p0{},p1{};
                        p0 = verts.pack(dim_c<3>,ptag,edge[0]);
                        p1 = verts.pack(dim_c<3>,ptag,edge[1]);
                        float minv0 = verts("minv",edge[0]);
                        float minv1 = verts("minv",edge[1]);
                        float r = cquads("r",coffset + gi);

                        vec3 dp0{},dp1{};
                        CONSTRAINT::solve_DistanceConstraint(
                            p0,minv0,
                            p1,minv1,
                            r,
                            s,
                            dt,
                            lambda,
                            dp0,dp1);
                        
                        
                        verts.tuple(dim_c<3>,ptag,edge[0]) = p0 + dp0;
                        verts.tuple(dim_c<3>,ptag,edge[1]) = p1 + dp1;

                        // float m0 = verts("m",edge[0]);
                        // float m1 = verts("m",edge[1]);
                        // auto ghost_impulse = (dp0 * m0 + dp1 * m1).norm();
                        // if(ghost_impulse > 1e-6)
                        //     printf("dmomentum : %f\n",(float)ghost_impulse);
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

                        vec3 dp[4] = {};
                        CONSTRAINT::solve_IsometricBendingConstraint(
                            p[0],minv[0],
                            p[1],minv[1],
                            p[2],minv[2],
                            p[3],minv[3],
                            Q,
                            s,
                            dt,
                            lambda,
                            dp[0],dp[1],dp[2],dp[3]);

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

            // total_ghost_impulse_X.setVal(0);
            // total_ghost_impulse_Y.setVal(0);
            // total_ghost_impulse_Z.setVal(0);
            // if(category == category_c::isometric_bending_constraint)
            //     std::cout << "isometric_bending_constraint output" << std::endl;
            // if(category == category_c::edge_length_constraint)
            //     std::cout << "edge_length_constraint output" << std::endl;     
            
            // cudaPol(zs::range(verts.size()),[
            //     verts = proxy<space>({},verts),
            //     ptag = zs::SmallString(ptag)
            //     // exec_tag,
            //     // total_ghost_impulse_X = proxy<space>(total_ghost_impulse_X),
            //     // total_ghost_impulse_Y = proxy<space>(total_ghost_impulse_Y),
            //     // total_ghost_impulse_Z = proxy<space>(total_ghost_impulse_Z),
            //     // pv_buffer = proxy<space>(pv_buffer)
            //     ] ZS_LAMBDA(int vi) mutable {
            //         auto cv = verts.pack(dim_c<3>,ptag,vi);
            //         // auto pv = pv_buffer[vi];
            //         // auto m = verts("m",vi);
            //         // auto dv = m * (cv - pv);
            //         // for(int i = 0;i != 3;++i)
            //         // atomic_add(exec_tag,&total_ghost_impulse_X[0],dv[0]);
            //         // atomic_add(exec_tag,&total_ghost_impulse_Y[0],dv[1]);
            //         // atomic_add(exec_tag,&total_ghost_impulse_Z[0],dv[2]);
            //         printf("cv[%d] : %f %f %f\n",vi,(float)cv[0],(float)cv[1],(float)cv[2]);
            // });

            // auto tgi = total_ghost_impulse.getVal(0);
            // auto tgx = total_ghost_impulse_X.getVal(0);
            // auto tgy = total_ghost_impulse_Y.getVal(0);
            // auto tgz = total_ghost_impulse_Z.getVal(0);
            // printf("ghost_impulse[%d][%d] : %f %f %f\n",(int)coffset,(int)group_size,(float)tgx,(float)tgy,(float)tgz);
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
                        if(vn < -thickness * 0.05) {
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

        for(int iter = 0;iter != nm_ccd_iters;++iter) {

            cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});
            cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

            nm_ccd_collision.setVal(0);

            if(do_pt_detection) {
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

            // std::cout << "apply impulse" << std::endl;
            cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                vtemp = proxy<space>({},vtemp),
                impulse_buffer = proxy<space>(impulse_buffer),
                impulse_count = proxy<space>(impulse_count),
                relaxation_rate = relaxation_rate,
                nm_ccd_collision = proxy<space>(nm_ccd_collision),
                eps = eps,
                thickness = thickness,
                exec_tag = exec_tag] ZS_LAMBDA(int vi) mutable {
                if(impulse_count[vi] == 0)
                    return;
                if(impulse_buffer[vi].norm() < eps)
                    return;

                auto impulse = relaxation_rate * impulse_buffer[vi] / impulse_count[vi];
                if(impulse.norm() > thickness * 0.01)
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

struct SDFColliderProject : INode {
    using T = float;
    using vec3 = zs::vec<T,3>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zsparticles = get_input<ZenoParticles>("zsparticles");

        auto radius = get_input2<float>("radius");
        auto center = get_input2<zeno::vec3f>("center");
        auto cv = get_input2<zeno::vec3f>("center_velocity");
        auto w = get_input2<zeno::vec3f>("angular_velocity");

        // prex
        auto xtag = get_input2<std::string>("xtag");
        // x
        auto ptag = get_input2<std::string>("ptag");
        auto friction = get_input2<T>("friction");

        // auto do_stablize = get_input2<bool>("do_stablize");

        auto& verts = zsparticles->getParticles();

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            ptag = zs::SmallString(ptag),
            friction,
            radius,
            center,
            // do_stablize,
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
                auto rel_vel = pred - pos - center_vel;

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
        set_output("zsparticles",zsparticles);
    }

};

ZENDEFNODE(SDFColliderProject, {{{"zsparticles"},
                                {"float","radius","1"},
                                {"center"},
                                {"center_velocity"},
                                {"angular_velocity"},
                                {"string","xtag","x"},
                                {"string","ptag","x"},
                                {"float","friction","0"}
                                // {"bool","do_stablize","0"}
                            },
							{{"zsparticles"}},
							{},
							{"PBD"}});


struct BuildZSLBvhFromAABB : INode {

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
        auto cudaPol = cuda_exec().device(0);

        auto pars = get_input<ZenoParticles>("ZSParticles");
        auto srcTag = get_input2<std::string>("fromTag");
        auto dstTag = get_input2<std::string>("toTag");
        auto pscaleTag = get_input2<std::string>("pscaleTag");

        auto out = std::make_shared<ZenoLinearBvh>();
        auto &bvh = out->get();

        buildBvh(cudaPol,pars->getParticles(),srcTag,dstTag,pscaleTag,bvh);
        out->thickness = 0;

        set_output("ZSLBvh",std::move(out));
    }
};

ZENDEFNODE(BuildZSLBvhFromAABB, {{{"ZSParticles"},
     {"string","fromTag","px"},
     {"string","toTag","x"},
     {"string","pscaleTag","pscale"}
    }, {"ZSLBvh"}, {}, {"XPBD"}});


struct XPBDDetangle : INode {
    // ray trace bvh
    template <typename TileVecT>
    void buildRayTraceBvh(zs::CudaExecutionPolicy &pol, 
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
    // particle sphere bvh
    template <typename TileVecT>
    void buildBvh(zs::CudaExecutionPolicy &pol,
            TileVecT &verts,
            const zs::SmallString& xtag,
            const zs::SmallString& pscaleTag,
            ZenoLinearBvh::lbvh_t& bvh) {
        using namespace zs;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> bvs{verts.get_allocator(), verts.size()};

        pol(range(verts.size()),
            [verts = proxy<space>({}, verts),
             bvs = proxy<space>(bvs),
             pscaleTag,xtag] ZS_LAMBDA(int vi) mutable {
                auto pos = verts.template pack<3>(xtag, vi);
                auto pscale = verts(pscaleTag,vi);
                bv_t bv{pos - pscale,pos + pscale};
                bvs[vi] = bv;
            });
        bvh.build(pol, bvs);
    }

    virtual void apply() override {
        using namespace zs;
        using bvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename bvh_t::Box;

        using vec3 = zs::vec<float,3>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto xtag = get_input2<std::string>("xtag");
        auto pxtag = get_input2<std::string>("pxtag");
        auto Xtag = get_input2<std::string>("Xtag");
        // auto ccdTag = get_input2<std::string>("ccdTag");
        // x
        auto pscaleTag = get_input2<std::string>("pscaleTag");
        // auto friction = get_input2<T>("friction");
        
        auto& verts = zsparticles->getParticles();     

        auto spBvh = bvh_t{};
        buildRayTraceBvh(cudaPol,verts,pxtag,xtag,pscaleTag,spBvh);       
        
        zs::Vector<float> ccd_buffer{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(ccd_buffer),[]ZS_LAMBDA(auto& t) mutable {t = (float)1;});
        // zs::Vector<int> dp_count{verts.get_allocator(),verts.size()};
        // cudaPol(zs::range(dp_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            Xtag = zs::SmallString(Xtag),
            pxtag = zs::SmallString(pxtag),
            exec_tag,
            ccd_buffer = proxy<space>(ccd_buffer),
            pscaleTag = zs::SmallString(pscaleTag),
            spBvh = proxy<space>(spBvh)] ZS_LAMBDA(int vi) mutable {
                auto dst = verts.pack(dim_c<3>,xtag,vi);
                auto src = verts.pack(dim_c<3>,pxtag,vi);
                auto vel = dst - src;
                auto rpos = verts.pack(dim_c<3>,Xtag,vi);
                // auto r = verts(ptag,vi);

                auto pscale = verts(pscaleTag,vi);
                bv_t bv{src,dst};
                bv._min -= pscale;
                bv._max += pscale;

                auto find_particle_sphere_collision_pairs = [&] (int nvi) mutable {
                    if(vi >= nvi)
                        return;

                    auto npscale = verts(pscaleTag,nvi);
                    auto thickness = pscale + npscale;

                    auto nrpos = verts.pack(dim_c<3>,Xtag,nvi);
                    auto rdist = (rpos - nrpos).norm();
                    if(rdist < thickness)
                        return;

                    auto ndst = verts.pack(dim_c<3>,xtag,nvi);
                    auto nsrc = verts.pack(dim_c<3>,pxtag,nvi);
                    auto nvel = ndst - nsrc;

                    auto t = LSL_GEO::ray_ray_intersect(src,vel,nsrc,nvel,thickness);
                    if(t <= (float)1 && t >= (float)0) {
                        atomic_min(exec_tag,&ccd_buffer[vi],t);
                        atomic_min(exec_tag,&ccd_buffer[nvi],t);
                    }
                };

                spBvh.iter_neighbors(bv,find_particle_sphere_collision_pairs);
        });

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            pxtag = zs::SmallString(pxtag),
            // friction,
            ccd_buffer = proxy<space>(ccd_buffer)] ZS_LAMBDA(int vi) mutable {
                auto src = verts.pack(dim_c<3>,pxtag,vi);
                auto dst = verts.pack(dim_c<3>,xtag,vi);

                auto vel = dst - src;
                auto project = src + vel * ccd_buffer[vi];

                verts.tuple(dim_c<3>,xtag,vi) = project;
        });

        set_output("zsparticles",zsparticles);
    }

};

ZENDEFNODE(XPBDDetangle, {{{"zsparticles"},
                                // {"float","relaxation_strength","1"},
                                {"string","xtag","x"},
                                {"string","pxtag","px"},
                                {"string","Xtag","X"},
                                {"string","pscaleTag","pscale"}
                                // {"float","friction","0"}                            
                            },
							{{"zsparticles"}},
							{
                                // {"string","ptag","x"}
                            },
							{"PBD"}});


struct XPBDDetangle2 : INode {
    // ray trace bvh
    template <typename TileVecT>
    void buildRayTraceBvh(zs::CudaExecutionPolicy &pol, 
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
    // particle sphere bvh
    // template <typename TileVecT>
    // void buildBvh(zs::CudaExecutionPolicy &pol,
    //         TileVecT &verts,
    //         const zs::SmallString& xtag,
    //         const zs::SmallString& pscaleTag,
    //         ZenoLinearBvh::lbvh_t& bvh) {
    //     using namespace zs;
    //     using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
    //     constexpr auto space = execspace_e::cuda;
    //     Vector<bv_t> bvs{verts.get_allocator(), verts.size()};

    //     pol(range(verts.size()),
    //         [verts = proxy<space>({}, verts),
    //             bvs = proxy<space>(bvs),
    //             pscaleTag,xtag] ZS_LAMBDA(int vi) mutable {
    //             auto pos = verts.template pack<3>(xtag, vi);
    //             auto pscale = verts(pscaleTag,vi);
    //             bv_t bv{pos - pscale,pos + pscale};
    //             bvs[vi] = bv;
    //         });
    //     bvh.build(pol, bvs);
    // }

    virtual void apply() override {
        using namespace zs;
        using bvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename bvh_t::Box;

        using vec3 = zs::vec<float,3>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto xtag = get_input2<std::string>("xtag");
        auto pxtag = get_input2<std::string>("pxtag");
        auto Xtag = get_input2<std::string>("Xtag");
        // auto ccdTag = get_input2<std::string>("ccdTag");
        // x
        auto pscaleTag = get_input2<std::string>("pscaleTag");
        // auto friction = get_input2<T>("friction");
        
        auto& verts = zsparticles->getParticles();    
        
        auto restitution_rate = get_input2<float>("restitution");
        auto relaxation_rate = get_input2<float>("relaxation");

        auto spBvh = bvh_t{};
        buildRayTraceBvh(cudaPol,verts,pxtag,xtag,pscaleTag,spBvh);       
        
        // zs::Vector<float> ccd_buffer{verts.get_allocator(),verts.size()};
        // cudaPol(zs::range(ccd_buffer),[]ZS_LAMBDA(auto& t) mutable {t = (float)1;});

        zs::Vector<vec3> dp_buffer{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(dp_buffer),[]ZS_LAMBDA(auto& dp) mutable {dp = vec3::uniform(0);});

        zs::Vector<int> dp_count{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(dp_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            Xtag = zs::SmallString(Xtag),
            pxtag = zs::SmallString(pxtag),
            restitution_rate = restitution_rate,
            exec_tag = exec_tag,
            eps = eps,
            dp_buffer = proxy<space>(dp_buffer),
            dp_count = proxy<space>(dp_count),
            pscaleTag = zs::SmallString(pscaleTag),
            spBvh = proxy<space>(spBvh)] ZS_LAMBDA(int vi) mutable {
                auto dst = verts.pack(dim_c<3>,xtag,vi);
                auto src = verts.pack(dim_c<3>,pxtag,vi);
                auto vel = dst - src;
                auto rpos = verts.pack(dim_c<3>,Xtag,vi);
                // auto r = verts(ptag,vi);

                auto pscale = verts(pscaleTag,vi);
                bv_t bv{src,dst};
                bv._min -= pscale;
                bv._max += pscale;

                auto find_particle_sphere_collision_pairs = [&] (int nvi) mutable {
                    if(vi >= nvi)
                        return;

                    auto npscale = verts(pscaleTag,nvi);
                    auto thickness = pscale + npscale;

                    auto nrpos = verts.pack(dim_c<3>,Xtag,nvi);
                    auto rdist = (rpos - nrpos).norm();
                    if(rdist < thickness)
                        return;

                    auto ndst = verts.pack(dim_c<3>,xtag,nvi);
                    auto nsrc = verts.pack(dim_c<3>,pxtag,nvi);
                    auto nvel = ndst - nsrc;

                    auto t = LSL_GEO::ray_ray_intersect(src,vel,nsrc,nvel,thickness);
                    if(t <= (float)1 && t >= (float)0) {
                        // atomic_min(exec_tag,&ccd_buffer[vi],t);
                        // atomic_min(exec_tag,&ccd_buffer[nvi],t);
                        // auto rel_vel = vel - nvel;
                        auto CR = restitution_rate;

                        auto minv_a = verts("minv",vi);
                        auto minv_b = verts("minv",nvi);
                        if(minv_a + minv_b < (T)2 * eps)
                            return;
                        
                        auto ua = vel;
                        auto ub = nvel;
                        auto ur = ua - ub;

                        vec3 va{},vb{};

                        if(minv_a > (T)eps) {
                            // ma / mb
                            auto ratio = minv_b / minv_a;
                            auto m = ratio + (T)1;
                            auto momt = ratio * ua + ub;
                            va = (momt - CR * ur) / m;
                            vb = (momt + CR * ratio * ur) / m;

                            if(isnan(va.norm()) || isnan(vb.norm())) {
                                printf("nan value detected : %f %f\n",(float)va.norm(),(float)vb.norm());
                            }
                        }else if(minv_b > (T)eps) {
                            // mb / ma
                            auto ratio = minv_a / minv_b;
                            auto m = ratio + (T)1;
                            auto momt = ua + ratio * ub;
                            va = (momt - CR * ratio * ur) / m;
                            vb = (momt + CR * ur) / m;
                            if(isnan(va.norm()) || isnan(vb.norm())) {
                                printf("nan value detected : %f %f\n",(float)va.norm(),(float)vb.norm());
                            }

                        }else {
                            printf("impossible reaching here\n");
                        }

                        // auto ma = verts("m",vi);
                        // auto mb = verts("m",nvi);

                        // auto m = ma + mb;
                        // auto momt = ma * ua + mb * ub;


                        // auto va = (momt - CR * mb * ur) / m;
                        // auto vb = (momt + CR * ma * ur) / m;

                        auto dpa = (va - ua) * (1 - t);
                        auto dpb = (vb - ub) * (1 - t);

                        // auto dpa = (va - ua);
                        // auto dpb = (vb - ub);
                        // printf("find collision pair : %d %d\n",vi,nvi);
                        atomic_add(exec_tag,&dp_count[vi],1);
                        atomic_add(exec_tag,&dp_count[nvi],1);

                        for(int i = 0;i != 3;++i) {
                            atomic_add(exec_tag,&dp_buffer[vi][i],dpa[i]);
                            atomic_add(exec_tag,&dp_buffer[nvi][i],dpb[i]);
                        }
                    }
                };

                spBvh.iter_neighbors(bv,find_particle_sphere_collision_pairs);
        });

        // zs:Vector<T> dpnorm{verts.get_allocator(),1};
        // dpnorm.setVal(0);

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            relaxation_rate = relaxation_rate,
            pxtag = zs::SmallString(pxtag),
            exec_tag,
            // dpnorm = proxy<space>(dpnorm),
            dp_count = proxy<space>(dp_count),
            dp_buffer = proxy<space>(dp_buffer)] ZS_LAMBDA(int vi) mutable {
                if(dp_count[vi] == 0)
                    return;
                auto minv = verts("minv",vi);
                auto dp = dp_buffer[vi] * relaxation_rate / (T)dp_count[vi];
                // atomic_add(exec_tag,&dpnorm[0],dp.norm());
                verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + dp;
        });

        // std::cout << "detangle_dp_norm : " << dpnorm.getVal(0) << std::endl;

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(XPBDDetangle2, {{{"zsparticles"},
                                // {"float","relaxation_strength","1"},
                                {"string","xtag","x"},
                                {"string","pxtag","px"},
                                {"string","Xtag","X"},
                                {"string","pscaleTag","pscale"},
                                {"float","restitution","0.1"},
                                {"float","relaxation","1"},
                                // {"float","friction","0"}                            
                            },
                            {{"zsparticles"}},
                            {
                                // {"string","ptag","x"}
                            },
                            {"PBD"}});



// struct XPBDDetangle3 : INode {
//     // ray trace bvh
//     template <typename TileVecT,typename ElmTileVec>
//     void buildTriRayTraceBvh(zs::CudaExecutionPolicy &pol, 
//             const TileVecT &verts, 
//             const ElmTileVec& tris,
//             const T& dt,
//             const zs::SmallString& xTag,
//             const zs::SmallString& vTag,
//             const T& thickness,ZenoLinearBvh::lbvh_t &bvh) {
//         using namespace zs;
//         using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
//         constexpr auto space = execspace_e::cuda;
//         Vector<bv_t> bvs{tris.get_allocator(), tris.size()};
//         retrieve_bounding_volumes(pol,verts,xTag,tris,verts,vTag,dt,0,bvs); 
//         bvh.build(pol, bvs);
//     }

//     virtual void apply() override {
//         using namespace zs;
//         using bvh_t = typename ZenoLinearBvh::lbvh_t;
//         using bv_t = typename bvh_t::Box;

//         using vec3 = zs::vec<float,3>;
//         using vec2i = zs::vec<int,2>;
//         using vec3i = zs::vec<int,3>;
//         using vec4i = zs::vec<int,4>;
//         using mat4 = zs::vec<int,4,4>;

//         constexpr auto space = execspace_e::cuda;
//         auto cudaPol = cuda_exec();
//         constexpr auto exec_tag = wrapv<space>{};
//         constexpr auto eps = (T)1e-7;

//         auto zsparticles = get_input<ZenoParticles>("zsparticles");
//         auto xtag = get_input2<std::string>("xtag");
//         auto vtag = get_input2<std::string>("vtag");
//         // auto pxtag = get_input2<std::string>("pxtag");
//         // auto Xtag = get_input2<std::string>("Xtag");
//         // auto ccdTag = get_input2<std::string>("ccdTag");
//         // x
//         // auto pscaleTag = get_input2<std::string>("pscaleTag");
//         // auto friction = get_input2<T>("friction");
        
//         auto& verts = zsparticles->getParticles();   

//         auto restitution_rate = get_input2<float>("restitution");
//         auto relaxation_rate = get_input2<float>("relaxation");

//         auto triRayBvh = bvh_t{};
//         buildTriRayTraceBvh(cudaPol,verts,pxtag,xtag,pscaleTag,triRayBvh);       
        
//         // zs::Vector<float> ccd_buffer{verts.get_allocator(),verts.size()};
//         // cudaPol(zs::range(ccd_buffer),[]ZS_LAMBDA(auto& t) mutable {t = (float)1;});

//         zs::Vector<vec3> dp_buffer{verts.get_allocator(),verts.size()};
//         cudaPol(zs::range(dp_buffer),[]ZS_LAMBDA(auto& dp) mutable {dp = vec3::uniform(0);});

//         zs::Vector<int> dp_count{verts.get_allocator(),verts.size()};
//         cudaPol(zs::range(dp_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

//         cudaPol(zs::range(verts.size()),[
//             verts = proxy<space>({},verts),
//             xtag = zs::SmallString(xtag),
//             Xtag = zs::SmallString(Xtag),
//             pxtag = zs::SmallString(pxtag),
//             restitution_rate = restitution_rate,
//             exec_tag = exec_tag,
//             eps = eps,
//             dp_buffer = proxy<space>(dp_buffer),
//             dp_count = proxy<space>(dp_count),
//             pscaleTag = zs::SmallString(pscaleTag),
//             spBvh = proxy<space>(spBvh)] ZS_LAMBDA(int vi) mutable {
//                 auto dst = verts.pack(dim_c<3>,xtag,vi);
//                 auto src = verts.pack(dim_c<3>,pxtag,vi);
//                 auto vel = dst - src;
//                 auto rpos = verts.pack(dim_c<3>,Xtag,vi);
//                 // auto r = verts(ptag,vi);

//                 auto pscale = verts(pscaleTag,vi);
//                 bv_t bv{src,dst};
//                 bv._min -= pscale;
//                 bv._max += pscale;

//                 auto find_particle_sphere_collision_pairs = [&] (int nvi) mutable {
//                     if(vi >= nvi)
//                         return;

//                     auto npscale = verts(pscaleTag,nvi);
//                     auto thickness = pscale + npscale;

//                     auto nrpos = verts.pack(dim_c<3>,Xtag,nvi);
//                     auto rdist = (rpos - nrpos).norm();
//                     if(rdist < thickness)
//                         return;

//                     auto ndst = verts.pack(dim_c<3>,xtag,nvi);
//                     auto nsrc = verts.pack(dim_c<3>,pxtag,nvi);
//                     auto nvel = ndst - nsrc;

//                     auto t = LSL_GEO::ray_ray_intersect(src,vel,nsrc,nvel,thickness);
//                     if(t <= (float)1 && t >= (float)0) {
//                         // atomic_min(exec_tag,&ccd_buffer[vi],t);
//                         // atomic_min(exec_tag,&ccd_buffer[nvi],t);
//                         // auto rel_vel = vel - nvel;
//                         auto CR = restitution_rate;

//                         auto minv_a = verts("minv",vi);
//                         auto minv_b = verts("minv",nvi);
//                         if(minv_a + minv_b < (T)2 * eps)
//                             return;
                        
//                         auto ua = vel;
//                         auto ub = nvel;
//                         auto ur = ua - ub;

//                         vec3 va{},vb{};

//                         if(minv_a > (T)eps) {
//                             // ma / mb
//                             auto ratio = minv_b / minv_a;
//                             auto m = ratio + (T)1;
//                             auto momt = ratio * ua + ub;
//                             va = (momt - CR * ur) / m;
//                             vb = (momt + CR * ratio * ur) / m;

//                             if(isnan(va.norm()) || isnan(vb.norm())) {
//                                 printf("nan value detected : %f %f\n",(float)va.norm(),(float)vb.norm());
//                             }
//                         }else if(minv_b > (T)eps) {
//                             // mb / ma
//                             auto ratio = minv_a / minv_b;
//                             auto m = ratio + (T)1;
//                             auto momt = ua + ratio * ub;
//                             va = (momt - CR * ratio * ur) / m;
//                             vb = (momt + CR * ur) / m;
//                             if(isnan(va.norm()) || isnan(vb.norm())) {
//                                 printf("nan value detected : %f %f\n",(float)va.norm(),(float)vb.norm());
//                             }

//                         }else {
//                             printf("impossible reaching here\n");
//                         }

//                         // auto ma = verts("m",vi);
//                         // auto mb = verts("m",nvi);

//                         // auto m = ma + mb;
//                         // auto momt = ma * ua + mb * ub;


//                         // auto va = (momt - CR * mb * ur) / m;
//                         // auto vb = (momt + CR * ma * ur) / m;

//                         auto dpa = (va - ua) * (1 - t);
//                         auto dpb = (vb - ub) * (1 - t);

//                         // auto dpa = (va - ua);
//                         // auto dpb = (vb - ub);
//                         // printf("find collision pair : %d %d\n",vi,nvi);
//                         atomic_add(exec_tag,&dp_count[vi],1);
//                         atomic_add(exec_tag,&dp_count[nvi],1);

//                         for(int i = 0;i != 3;++i) {
//                             atomic_add(exec_tag,&dp_buffer[vi][i],dpa[i]);
//                             atomic_add(exec_tag,&dp_buffer[nvi][i],dpb[i]);
//                         }
//                     }
//                 };

//                 spBvh.iter_neighbors(bv,find_particle_sphere_collision_pairs);
//         });

//         // zs:Vector<T> dpnorm{verts.get_allocator(),1};
//         // dpnorm.setVal(0);

//         cudaPol(zs::range(verts.size()),[
//             verts = proxy<space>({},verts),
//             xtag = zs::SmallString(xtag),
//             relaxation_rate = relaxation_rate,
//             pxtag = zs::SmallString(pxtag),
//             exec_tag,
//             // dpnorm = proxy<space>(dpnorm),
//             dp_count = proxy<space>(dp_count),
//             dp_buffer = proxy<space>(dp_buffer)] ZS_LAMBDA(int vi) mutable {
//                 if(dp_count[vi] == 0)
//                     return;
//                 auto minv = verts("minv",vi);
//                 auto dp = dp_buffer[vi] * relaxation_rate / (T)dp_count[vi];
//                 // atomic_add(exec_tag,&dpnorm[0],dp.norm());
//                 verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + dp;
//         });

//         // std::cout << "detangle_dp_norm : " << dpnorm.getVal(0) << std::endl;

//         set_output("zsparticles",zsparticles);
//     }
// };
                            
// ZENDEFNODE(XPBDDetangle3, {{{"zsparticles"},
//                                 // {"float","relaxation_strength","1"},
//                                 {"string","xtag","x"},
//                                 {"string","pxtag","px"},
//                                 {"string","Xtag","X"},
//                                 {"string","pscaleTag","pscale"},
//                                 {"float","restitution","0.1"},
//                                 {"float","relaxation","1"},
//                                 // {"float","friction","0"}                            
//                             },
//                             {{"zsparticles"}},
//                             {
//                                 // {"string","ptag","x"}
//                             },
//                             {"PBD"}});


// make sure the mesh is intersection-free before using this node
// we use linear trajectory here, such that vel = (x - px) / dt
struct ExchangeLocalMomentum : INode {
    // ray trace bvh
    template <typename TileVecT>
    void buildSphereBvh(zs::CudaExecutionPolicy &pol,
            TileVecT &verts,
            const zs::SmallString& xtag,
            const zs::SmallString& pscaleTag,
            ZenoLinearBvh::lbvh_t& bvh) {
        using namespace zs;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> bvs{verts.get_allocator(), verts.size()};

        pol(range(verts.size()),
            [verts = proxy<space>({}, verts),
                pscaleTag = zs::SmallString(pscaleTag),
                bvs = proxy<space>(bvs),xtag] ZS_LAMBDA(int vi) mutable {
                auto pos = verts.template pack<3>(xtag, vi);
                auto r = verts(pscaleTag,vi);
                bv_t bv{pos - r,pos + r};
                bvs[vi] = bv;
            });
        bvh.build(pol, bvs);
    }

    virtual void apply() override {
        using namespace zs; 
        using bvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename bvh_t::Box;

        using vec3 = zs::vec<float,3>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};
        constexpr auto eps = (T)1e-7;

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto xtag = get_input2<std::string>("xtag");
        auto vtag = get_input2<std::string>("vtag");
        auto Xtag = get_input2<std::string>("Xtag");
        // auto ccdTag = get_input2<std::string>("ccdTag");
        // x
        auto pscaleTag = get_input2<std::string>("pscaleTag");
        // auto thickness = get_input2<float>("thickness");
        // auto friction = get_input2<T>("friction");
        
        auto& verts = zsparticles->getParticles();    
        
        auto restitution_rate = get_input2<float>("restitution");
        auto relaxation_rate = get_input2<float>("relaxation");

        zs::Vector<vec3> impulse_buffer{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(impulse_buffer),[]ZS_LAMBDA(auto& imp) mutable {imp = vec3::uniform(0);});

        zs::Vector<int> impulse_count{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(impulse_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

        auto spBvh = bvh_t{};
        buildSphereBvh(cudaPol,verts,xtag,pscaleTag,spBvh);   

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            pscaleTag = zs::SmallString(pscaleTag),
            Xtag = zs::SmallString(Xtag),
            vtag = zs::SmallString(vtag),
            restitution_rate = restitution_rate,
            exec_tag = exec_tag,
            // thickness = thickness,
            eps = eps,
            impulse_buffer = proxy<space>(impulse_buffer),
            impulse_count = proxy<space>(impulse_count),
            spBvh = proxy<space>(spBvh)] ZS_LAMBDA(int vi) mutable {
                auto pos = verts.pack(dim_c<3>,xtag,vi);
                auto vel = verts.pack(dim_c<3>,vtag,vi);
                auto r = verts(pscaleTag,vi);
                auto rpos = verts.pack(dim_c<3>,Xtag,vi);
                bv_t bv{pos - r,pos + r};

                auto find_particle_sphere_collision_pairs = [&] (int nvi) mutable {
                    if(vi >= nvi)
                        return;

                    auto npos = verts.pack(dim_c<3>,xtag,nvi);
                    auto nvel = verts.pack(dim_c<3>,vtag,nvi);
                    auto nr = verts(pscaleTag,nvi);
                    auto nrpos = verts.pack(dim_c<3>,Xtag,nvi);
                    auto rdist = (nrpos - rpos).norm();

                    auto thickness = r + nr;

                    if(rdist < thickness)
                        return;


                    auto dp = pos - npos;
                    auto dv = vel - nvel;
                    if(dp.dot(dv) > 0)
                        return;

                    auto dist = (pos - npos).norm();
                    // as the bvh is bounding box, for sphere intersection, extra distance check is needed
                    if(dist < (float)thickness) {
                        auto CR = restitution_rate;

                        auto minv_a = verts("minv",vi);
                        auto minv_b = verts("minv",nvi);
                        if(minv_a + minv_b < (T)2 * eps)
                            return;
                        
                        auto ua = vel;
                        auto ub = nvel;
                        auto ur = ua - ub;

                        vec3 va{},vb{};

                        // exchange momentum
                        if(minv_a > (T)eps) {
                            // ma / mb
                            auto ratio = minv_b / minv_a;
                            auto m = ratio + (T)1;
                            auto momt = ratio * ua + ub;
                            va = (momt - CR * ur) / m;
                            vb = (momt + CR * ratio * ur) / m;

                            if(isnan(va.norm()) || isnan(vb.norm())) {
                                printf("nan value detected : %f %f\n",(float)va.norm(),(float)vb.norm());
                            }
                        }else if(minv_b > (T)eps) {
                            // mb / ma
                            auto ratio = minv_a / minv_b;
                            auto m = ratio + (T)1;
                            auto momt = ua + ratio * ub;
                            va = (momt - CR * ratio * ur) / m;
                            vb = (momt + CR * ur) / m;
                            if(isnan(va.norm()) || isnan(vb.norm())) {
                                printf("nan value detected : %f %f\n",(float)va.norm(),(float)vb.norm());
                            }

                        }else {
                            printf("impossible reaching here\n");
                        }

                        auto ma = verts("m",vi);
                        auto mb = verts("m",nvi);

                        auto dva = (va - ua) * ma;
                        auto dvb = (vb - ub) * mb;

                        // auto dpa = (va - ua);
                        // auto dpb = (vb - ub);
                        // printf("find collision pair : %d %d\n",vi,nvi);
                        atomic_add(exec_tag,&impulse_count[vi],1);
                        atomic_add(exec_tag,&impulse_count[nvi],1);

                        for(int i = 0;i != 3;++i) {
                            atomic_add(exec_tag,&impulse_buffer[vi][i],dva[i]);
                            atomic_add(exec_tag,&impulse_buffer[nvi][i],dvb[i]);
                        }
                    }
                };

                spBvh.iter_neighbors(bv,find_particle_sphere_collision_pairs);
        });

        // apply the impulse
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            // xtag = zs::SmallString(xtag),
            relaxation_rate = relaxation_rate,
            vtag = zs::SmallString(vtag),
            exec_tag,
            // dpnorm = proxy<space>(dpnorm),
            impulse_count = proxy<space>(impulse_count),
            impulse_buffer = proxy<space>(impulse_buffer)] ZS_LAMBDA(int vi) mutable {
                if(impulse_count[vi] == 0)
                    return;
                auto minv = verts("minv",vi);
                auto dv = minv * impulse_buffer[vi] * relaxation_rate / (T)impulse_count[vi];
                // atomic_add(exec_tag,&dpnorm[0],dp.norm());
                verts.tuple(dim_c<3>,vtag,vi) = verts.pack(dim_c<3>,vtag,vi) + dv;
        });        

        set_output("zsparticles",zsparticles);
    }

};
                            
ZENDEFNODE(ExchangeLocalMomentum, {{{"zsparticles"},
                                // {"float","relaxation_strength","1"},
                                {"string","xtag","x"},
                                {"string","Xtag","X"},
                                // {"string","pxtag","px"},
                                {"string","vtag","v"},
                                {"string","pscaleTag","pscale"},
                                // {"float","thickness","0.0001"},
                                {"float","restitution","0.1"},
                                {"float","relaxation","1"},
                                // {"float","friction","0"}                            
                            },
                            {{"zsparticles"}},
                            {
                                // {"string","ptag","x"}
                            },
                            {"PBD"}});

struct VisualizePoints : INode {
    virtual void apply() override {
        using namespace zs;
        using bvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename bvh_t::Box;

        using vec3 = zs::vec<float,3>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};

        auto zsparticles = get_input<ZenoParticles>("zsparticles");  
        auto xtag = get_input2<std::string>("xtag");
        const auto& verts = zsparticles->getParticles();   
        
        zs::Vector<vec3> points{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            points = proxy<space>(points)] ZS_LAMBDA(int vi) mutable {
                points[vi] = verts.pack(dim_c<3>,xtag,vi);
            });

        points = points.clone({zs::memsrc_e::host});
        auto points_field = std::make_shared<zeno::PrimitiveObject>();
        points_field->resize(verts.size());
        auto& sverts = points_field->attr<zeno::vec3f>("pos");
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();
        
        ompPol(zs::range(verts.size()),[
            &sverts,points = proxy<omp_space>(points)] (int vi) mutable {
                sverts[vi] = points[vi].to_array();
        });
        
        set_output("points",std::move(points_field));
    }
};

ZENDEFNODE(VisualizePoints, {
    {"zsparticles",{"string","xtag","x"}},
    {"points"},
    {{"float","scale","1.0"}},
    {"ZSGeometry"},
});



struct XPBDParticlesCollider : INode {
    // particle sphere bvh
    template <typename TileVecT>
    void buildBvh(zs::CudaExecutionPolicy &pol,
            TileVecT &verts,
            const zs::SmallString& xtag,
            const zs::SmallString& pscaleTag,
            ZenoLinearBvh::lbvh_t& bvh) {
        using namespace zs;
        using bv_t = typename ZenoLinearBvh::lbvh_t::Box;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> bvs{verts.get_allocator(), verts.size()};

        pol(range(verts.size()),
            [verts = proxy<space>({}, verts),
             bvs = proxy<space>(bvs),
             pscaleTag,xtag] ZS_LAMBDA(int vi) mutable {
                auto pos = verts.template pack<3>(xtag, vi);
                auto pscale = verts(pscaleTag,vi);
                bv_t bv{pos - pscale,pos + pscale};
                bvs[vi] = bv;
            });
        bvh.build(pol, bvs);
    }

    virtual void apply() override {
        using namespace zs;
        using bvh_t = typename ZenoLinearBvh::lbvh_t;
        using bv_t = typename bvh_t::Box;

        using vec3 = zs::vec<float,3>;
        using vec2i = zs::vec<int,2>;
        using vec3i = zs::vec<int,3>;
        using vec4i = zs::vec<int,4>;
        using mat4 = zs::vec<int,4,4>;

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();
        constexpr auto exec_tag = wrapv<space>{};

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        // auto bvh = get_input<bvh_t>("lbvh");
        // prex
        auto xtag = get_input2<std::string>("xtag");
        auto pxtag = get_input2<std::string>("pxtag");
        auto Xtag = get_input2<std::string>("Xtag");
        // x
        auto ptag = get_input2<std::string>("pscaleTag");
        auto friction = get_input2<T>("friction");
        
        auto& verts = zsparticles->getParticles();     

        auto spBvh = bvh_t{};
        buildBvh(cudaPol,verts,xtag,ptag,spBvh);

        // auto collisionThickness = 

        zs::Vector<float> dp_buffer{verts.get_allocator(),verts.size() * 3};
        cudaPol(zs::range(dp_buffer),[]ZS_LAMBDA(auto& v) mutable {v = 0;});
        zs::Vector<int> dp_count{verts.get_allocator(),verts.size()};
        cudaPol(zs::range(dp_count),[]ZS_LAMBDA(auto& c) mutable {c = 0;});

        // find collision pairs
        zs::bht<int,2,int> csPT{verts.get_allocator(),100000};
        csPT.reset(cudaPol,true);

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            pxtag = zs::SmallString(pxtag),
            Xtag = zs::SmallString(Xtag),
            ptag = zs::SmallString(ptag),
            spBvh = proxy<space>(spBvh),
            csPT = proxy<space>(csPT)] ZS_LAMBDA(int vi) mutable {
                auto pos = verts.pack(dim_c<3>,xtag,vi);
                auto ppos = verts.pack(dim_c<3>,pxtag,vi);
                auto vel = pos - ppos;
                auto rpos = verts.pack(dim_c<3>,Xtag,vi);
                auto r = verts(ptag,vi);
                
                auto bv = bv_t{pos - r,pos + r};

                auto find_particle_sphere_collision_pairs = [&] (int nvi) {
                    if(vi >= nvi)
                        return;

                    auto npos = verts.pack(dim_c<3>,xtag,nvi);
                    // auto nppos = verts.pack(dim_c<3>,pxtag,nvi);
                    // auto nvel = npos - nppos;

                    auto nrpos = verts.pack(dim_c<3>,Xtag,nvi);
                    auto nr = verts(ptag,nvi);

                    auto rdist = (rpos - nrpos).norm();
                    if(rdist < r + nr)
                        return;

                    auto dist = (pos - npos).norm();
                    if(dist > r + nr)
                        return;
                    
                    // find a collision pairs
                    csPT.insert(zs::vec<int,2>{vi,nvi});
                };

                spBvh.iter_neighbors(bv,find_particle_sphere_collision_pairs);
        });

        // std::cout << "csPT.size() = " << csPT.size() << std::endl;

        auto w = get_input2<float>("relaxation_strength");
        // process collision pairs
        cudaPol(zip(zs::range(csPT.size()),csPT._activeKeys),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            ptag = zs::SmallString(ptag),
            pxtag = zs::SmallString(pxtag),
            dp_buffer = proxy<space>(dp_buffer),
            dp_count = proxy<space>(dp_count),
            exec_tag,
            friction] ZS_LAMBDA(auto,const auto& pair) mutable {
                auto vi = pair[0];
                auto nvi = pair[1];
                auto p0 = verts.pack(dim_c<3>,xtag,vi);
                auto p1 = verts.pack(dim_c<3>,xtag,nvi);
                auto vel0 = p0 - verts.pack(dim_c<3>,pxtag,vi);
                auto vel1 = p1 - verts.pack(dim_c<3>,pxtag,nvi);

                auto nrm = (p0 - p1).normalized();

                auto r = verts(ptag,vi) + verts(ptag,nvi);

                float minv0 = verts("minv",vi);
                float minv1 = verts("minv",nvi);

                vec3 dp0{},dp1{};
                if(!CONSTRAINT::solve_DistanceConstraint(
                    p0,minv0,
                    p1,minv1,
                    r,
                    (float)1,
                    dp0,dp1)) {
                    // auto rel_vel = vel0 - vel1 + dp0 - dp1;
                    // auto tan_vel = rel_vel - nrm * rel_vel.dot(nrm);
                    // auto tan_len = tan_vel.norm();
                    // auto max_tan_len = (dp0 - dp1).norm() * friction;

                    // if(tan_len > (T)1e-6) {
                    //     auto ratio = (T)max_tan_len / (T)tan_len;
                    //     auto alpha = zs::min(ratio,(T)1.0);

                    //     dp = -tan_vel * zs::min(alpha,(T)1.0);
                    //     pred += dp;
                    // }

                    for(int i = 0;i != 3;++i)
                        atomic_add(exec_tag,&dp_buffer[vi * 3 + i], dp0[i]);
                    for(int i = 0;i != 3;++i)
                        atomic_add(exec_tag,&dp_buffer[nvi * 3 + i],dp1[i]);

                    atomic_add(exec_tag,&dp_count[vi],(int)1);
                    atomic_add(exec_tag,&dp_count[nvi],(int)1);
                }
        });

        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            xtag = zs::SmallString(xtag),
            w,
            dp_count = proxy<space>(dp_count),
            dp_buffer = proxy<space>(dp_buffer)] ZS_LAMBDA(int vi) mutable {
                if(dp_count[vi] > 0) {
                    auto dp = vec3{dp_buffer[vi * 3 + 0],dp_buffer[vi * 3 + 1],dp_buffer[vi * 3 + 2]};
                    verts.tuple(dim_c<3>,xtag,vi) = verts.pack(dim_c<3>,xtag,vi) + w * dp / (T)dp_count[vi];
                }
        });

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(XPBDParticlesCollider, {{{"zsparticles"},
                                {"float","relaxation_strength","1"},
                                {"string","xtag","x"},
                                {"string","pxtag","px"},
                                {"string","Xtag","X"},
                                {"string","pscaleTag","pscale"},
                                {"float","friction","0"}                            
                            },
							{{"zsparticles"}},
							{
                                // {"string","ptag","x"}
                            },
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
                    if(category == category_c::isometric_bending_constraint) {
                        return;
                        auto quad = cquads.pack(dim_c<4>,"inds",ci,int_c);
                        vec3 p[4] = {};
                        float minv[4] = {};
                        for(int i = 0;i != 4;++i) {
                            p[i] = verts.pack(dim_c<3>,ptag,quad[i]);
                            minv[i] = verts("minv",quad[i]);
                        }

                        auto Q = cquads.pack(dim_c<4,4>,"Q",ci);

                        vec3 dp[4] = {};
                        float lambda = 0;
                        CONSTRAINT::solve_IsometricBendingConstraint(
                            p[0],minv[0],
                            p[1],minv[1],
                            p[2],minv[2],
                            p[3],minv[3],
                            Q,
                            (float)1,
                            dp[0],
                            dp[1],
                            dp[2],
                            dp[3]);

                        auto has_nan = false;
                        for(int i = 0;i != 4;++i)
                            if(zs::isnan(dp[i].norm()))
                                has_nan = true;
                        if(has_nan) {
                            printf("nan dp detected : %f %f %f %f %f %f %f %f\n",
                                (float)p[0].norm(),
                                (float)p[1].norm(),
                                (float)p[2].norm(),
                                (float)p[3].norm(),
                                (float)dp[0].norm(),
                                (float)dp[1].norm(),
                                (float)dp[2].norm(),
                                (float)dp[3].norm());
                            return;
                        }
                        for(int i = 0;i != 4;++i)
                            for(int j = 0;j != 3;++j)
                                atomic_add(exec_tag,&dp_buffer[quad[i] * 3 + j],dp[i][j]);
                        for(int i = 0;i != 4;++i)
                            atomic_add(exec_tag,&dp_count[quad[i]],(int)1);
                    }
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