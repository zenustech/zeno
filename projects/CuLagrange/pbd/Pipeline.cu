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
#include "../geometry/kernel/topology.hpp"
#include "../geometry/kernel/geo_math.hpp"

namespace zeno {

// we only need to record the topo here
// serve triangulate mesh or strands only currently
struct MakeSurfaceConstraintTopology : INode {

    virtual void apply() override {
        using namespace zs;
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
            constraint->category = ZenoParticles::curve;

            auto quads_vec = tilevec_topo_to_zsvec_topo(cudaPol,quads,wrapv<3>{});
            zs::Vector<zs::vec<int,2>> edge_topos{quads.get_allocator(),0};
            retrieve_edges_topology(cudaPol,quads_vec,edge_topos);
            // zs::Vector<zs::vec<int,4>> edge_topos{quads.get_allocator(),edge_topos_broken.size()};
            // cudaPol(zs::range(edge_topos.size()),[
            //     edge_topos = proxy<space>(edge_topos),
            //     edge_topos_broken = proxy<space>(edge_topos_broken)] ZS_LAMBDA(int ei) mutable {
            //         edge_topos[ei][0] = edge_topos_broken[ei][0];
            //         edge_topos[ei][1] = edge_topos_broken[ei][1];
            //         edge_topos[ei][2] = -1;
            //         edge_topos[ei][3] = -1;
            // });


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
                    // auto edge_full = edge_topos[ei];
                    // vec2i edge{edge_full[0],edge_full[1]};

                    eles.tuple(dim_c<2>,"inds",oei) = edge_topos[ei].reinterpret_bits(float_c);
                    vec3 x[2] = {};
                    for(int i = 0;i != 2;++i)
                        x[i] = verts.pack(dim_c<3>,"x",edge_topos[ei][i]);
                    eles("r",oei) = (x[0] - x[1]).norm() * rest_scale;
            });            

        }

        if(type == "bending") {
            constraint->category = ZenoParticles::tri_bending_spring;
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
                    eles.tuple(dim_c<4>,"inds",oei) = bd_topos[ei].reinterpret_bits(float_c);
                    vec3 x[4] = {};
                    for(int i = 0;i != 4;++i)
                        x[i] = verts.pack(dim_c<3>,"x",bd_topos[ei][i]);

                    auto e0 = x[1] - x[0];
                    auto e1 = x[2] - x[0];
                    auto e2 = x[3] - x[0];
                    auto e3 = x[2] - x[1];
                    auto e4 = x[3] - x[1];

                    auto c01 = LSL_GEO::cotTheta(e0, e1);
                    auto c02 = LSL_GEO::cotTheta(e0, e2);
                    auto c03 = LSL_GEO::cotTheta(-e0, e3);
                    auto c04 = LSL_GEO::cotTheta(-e0, e4);

                    auto A0 = 0.5f * (e0.cross(e1)).norm();
                    auto A1 = 0.5f * (e0.cross(e2)).norm();

                    auto coef = -3.f / (2.f * (A0 + A1));
                    float K[4] = { c03 + c04, c01 + c02, -c01 - c03, -c02 - c04 };
                    float K2[4] = { coef * K[0], coef * K[1], coef * K[2], coef * K[3] };

                    mat4 Q = mat4::uniform(0);

                    for (unsigned char j = 0; j < 4; j++)
                    {
                        for (unsigned char k = 0; k < j; k++)
                        {
                            Q(j, k) = Q(k, j) = K[j] * K2[k];
                        }
                        Q(j, j) = K[j] * K2[j];
                    }

                    eles.tuple(dim_c<16>,"Q",oei) = Q;
            });
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

        set_output("source",source);
        set_output("constraint",constraint);
    };
};

ZENDEFNODE(MakeSurfaceConstraintTopology, {{
                                {"source"},
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

        if(constraints_ptr->category == ZenoParticles::curve) {
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
        }else if(constraints_ptr->category == ZenoParticles::tri_bending_spring) {
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

                pverts[ci * 2 + 0] = (cverts[0] + cverts[1] + cverts[2]) / (T)3.0;
                pverts[ci * 2 + 1] = (cverts[0] + cverts[1] + cverts[3]) / (T)3.0;
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
        auto category = constraints->category;

        // zs::Vector<vec3> pv_buffer{verts.get_allocator(),verts.size()};
        // zs::Vector<float> total_ghost_impulse_X{verts.get_allocator(),1};
        // zs::Vector<float> total_ghost_impulse_Y{verts.get_allocator(),1};
        // zs::Vector<float> total_ghost_impulse_Z{verts.get_allocator(),1};

        for(int g = 0;g != nm_group;++g) {
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
                cquads = proxy<space>({},cquads)] ZS_LAMBDA(int gi) mutable {
                    float s = cquads("stiffness",coffset + gi);
                    float lambda = cquads("lambda",coffset + gi);

                    if(category == ZenoParticles::curve) {
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
                    if(category == ZenoParticles::tri_bending_spring) {
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

                        for(int i = 0;i != 4;++i)
                            verts.tuple(dim_c<3>,ptag,quad[i]) = p[i] + dp[i];
                    }
                    cquads("lambda",coffset + gi) = lambda;
            });

            // total_ghost_impulse_X.setVal(0);
            // total_ghost_impulse_Y.setVal(0);
            // total_ghost_impulse_Z.setVal(0);
            // cudaPol(zs::range(verts.size()),[
            //     verts = proxy<space>({},verts),
            //     ptag = zs::SmallString(ptag),
            //     exec_tag,
            //     total_ghost_impulse_X = proxy<space>(total_ghost_impulse_X),
            //     total_ghost_impulse_Y = proxy<space>(total_ghost_impulse_Y),
            //     total_ghost_impulse_Z = proxy<space>(total_ghost_impulse_Z),
            //     pv_buffer = proxy<space>(pv_buffer)] ZS_LAMBDA(int vi) mutable {
            //         auto cv = verts.pack(dim_c<3>,ptag,vi);
            //         auto pv = pv_buffer[vi];
            //         auto m = verts("m",vi);
            //         auto dv = m * (cv - pv);
            //         // for(int i = 0;i != 3;++i)
            //         atomic_add(exec_tag,&total_ghost_impulse_X[0],dv[0]);
            //         atomic_add(exec_tag,&total_ghost_impulse_Y[0],dv[1]);
            //         atomic_add(exec_tag,&total_ghost_impulse_Z[0],dv[2]);
            // });

            // auto tgi = total_ghost_impulse.getVal(0);
            // auto tgx = total_ghost_impulse_X.getVal(0);
            // auto tgy = total_ghost_impulse_Y.getVal(0);
            // auto tgz = total_ghost_impulse_Z.getVal(0);
            // printf("ghost_impulse[%d][%d] : %f %f %f\n",(int)coffset,(int)group_size,(float)tgx,(float)tgy,(float)tgz);
        }      

        set_output("constraints",constraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolve, {{{"zsparticles"},{"constraints"},{"float","dt","0.5"}},
							{{"zsparticles"},{"constraints"}},
							{{"string","ptag","X"}},
							{"PBD"}});

struct XPBDSolveSmooth : INode {

    virtual void apply() override {
        using namespace zs;
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
        auto category = constraints->category;

        // zs::Vector<vec3> pv_buffer{verts.get_allocator(),verts.size()};
        // zs::Vector<float> total_ghost_impulse_X{verts.get_allocator(),1};
        // zs::Vector<float> total_ghost_impulse_Y{verts.get_allocator(),1};
        // zs::Vector<float> total_ghost_impulse_Z{verts.get_allocator(),1};

        for(int g = 0;g != nm_group;++g) {
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
                cquads = proxy<space>({},cquads)] ZS_LAMBDA(int gi) mutable {
                    float s = cquads("stiffness",coffset + gi);
                    float lambda = cquads("lambda",coffset + gi);

                    if(category == ZenoParticles::curve) {
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
                    if(category == ZenoParticles::tri_bending_spring) {
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

                        for(int i = 0;i != 4;++i)
                            verts.tuple(dim_c<3>,ptag,quad[i]) = p[i] + dp[i];
                    }
                    cquads("lambda",coffset + gi) = lambda;
            });

            // total_ghost_impulse_X.setVal(0);
            // total_ghost_impulse_Y.setVal(0);
            // total_ghost_impulse_Z.setVal(0);
            // cudaPol(zs::range(verts.size()),[
            //     verts = proxy<space>({},verts),
            //     ptag = zs::SmallString(ptag),
            //     exec_tag,
            //     total_ghost_impulse_X = proxy<space>(total_ghost_impulse_X),
            //     total_ghost_impulse_Y = proxy<space>(total_ghost_impulse_Y),
            //     total_ghost_impulse_Z = proxy<space>(total_ghost_impulse_Z),
            //     pv_buffer = proxy<space>(pv_buffer)] ZS_LAMBDA(int vi) mutable {
            //         auto cv = verts.pack(dim_c<3>,ptag,vi);
            //         auto pv = pv_buffer[vi];
            //         auto m = verts("m",vi);
            //         auto dv = m * (cv - pv);
            //         // for(int i = 0;i != 3;++i)
            //         atomic_add(exec_tag,&total_ghost_impulse_X[0],dv[0]);
            //         atomic_add(exec_tag,&total_ghost_impulse_Y[0],dv[1]);
            //         atomic_add(exec_tag,&total_ghost_impulse_Z[0],dv[2]);
            // });

            // auto tgi = total_ghost_impulse.getVal(0);
            // auto tgx = total_ghost_impulse_X.getVal(0);
            // auto tgy = total_ghost_impulse_Y.getVal(0);
            // auto tgz = total_ghost_impulse_Z.getVal(0);
            // printf("ghost_impulse[%d][%d] : %f %f %f\n",(int)coffset,(int)group_size,(float)tgx,(float)tgy,(float)tgz);
        }      

        set_output("constraints",constraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolveSmooth, {{{"zsparticles"},{"constraints"},{"float","dt","0.5"}},
							{{"zsparticles"},{"constraints"}},
							{{"string","ptag","X"}},
							{"PBD"}});

};
