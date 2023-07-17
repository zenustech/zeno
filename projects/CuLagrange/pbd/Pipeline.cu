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

        auto type = get_param<std::string>("topo_type");

        if(source->category != ZenoParticles::surface)
            throw std::runtime_error("Try adding Constraint topology to non-surface ZenoParticles");

        const auto& verts = source->getParticles();
        const auto& quads = source->getQuadraturePoints();

        auto uniform_stiffness = get_input2<float>("stiffness");


        if(type == "stretch") {
            auto quads_vec = tilevec_topo_to_zsvec_topo(cudaPol,quads,wrapv<3>{});
            zs::Vector<zs::vec<int,2>> edge_topos{quads.get_allocator(),0};
            retrieve_edges_topology(cudaPol,quads_vec,edge_topos);
            // std::cout << "quads.size() = " << quads.size() << "\t" << "edge_topos.size() = " << edge_topos.size() << std::endl;
            
            constraint->category = ZenoParticles::curve;
            constraint->sprayedOffset = 0;
            constraint->elements = typename ZenoParticles::particles_t({{"inds",2},{"r",1},{"stiffness",1},{"lambda",1}}, edge_topos.size(), zs::memsrc_e::device,0);
            auto &eles = constraint->getQuadraturePoints();
            cudaPol(zs::range(eles.size()),[
                verts = proxy<space>({},verts),
                eles = proxy<space>({},eles),
                uniform_stiffness = uniform_stiffness,
                edge_topos = proxy<space>(edge_topos)] ZS_LAMBDA(int ei) mutable {
                    eles.tuple(dim_c<2>,"inds",ei) = edge_topos[ei].reinterpret_bits(float_c);
                    vec3 x[2] = {};
                    for(int i = 0;i != 2;++i)
                        x[i] = verts.pack(dim_c<3>,"x",edge_topos[ei][i]);
                    eles("r",ei) = (x[0] - x[1]).norm();
                    eles("stiffness",1) = uniform_stiffness;
                    eles("lambda",1) = .0f;
            });
        }

        if(type == "iso_bending") {
            constraint->category = ZenoParticles::tri_bending_spring;
            constraint->sprayedOffset = 0;

            const auto& halfedges = (*source)[ZenoParticles::s_surfHalfEdgeTag];

            zs::Vector<zs::vec<int,4>> bd_topos{quads.get_allocator(),0};
            retrieve_tri_bending_topology(cudaPol,quads,halfedges,bd_topos);

            // std::cout << "halfedges.size() = " << halfedges.size() << "\t" << "bd_topos.size() = " << bd_topos.size() << std::endl;

            constraint->elements = typename ZenoParticles::particles_t({{"inds",4},{"Q", 4 * 4},{"stiffness",1},{"lambda",1}},bd_topos.size(),zs::memsrc_e::device,0);
            auto& eles = constraint->getQuadraturePoints();

            cudaPol(zs::range(eles.size()),[
                eles = proxy<space>({},eles),
                uniform_stiffness = uniform_stiffness,
                bd_topos = proxy<space>(bd_topos),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ei) mutable {
                    eles.tuple(dim_c<4>,"inds",ei) = bd_topos[ei].reinterpret_bits(float_c);
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

                    eles.tuple(dim_c<16>,"Q",ei) = Q;
                    eles("stiffness",1) = uniform_stiffness;
                    eles("lambda",1) = .0f;
            });
        }

        set_output("source",source);
        set_output("constraint",constraint);
    };
};

ZENDEFNODE(MakeSurfaceConstraintTopology, {{{"source"},{"float","stiffness","0.5"}},
							{{"constraint"}},
							{
                                {"enum stretch iso_bending","topo_type","stretch"},
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

struct XPBDSolve : INode {

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto zsparticles = get_input("zsparticles");
        auto constraints = get_input("constraints");   

        set_output("constraints",constraints);
        set_output("zsparticles",zsparticles);
    };
};

ZENDEFNODE(XPBDSolve, {{{"zsparticles"},{"constraints"}},
							{{"zsparticles"},{"constraints"}},
							{},
							{"PBD"}});

};
