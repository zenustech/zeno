#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/calculate_facet_normal.hpp"
#include "kernel/global_intersection_analysis.hpp"

namespace zeno {

struct MarkSelfIntersectionRegion : zeno::INode {

    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;
    using vec3 = zs::vec<T, 3>; 

    virtual void apply() override {
        using namespace zs;
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto& verts = zsparticles->getParticles();
        bool is_tet_volume_mesh = zsparticles->category == ZenoParticles::category_e::tet;
        const auto &tris = is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints(); 

        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();  

        dtiles_t tri_buffer{tris.get_allocator(),{
            {"inds",3},
            {"nrm",3},
            {"he_inds",1}
        },tris.size()};
        dtiles_t verts_buffer{verts.get_allocator(),{
            {"inds",1},
            {"x",3},
            {"he_inds",1}
        },is_tet_volume_mesh ? (*zsparticles)[ZenoParticles::s_surfVertTag].size() : verts.size()};

        TILEVEC_OPS::copy(cudaPol,tris,"he_inds",tri_buffer,"he_inds");
        if(is_tet_volume_mesh) {
            const auto &points = (*zsparticles)[ZenoParticles::s_surfVertTag];
            TILEVEC_OPS::copy(cudaPol,points,"inds",verts_buffer,"inds");
            TILEVEC_OPS::copy(cudaPol,points,"he_inds",verts_buffer,"he_inds");
            topological_sample(cudaPol,points,verts,"x",verts_buffer);
            TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
            reorder_topology(cudaPol,points,tri_buffer);

        }else {
            TILEVEC_OPS::copy(cudaPol,tris,"inds",tri_buffer,"inds");
            TILEVEC_OPS::copy(cudaPol,verts,"x",verts_buffer,"x");
            cudaPol(zs::range(verts.size()),[
                verts = proxy<cuda_space>({},verts),
                verts_buffer = proxy<cuda_space>({},verts_buffer)] ZS_LAMBDA(int vi) mutable {
                    verts_buffer("inds",vi) = reinterpret_bits<T>(vi);
            });
        }

        calculate_facet_normal(cudaPol,verts_buffer,"x",tri_buffer,tri_buffer,"nrm");

        dtiles_t inst_buffer_info{tris.get_allocator(),{
            {"pair",2},
            {"type",1},
            {"its_edge_mark",6},
            {"int_points",6}
        },tris.size() * 2};

        dtiles_t gia_res{verts_buffer.get_allocator(),{
            {"ring_mask",1},
            {"type_mask",1},
            {"color_mask",1},
            {"is_loop_vertex",1}
        },verts_buffer.size()};

        dtiles_t tris_gia_res{tri_buffer.get_allocator(),{
            {"ring_mask",1},
            {"type_mask",1},
            {"color_mask",1},
        },tri_buffer.size()};

        auto& halfedges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
        auto ring_mask_width = GIA::do_global_self_intersection_analysis(cudaPol,
            verts_buffer,"x",tri_buffer,halfedges,gia_res,tris_gia_res);    

        auto markTag = get_input2<std::string>("markTag");

        if(!verts.hasProperty("markTag")) {
            verts.append_channels(cudaPol,{{markTag,1}});
        }
        TILEVEC_OPS::fill(cudaPol,verts,markTag,(T)0.0);
        cudaPol(zs::range(verts_buffer.size()),[
            gia_res = proxy<cuda_space>({},gia_res),
            verts = proxy<cuda_space>({},verts),
            ring_mask_width = ring_mask_width,
            verts_buffer = proxy<cuda_space>({},verts_buffer),
            markTag = zs::SmallString(markTag)
        ] ZS_LAMBDA(int pi) mutable {
            auto vi = zs::reinterpret_bits<int>(verts_buffer("inds",pi));
            int ring_mask = 0;
            for(int i = 0;i != ring_mask_width;++i) {
                ring_mask |= zs::reinterpret_bits<int>(gia_res("ring_mask",pi * ring_mask_width + i));
            }
            verts(markTag,vi) = ring_mask == 0 ? (T)0.0 : (T)1.0;
        });
        set_output("zsparticles",zsparticles);
    } 

};

ZENDEFNODE(MarkSelfIntersectionRegion, {{{"zsparticles"},{gParamType_String,"markTag","markTag"}},
                            {{"zsparticles"}},
                            {
                                
                            },
                            {"ZSGeometry"}});

};