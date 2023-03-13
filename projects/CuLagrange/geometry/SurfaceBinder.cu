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
#include "Utils.hpp"

#include "kernel/tiled_vector_ops.hpp"
#include "kernel/geo_math.hpp"

#include "zensim/container/Bvh.hpp"
#include "zensim/container/Bvs.hpp"
#include "zensim/container/Bvtt.hpp"



namespace zeno {

struct ZSSurfaceBind : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec2 = zs::vec<T,2>;
    using vec3 = zs::vec<T, 3>;
    using mat3 = zs::vec<T, 3, 3>;
    using mat9 = zs::vec<T,9,9>;
    using mat12 = zs::vec<T,12,12>;

    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;

    using pair3_t = zs::vec<Ti,3>;
    using pair4_t = zs::vec<Ti,4>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zssurf = get_input<ZenoParticles>("zssurf");
        auto kboundary = get_input<ZenoParticles>("kboundary");
        auto &verts = zssurf->getParticles();
        auto &tris = zssurf->getQuadraturePoints();
        const auto& kb_verts = kboundary->getParticles();

        dtiles_t kverts{kb_verts.get_allocator(),
            {
                {"x",3},
                {"inds",1},
                {"nrm",3},
                {"tag",1}
            },kb_verts.size()};
        TILEVEC_OPS::copy(cudaPol,kb_verts,"x",kverts,"x");
        TILEVEC_OPS::copy(cudaPol,kb_verts,"nrm",kverts,"nrm");
        if(kb_verts.hasProperty("tag"))
            TILEVEC_OPS::copy(cudaPol,kb_verts,"tag",kverts,"tag");
        else
            TILEVEC_OPS::fill(cudaPol,kverts,"tag",(T)0.0);
        cudaPol(zs::range(kverts.size()),
            [kverts = proxy<space>({},kverts)] ZS_LAMBDA(int vi) mutable {
                kverts("inds",vi) = reinterpret_bits<T>(vi);
        }); 

        auto max_nm_binders = get_param<int>("max_nm_binders");
        auto binder_tag = get_param<std::string>("binder_tag");
        auto thickness_tag = get_param<std::string>("thickness_tag");
        auto inversion_tag = get_param<std::string>("inversion_tag");

        tris.append_channels(cudaPol,{
            {binder_tag,max_nm_binders},
            {thickness_tag,max_nm_binders},
            {inversion_tag,max_nm_binders},
        });
        TILEVEC_OPS::fill(cudaPol,tris,binder_tag,zs::reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,tris,thickness_tag,(T)0.0);
        TILEVEC_OPS::fill(cudaPol,tris,inversion_tag,(T)-1.0);

        auto kpBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,kverts,kverts,wrapv<1>{},(T)0.0,"x");
        kpBvh.build(cudaPol,bvs);

        auto kinInCollisionEps = get_input2<float>("kinInColEps");
        auto kinOutCollisionEps = get_input2<float>("kinOutColEps");
        auto thickness = kinInCollisionEps + kinOutCollisionEps;

        // compuate normal
        if(!tris.hasProperty("nrm"))
            tris.append_channels(cudaPol,{{"nrm",3}});
        cudaPol(zs::range(tris.size()),
            [tris = proxy<space>({},tris),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) {
            auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
            auto v0 = verts.template pack<3>("x",tri[0]);
            auto v1 = verts.template pack<3>("x",tri[1]);
            auto v2 = verts.template pack<3>("x",tri[2]);

            auto e01 = v1 - v0;
            auto e02 = v2 - v0;

            auto nrm = e01.cross(e02);
            auto nrm_norm = nrm.norm();
            if(nrm_norm < 1e-8)
                nrm = zs::vec<T,3>::zeros();
            else
                nrm = nrm / nrm_norm;

            tris.tuple(dim_c<3>,"nrm",ti) = nrm;
        });        

        cudaPol(zs::range(tris.size()),
            [tris = proxy<space>({},tris),
                verts = proxy<space>({},verts),
                kverts = proxy<space>({},kverts),
                kpBvh = proxy<space>(kpBvh),
                binder_tag = zs::SmallString(binder_tag),
                thickness_tag = zs::SmallString(thickness_tag),
                inversion_tag = zs::SmallString(inversion_tag),
                max_nm_binders = max_nm_binders,
                kinInCollisionEps = kinInCollisionEps,
                kinOutCollisionEps = kinOutCollisionEps,
                thickness = thickness] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            auto p = vec3::zeros();
            for(int i = 0;i != 3;++i)
                p += verts.pack(dim_c<3>,"x",tri[i])/(T)3.0;
            auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};

            // printf("testing tri[%d] : %f %f %f\n",ti,(float)p[0],(float)p[1],(float)p[2]);

            int nm_binders = 0;
            int nm_tag = 0;
            auto binder_tags_vec = zs::vec<T,16>::uniform((T)-1.0);
            auto process_vertex_facet_binding_pairs = [&](int kpi) {
                // printf("testing %d tri and %d kp\n",ti,kpi);
                if(nm_binders >= max_nm_binders)
                    return;
                auto kp = kverts.pack(dim_c<3>,"x",kpi);
                auto seg = p - kp;

                auto t0 = verts.pack(dim_c<3>,"x",tri[0]);
                auto t1 = verts.pack(dim_c<3>,"x",tri[1]);
                auto t2 = verts.pack(dim_c<3>,"x",tri[2]);

                T barySum = (T)0.0;
                T distance = LSL_GEO::pointTriangleDistance(t0,t1,t2,kp,barySum);

                auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                auto knrm = kverts.pack(dim_c<3>,"nrm",kpi);
                if(nrm.dot(knrm) < (T)0.5)
                    return;
                auto dist = seg.dot(nrm);

                auto collisionEps = dist < 0 ? kinOutCollisionEps : kinInCollisionEps;

                if(distance > collisionEps)
                    return;

                auto ntris = tris.pack(dim_c<3>,"ff_inds",ti).reinterpret_bits(int_c);
                for(int i = 0;i != 3;++i){
                    auto nti = ntris[i];
                    if(nti < 0){
                        printf("negative ff_inds detected\n");
                        return;
                    }
                    auto edge_normal = tris.pack(dim_c<3>,"nrm",ti) + tris.pack(dim_c<3>,"nrm",nti);
                    edge_normal = (edge_normal)/(edge_normal.norm() + (T)1e-6);
                    auto e0 = verts.pack(dim_c<3>,"x",tri[(i + 0) % 3]);
                    auto e1 = verts.pack(dim_c<3>,"x",tri[(i + 1) % 3]);  
                    auto e10 = e1 - e0;
                    auto bisector_normal = edge_normal.cross(e10).normalized();

                    seg = kp - verts.pack(dim_c<3>,"x",tri[i]);
                    if(bisector_normal.dot(seg) < 0)
                        return;
                }
                // printf("bind tri[%d] to kp[%d]\n",ti,kpi);
                binder_tags_vec[nm_binders] = kverts("tag",kpi);
                bool new_tag = true;
                for(int i = 0;i != nm_binders;++i)
                    if(zs::abs(binder_tags_vec[i] - kverts("tag",kpi)) < 1e-4)
                        new_tag = false;
                if(new_tag)
                    nm_tag++;

                tris(binder_tag,nm_binders,ti) = reinterpret_bits<T>(kpi);
                tris(thickness_tag,nm_binders,ti) = distance;
                tris(inversion_tag,nm_binders,ti) = dist < 0 ? (T)1.0 : (T)-1.0;
                nm_binders++;
            };
            kpBvh.iter_neighbors(bv,process_vertex_facet_binding_pairs);

            if(nm_tag > 1)
                for(int i = 0;i != nm_binders;++i)
                    tris(binder_tag,i,ti) = reinterpret_bits<T>((int)-1);
        });


        set_output("zssurf",zssurf);
    }
};


ZENDEFNODE(ZSSurfaceBind, {{"zssurf","kboundary",
                                    {"float","kinInColEps","0.01"},
                                    {"float","kinOutColEps","0.02"}
                                    },
                                  {"zssurf"},
                                  {
                                    {"int","max_nm_binders","4"},
                                    {"string","binder_tag","binderTag"},
                                    {"string","thickness_tag","thicknessTag"},
                                    {"string","inversion_tag","inversionTag"}
                                  },
                                  {"ZSGeometry"}});

struct VisualizeSurfaceBinder : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec2 = zs::vec<T,2>;
    using vec3 = zs::vec<T, 3>;
    using mat3 = zs::vec<T, 3, 3>;
    using mat9 = zs::vec<T,9,9>;
    using mat12 = zs::vec<T,12,12>;

    using bvh_t = zs::LBvh<3,int,T>;
    using bv_t = zs::AABBBox<3, T>;

    using pair3_t = zs::vec<Ti,3>;
    using pair4_t = zs::vec<Ti,4>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();

        auto zssurf = get_input<ZenoParticles>("zssurf");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        const auto& verts = zssurf->getParticles();
        const auto& tris = zssurf->getQuadraturePoints();

        const auto& kb_verts = kboundary->getParticles();
        auto binder_tag = get_param<std::string>("binder_tag");
        auto max_nm_binders = tris.getChannelSize(binder_tag);

        dtiles_t tverts_buffer{tris.get_allocator(),
            {
                {"x",3},
                {"binder_inds",max_nm_binders}
        },tris.size()};
        dtiles_t kverts_buffer{kb_verts.get_allocator(),
            {
                {"x",3}
        },kb_verts.size()};

        cudaPol(zs::range(tris.size()),
            [tris = proxy<cuda_space>({},tris),
                verts = proxy<cuda_space>({},verts),
                tverts_buffer = proxy<cuda_space>({},tverts_buffer)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            tverts_buffer.tuple(dim_c<3>,"x",ti) = vec3::zeros();
            for(int i = 0;i != 3;++i)
                tverts_buffer.tuple(dim_c<3>,"x",ti) = tverts_buffer.pack(dim_c<3>,"x",ti) + verts.pack(dim_c<3>,"x",tri[i])/(T)3.0;
        });
        TILEVEC_OPS::copy(cudaPol,tris,binder_tag,tverts_buffer,"binder_inds");
        TILEVEC_OPS::copy(cudaPol,kb_verts,"x",kverts_buffer,"x");

        tverts_buffer = tverts_buffer.clone({zs::memsrc_e::host});
        kverts_buffer = kverts_buffer.clone({zs::memsrc_e::host});

        auto binder_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& binder_verts = binder_vis->verts;
        auto& binder_lines = binder_vis->lines;
        binder_verts.resize(tverts_buffer.size() * (max_nm_binders + 1));
        binder_lines.resize(tverts_buffer.size() * max_nm_binders);

        ompPol(zs::range(tverts_buffer.size()),
            [tverts_buffer = proxy<omp_space>({},tverts_buffer),
                kverts_buffer = proxy<omp_space>({},kverts_buffer),
                &binder_verts,&binder_lines,max_nm_binders = max_nm_binders] (int ti) mutable {
            binder_verts[ti * (max_nm_binders + 1) + 0] = tverts_buffer.pack(dim_c<3>,"x",ti).to_array();
            for(int i = 0;i != max_nm_binders;++i){
                auto idx = reinterpret_bits<int>(tverts_buffer("binder_inds",i,ti));
                if(idx < 0){
                    binder_verts[ti * (max_nm_binders + 1) + i + 1] = tverts_buffer.pack(dim_c<3>,"x",ti).to_array();
                }else{
                    binder_verts[ti * (max_nm_binders + 1) + i + 1] = kverts_buffer.pack(dim_c<3>,"x",idx).to_array();
                }
                binder_lines[ti * max_nm_binders + i] = zeno::vec2i{ti * (max_nm_binders + 1) + 0,ti * (max_nm_binders + 1) + i + 1};
            }
        });

        set_output("binder_vis",std::move(binder_vis));
    }
};

ZENDEFNODE(VisualizeSurfaceBinder, {{"zssurf","kboundary"},
                                  {"binder_vis"},
                                  {
                                    {"string","binder_tag","binderTag"}
                                  },
                                  {"ZSGeometry"}});

};