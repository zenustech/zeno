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

#include "../fem/collision_energy/collision_utils.hpp"

#include "kernel/compute_characteristic_length.hpp"

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
        auto &tris = zssurf->category == ZenoParticles::category_e::tet ? (*zssurf)[ZenoParticles::s_surfTriTag] : zssurf->getQuadraturePoints();

        auto markTag = get_param<std::string>("mark_tag");
        auto& kb_verts = kboundary->getParticles();
        if(!kb_verts.hasProperty(markTag))
            kb_verts.append_channels(cudaPol,{{markTag,1}});
        TILEVEC_OPS::fill(cudaPol,kb_verts,markTag,(T)0.0);


        dtiles_t kverts{kb_verts.get_allocator(),
            {
                {"x",3},
                {"inds",1},
                {"nrm",3},
                {"tag",1},
                {"binderFailTag",1}
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
        if(kb_verts.hasProperty("binderFailTag"))
            TILEVEC_OPS::copy(cudaPol,kb_verts,"binderFailTag",kverts,"binderFailTag");
        else
            TILEVEC_OPS::fill(cudaPol,kverts,"binderFailTag",(T)0.0);            

        auto max_nm_binders = get_param<int>("max_nm_binders");
        auto binder_tag = get_param<std::string>("binder_tag");
        auto thickness_tag = get_param<std::string>("thickness_tag");
        auto inversion_tag = get_param<std::string>("inversion_tag");
        auto align_direction = get_param<bool>("align_direction");


        tris.append_channels(cudaPol,{
            {binder_tag,max_nm_binders},
            {thickness_tag,max_nm_binders},
            {inversion_tag,max_nm_binders},
            {"nm_binders",1}
        });
        TILEVEC_OPS::fill(cudaPol,tris,binder_tag,zs::reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,tris,thickness_tag,(T)0.0);
        TILEVEC_OPS::fill(cudaPol,tris,inversion_tag,(T)-1.0);
        TILEVEC_OPS::fill(cudaPol,tris,"nm_binders",reinterpret_bits<T>((int)0));

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
                kb_verts = proxy<space>({},kb_verts),
                kverts = proxy<space>({},kverts),
                kpBvh = proxy<space>(kpBvh),
                markTag = zs::SmallString(markTag),
                binder_tag = zs::SmallString(binder_tag),
                thickness_tag = zs::SmallString(thickness_tag),
                inversion_tag = zs::SmallString(inversion_tag),
                max_nm_binders = max_nm_binders,
                kinInCollisionEps = kinInCollisionEps,
                kinOutCollisionEps = kinOutCollisionEps,
                align_direction = align_direction,
                thickness = thickness] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            auto p = vec3::zeros();
            for(int i = 0;i != 3;++i)
                p += verts.pack(dim_c<3>,"x",tri[i])/(T)3.0;
            auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};

            if(verts.hasProperty("binderFailTag")){
                for(int i = 0;i != 3;++i)
                    if(verts("binderFailTag",tri[i]) > (T)0.5)
                        return;
            }

            // printf("testing tri[%d] : %f %f %f\n",ti,(float)p[0],(float)p[1],(float)p[2]);

            int nm_binders = 0;
            int nm_tag = 0;
            auto binder_tags_vec = zs::vec<T,16>::uniform((T)-1.0);
            auto process_vertex_facet_binding_pairs = [&](int kpi) {
                // printf("testing %d tri and %d kp\n",ti,kpi);
                // if(kverts.hasProperty("binderFailTag"))
                if(kverts("binderFailTag",kpi) > (T)0.5)
                    return;
                if(nm_binders >= max_nm_binders)
                    return;
                auto kp = kverts.pack(dim_c<3>,"x",kpi);
                auto seg = kp - p;

                auto t0 = verts.pack(dim_c<3>,"x",tri[0]);
                auto t1 = verts.pack(dim_c<3>,"x",tri[1]);
                auto t2 = verts.pack(dim_c<3>,"x",tri[2]);

                T barySum = (T)0.0;
                T distance = LSL_GEO::pointTriangleDistance(t0,t1,t2,kp,barySum);

                auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                auto knrm = kverts.pack(dim_c<3>,"nrm",kpi);
                // alignment

                if(nrm.dot(knrm) < (T)0.67 && align_direction)
                    return;
                if(nrm.dot(knrm) > (T)-0.67 && !align_direction)
                    return;

                auto dist = seg.dot(nrm);

                auto collisionEps = dist > 0 ? kinOutCollisionEps : kinInCollisionEps;
                if(distance < 1e-4)
                    return;

                if(distance > collisionEps)
                    return;

                auto ntris = tris.pack(dim_c<3>,"ff_inds",ti).reinterpret_bits(int_c);
                for(int i = 0;i != 3;++i){
                    auto nti = ntris[i];
                    if(nti < 0){
                        // printf("negative ff_inds detected\n");
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
                kb_verts(markTag,kpi) = (T)1.0;
            };
            kpBvh.iter_neighbors(bv,process_vertex_facet_binding_pairs);

            if(nm_tag > 1)
                for(int i = 0;i != nm_binders;++i)
                    tris(binder_tag,i,ti) = reinterpret_bits<T>((int)-1);
            tris("nm_binders",ti) = reinterpret_bits<T>(nm_tag);
        });


        set_output("zssurf",zssurf);
        set_output("kboundary",kboundary);
    }
};


ZENDEFNODE(ZSSurfaceBind, {{"zssurf","kboundary",
                                    {"float","kinInColEps","0.01"},
                                    {"float","kinOutColEps","0.02"}
                                    },
                                  {"zssurf","kboundary"},
                                  {
                                    {"int","max_nm_binders","4"},
                                    {"string","binder_tag","binderTag"},
                                    {"string","thickness_tag","thicknessTag"},
                                    {"string","inversion_tag","inversionTag"},
                                    {"string","mark_tag","markTag"},
                                    {"bool","align_direction","1"},
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
        // const auto& tris = zssurf->getQuadraturePoints();
        const auto &tris = zssurf->category == ZenoParticles::category_e::tet ? (*zssurf)[ZenoParticles::s_surfTriTag] : zssurf->getQuadraturePoints();

        const auto& kb_verts = kboundary->getParticles();
        auto binder_tag = get_param<std::string>("binder_tag");
        auto scale = get_param<float>("scale");
        auto max_nm_binders = tris.getChannelSize(binder_tag);

        dtiles_t tverts_buffer{tris.get_allocator(),
            {
                {"x",3},
                {"xk",3}
        },tris.size() * max_nm_binders};
        // dtiles_t kverts_buffer{kb_verts.get_allocator(),{max_nm_binders
        //         {"x",3}
        // },kb_verts.size()};

        // TILEVEC_OPS::copy(cudaPol,tris,binder_tag,tverts_buffer,"binder_inds");
        // TILEVEC_OPS::copy(cudaPol,kb_verts,"x",kverts_buffer,"x");

        cudaPol(zs::range(tris.size()),
            [tris = proxy<cuda_space>({},tris),
                kb_verts = proxy<cuda_space>({},kb_verts),
                verts = proxy<cuda_space>({},verts),max_nm_binders = max_nm_binders,
                binder_tag = zs::SmallString(binder_tag),
                // kverts_buffer = proxy<cuda_space>({},kverts_buffer),
                tverts_buffer = proxy<cuda_space>({},tverts_buffer)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            for(int i = 0;i != max_nm_binders;++i){
                // tverts_buffer.tuple(dim_c<3>,"x",ti * max_nm_binders + i) = vec3::zeros();
                auto idx = reinterpret_bits<int>(tris(binder_tag,i,ti));
                if(idx < 0) {
                    tverts_buffer.tuple(dim_c<3>,"x",ti * max_nm_binders + i) = vec3::zeros();
                    for(int j = 0;j != 3;++j)
                        tverts_buffer.tuple(dim_c<3>,"x",ti * max_nm_binders + i) = tverts_buffer.pack(dim_c<3>,"x",ti * max_nm_binders + i) + verts.pack(dim_c<3>,"x",tri[j])/(T)3.0;
                    tverts_buffer.tuple(dim_c<3>,"xk",ti * max_nm_binders + i) = tverts_buffer.pack(dim_c<3>,"x",ti * max_nm_binders + i);
                }else {
                        vec3 cp[4] = {};
                        cp[0] = kb_verts.pack(dim_c<3>,"x",idx);
                        cp[1] = verts.pack(dim_c<3>,"x",tri[0]);
                        cp[2] = verts.pack(dim_c<3>,"x",tri[1]);
                        cp[3] = verts.pack(dim_c<3>,"x",tri[2]);

                        auto bary = LSL_GEO::getInsideBarycentricCoordinates(cp);
                        auto bp = vec3::zeros();
                        for(int j = 0;j != 3;++j)
                            bp += bary[j] * cp[j + 1];
                        tverts_buffer.tuple(dim_c<3>,"x",ti * max_nm_binders + i) = bp;
                        tverts_buffer.tuple(dim_c<3>,"xk",ti * max_nm_binders + i) = cp[0];
                }
            }
        });


        tverts_buffer = tverts_buffer.clone({zs::memsrc_e::host});
        // kverts_buffer = kverts_buffer.clone({zs::memsrc_e::host});

        auto binder_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& binder_verts = binder_vis->verts;
        auto& binder_lines = binder_vis->lines;

        binder_verts.resize(tverts_buffer.size() * 2);
        binder_lines.resize(tverts_buffer.size());

        ompPol(zs::range(tverts_buffer.size()),
            [tverts_buffer = proxy<omp_space>({},tverts_buffer),scale = scale,
                &binder_verts,&binder_lines,max_nm_binders = max_nm_binders] (int cpi) mutable {
            binder_verts[cpi * 2 + 0] = tverts_buffer.pack(dim_c<3>,"x",cpi).to_array();
            binder_verts[cpi * 2 + 1] = tverts_buffer.pack(dim_c<3>,"xk",cpi).to_array();

            auto dir = binder_verts[cpi * 2 + 1] - binder_verts[cpi * 2 + 0];
            binder_verts[cpi * 2 + 1] = binder_verts[cpi * 2 + 0] + dir * scale;
            binder_lines[cpi] = vec2i(cpi * 2 + 1,cpi * 2 + 0);
        });

        set_output("binder_vis",std::move(binder_vis));
    }
};

ZENDEFNODE(VisualizeSurfaceBinder, {{"zssurf","kboundary"},
                                  {"binder_vis"},
                                  {
                                    {"string","binder_tag","binderTag"},
                                    {"float","scale","1.0"}
                                  },
                                  {"ZSGeometry"}});

// one kinematic points can only collide with one tris
struct ZSDynamicSurfaceBind : zeno::INode {

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
        
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();


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
            if(nrm_norm < 1e-6)
                nrm = zs::vec<T,3>::zeros();
            else
                nrm = nrm / (nrm_norm + 1e-5);

            tris.tuple(dim_c<3>,"nrm",ti) = nrm;
        });     

        auto& kverts = kboundary->getParticles();
        dtiles_t kverts_tmp{kverts.get_allocator(),
            {
                {"x",3},
                {"inds",1},
            },kverts.size()};
        TILEVEC_OPS::copy(cudaPol,kverts,"x",kverts,"x");
        cudaPol(zs::range(kverts_tmp.size()),
            [kverts_tmp = proxy<space>({},kverts_tmp)] ZS_LAMBDA(int vi) mutable {
                kverts_tmp("inds",vi) = reinterpret_bits<T>(vi);
        }); 

        auto binder_tag = get_param<std::string>("binder_tag");
        auto thickness_tag = get_param<std::string>("thickness_tag");
        auto inversion_tag = get_param<std::string>("inversion_tag");

        if(!kverts.hasProperty(binder_tag)){
            kverts.append_channels(cudaPol,{{binder_tag,1}});
            TILEVEC_OPS::fill(cudaPol,kverts,binder_tag,zs::reinterpret_bits<T>((int)-1));
        }
        if(!kverts.hasProperty(thickness_tag)){
            kverts.append_channels(cudaPol,{{thickness_tag,1}});
            TILEVEC_OPS::fill(cudaPol,kverts,thickness_tag,(T)0.0);
        }
        if(!kverts.hasProperty(inversion_tag)){
            kverts.append_channels(cudaPol,{{inversion_tag,1}});
            TILEVEC_OPS::fill(cudaPol,kverts,inversion_tag,(T)0.0);
        }


// unleash the binder which is no longer active
        auto force_tag = get_param<std::string>("force_tag");
        cudaPol(zs::range(kverts.size()),
            [kverts = proxy<space>({},kverts),
                verts = proxy<space>({},verts),
                tris = proxy<space>({},tris),
                binder_tag = zs::SmallString(binder_tag),
                inversion_tag = zs::SmallString(inversion_tag),
                force_tag = zs::SmallString(force_tag)] ZS_LAMBDA(int kvi) mutable {
            auto binder_idx = reinterpret_bits<int>(kverts(binder_tag,kvi));
            if(binder_idx < 0)
                return;

            auto tri = tris.pack(dim_c<3>,"inds",binder_idx).reinterpret_bits(int_c);

            // if the kvert and tri are in close contact
            // a. they are in penertration
            auto seg = kverts.pack(dim_c<3>,"x",kvi) - verts.pack(dim_c<3>,"x",tri[0]);
            auto nrm = tris.pack(dim_c<3>,"nrm",binder_idx);
            auto orient = dot(seg,nrm);

            auto inversion = kverts(inversion_tag,kvi);
            if(inversion*orient > 0)
                return;
            //b. they are not in penertration but the elastic force try to stick them together
            auto f = vec3::zeros();
            for(int i = 0;i != 3;++i)
                f += verts.pack(dim_c<3>,force_tag,tri[i]);
            auto tcenter = vec3::zeros();
            for(int i = 0;i != 3;++i)
                tcenter += verts.pack(dim_c<3>,"x",tri[0])/(T)3.0;
            
            seg = kverts.pack(dim_c<3>,"x",kvi) - tcenter;
            if(seg.dot(f) > 1e-6)
                return;
            
            // the kvert and tri are separating
            kverts(binder_tag,kvi) = reinterpret_bits<T>((int)-1);
        });
        
        auto dt = get_input2<float>("dt");
        auto outCollisionEps = get_input2<float>("outCollisionEps");
        // auto max_v = TILEVEC_OPS::max_norm<3>(cudaPol,kb_verts,"v");
        // auto bvh_thickness = max_v * dt * (T)1.2;

        auto triBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,verts,tris,wrapv<3>{},(T)0.0,"x");
        triBvh.build(cudaPol,bvs);  

        cudaPol(zs::range(kverts.size()),
            [kverts = proxy<space>({},kverts),
                triBvh = proxy<space>(triBvh),
                verts = proxy<space>({},verts),
                tris = proxy<space>({},tris),
                dt = dt,outCollisionEps = outCollisionEps,
                binder_tag = zs::SmallString(binder_tag),
                thickness_tag = zs::SmallString(thickness_tag),
                inversion_tag = zs::SmallString(inversion_tag)] ZS_LAMBDA(int kpi) mutable {

            auto ori_bind_idx = reinterpret_bits<int>(kverts(binder_tag,kpi));
            if(ori_bind_idx > -1)
                return;

            auto kp  = kverts.pack(dim_c<3>,"x",kpi);
            auto kv = kverts.pack(dim_c<3>,"v",kpi);
            auto bvh_thickness = kv.norm() * dt * (T)1.2;
            auto bv = bv_t{get_bounding_box(kp - bvh_thickness,kp + bvh_thickness)};
            auto kpt = kp + dt * kv;

            auto min_r = (T)1e8;
            int min_ti = -1;
            vec3 min_rd{};
            vec3 min_vs[3] = {};

            auto process_raytrace_pairs = [&](int ti) {
                auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);

                vec3 vs[3] = {};

                vs[0] = verts.pack(dim_c<3>,"x",tri[0]);
                vs[1] = verts.pack(dim_c<3>,"x",tri[1]);
                vs[2] = verts.pack(dim_c<3>,"x",tri[2]);

                // auto kpv0 = kp - vs[0];
                // auto kptv0 = kpt - vs[0];
                // auto kpv0n = kpv0.dot(nrm);
                // auto kptv0n = kptv0.dot(nrm);
                // if(kpv0n.dot(kptv0n) > (T)0.0)
                //     return;

                // vec3 vols[3] = {};
                // mat3 ms = {};
                // for(int i = 0;i != 3;++i) {
                //     auto e01 = vs[(i+0)%3] - kp;
                //     auto e02 = vs[(i+1)%3] - kp;
                //     auto e03 = kpt - kp;
                //     vols[i] = e01.cross(e02).dot(e03);
                // }

                // bool intersect = false;
                // if(vols[0] > 0 && vols[1] > 0 && vols[2] > 0)
                //     intersect = true;
                // if(vols[0] < 0 && vols[1] < 0 && vols[2] < 0)
                //     intersect = true;                

                auto rd = kv/ (kv.norm() + (T)1e-6);
                // auto b = (T)0.0;
                // auto rtmp = (T)0.0;
                // auto dtmp = (T)0.0;
                // auto stmp = (T)0.0;
                // auto ttmp = (T)0.0;
                // auto bary = vec3::zeros();
                // auto ip_dist = (T)0.0;
                auto r = LSL_GEO::tri_ray_intersect(kp,rd,vs[0],vs[1],vs[2]);

                // if(r < 1e6) {
                //     printf("testing raytracing pairs : %d %d with r = %f b = %f rtmp = %f dtmp = %f stmp = %f ttmp = %f bary = %f %f %f dist = %f\n",
                //         kpi,ti,(float)r,(float)b,(float)rtmp,(float)dtmp,(float)stmp,(float)ttmp,
                //         (float)bary[0],(float)bary[1],(float)bary[2],(float)ip_dist);
                // }

                if(r < min_r){
                    min_r = r;
                    min_ti = ti;
                    min_vs[0] = vs[0];
                    min_vs[1] = vs[1];
                    min_vs[2] = vs[2];
                    min_rd = rd;
                }
            };
            triBvh.iter_neighbors(bv,process_raytrace_pairs);

            if(min_ti > -1) {
                auto rd = kv/ (kv.norm() + (T)1e-6);
                if(kv.norm() * dt < (min_r - outCollisionEps))
                    return;

                auto intersect_point = kp + rd * (min_r - outCollisionEps);
                T barySum = (T)0.0;
                T distance = LSL_GEO::pointTriangleDistance(min_vs[0],min_vs[1],min_vs[2],intersect_point,barySum);            

                auto seg = intersect_point - min_vs[0];
                auto nrm = tris.pack(dim_c<3>,"nrm",min_ti);
                auto orient = seg.dot(nrm);

                // printf("find binder %d %d\n",kpi,min_ti);

                kverts(binder_tag,kpi) = reinterpret_bits<T>(min_ti);
                kverts(thickness_tag,kpi) = distance;
                kverts(inversion_tag,kpi) = orient < 0 ? (T)1.0 : (T)-1.0;
            }
        });

        set_output("zsparticles",zsparticles);
        set_output("kboundary",kboundary);

    }
};

ZENDEFNODE(ZSDynamicSurfaceBind, {{"zsparticles","kboundary",
                                    {"float","outCollisionEps","0.01"},
                                    {"float","dt","1.0"}
                                    },
                                  {"zsparticles","kboundary"},
                                  {
                                    {"string","binder_tag","binderTag"},
                                    {"string","thickness_tag","thicknessTag"},
                                    {"string","inversion_tag","inversionTag"},
                                    {"string","force_tag","forceTag"}
                                  },
                                  {"ZSGeometry"}});

struct VisualZSDynamicBinder : zeno::INode {
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
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        const auto &verts = zsparticles->getParticles();
        const auto &tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();
        const auto& kverts = kboundary->getParticles();

        dtiles_t tverts_buffer{tris.get_allocator(),
            {
                {"x",3},
        },tris.size()};

        dtiles_t kverts_buffer{kverts.get_allocator(),
            {
                {"x",3},
                {"binder_idx",1}
        },kverts.size()};
        auto binder_tag = get_param<std::string>("binder_tag");
        // evaluate the center of the tris
        cudaPol(zs::range(tris.size()),
            [tris = proxy<space>({},tris),
                verts = proxy<space>({},verts),
                tverts_buffer = proxy<space>({},tverts_buffer)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            tverts_buffer.tuple(dim_c<3>,"x",ti) = vec3::zeros();
            for(int i = 0;i != 3;++i)
                tverts_buffer.tuple(dim_c<3>,"x",ti) = tverts_buffer.pack(dim_c<3>,"x",ti) + verts.pack(dim_c<3>,"x",tri[i])/(T)3.0;
        });
        TILEVEC_OPS::copy(cudaPol,kverts,"x",kverts_buffer,"x");
        TILEVEC_OPS::copy(cudaPol,kverts,binder_tag,kverts_buffer,"binder_idx");
        // cudaPol(zs::range(kverts_buffer.size()),
        //     [kverts_buffer = proxy<space>({},kverts_buffer)] ZS_LAMBDA(int kvi) mutable {
        //         printf("kverts_idx : %d\n",reinterpret_bits<int>(kverts_buffer("binder_idx",kvi)));
        // });        

        tverts_buffer = tverts_buffer.clone({zs::memsrc_e::host});
        kverts_buffer = kverts_buffer.clone({zs::memsrc_e::host});

        auto binder_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& binder_verts = binder_vis->verts;
        auto& binder_lines = binder_vis->lines;

        binder_verts.resize(kverts.size() * 2);
        binder_lines.resize(kverts.size());

        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();

        ompPol(zs::range(kverts_buffer.size()),
            [tverts_buffer = proxy<omp_space>({},tverts_buffer),
                kverts_buffer = proxy<omp_space>({},kverts_buffer),
                &binder_verts,&binder_lines] (int vi) mutable {
            binder_verts[vi * 2 + 0] = kverts_buffer.pack(dim_c<3>,"x",vi).to_array();
            auto ti = reinterpret_bits<int>(kverts_buffer("binder_idx",vi));
            if(ti < 0)
                binder_verts[vi * 2 + 1] = binder_verts[vi * 2 + 0];
            else
                binder_verts[vi * 2 + 1] = tverts_buffer.pack(dim_c<3>,"x",ti).to_array();
            binder_lines[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
        });       

        set_output("binder_vis",std::move(binder_vis)); 
    }
};

ZENDEFNODE(VisualZSDynamicBinder, {{"zsparticles","kboundary"},
                                  {"binder_vis"},
                                  {
                                    {"string","binder_tag","binderTag"}
                                  },
                                  {"ZSGeometry"}});


struct ZSSurfaceClosestPoints : zeno::INode {
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

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();
        
        // every vertex can only bind to one triangle
        auto& kverts = kboundary->getParticles();
        auto& ktris = kboundary->getQuadraturePoints();

        // auto max_nm_binder = get_input2<int>("nm_max_binders");
        auto project_pos_tag = get_param<std::string>("project_pos_tag");
        auto project_nrm_tag = get_param<std::string>("project_nrm_tag");
        auto project_idx_tag = get_param<std::string>("project_idx_tag");
        auto project_bary_tag = get_param<std::string>("project_bary_tag");
        auto align_direction = get_param<bool>("align_direction");

        // for each vertex of zsparticles, find a potential closest point on kboundary surface
        // add a plane constraint
        if(!verts.hasProperty(project_pos_tag) || !verts.hasProperty(project_nrm_tag) || !verts.hasProperty(project_idx_tag) || !verts.hasProperty(project_bary_tag)) {
            verts.append_channels(cudaPol,{
                {project_pos_tag,3},// the idx of triangle of kboudary
                {project_nrm_tag,3},
                {project_bary_tag,3},
                {project_idx_tag,1}
            });
        }

        TILEVEC_OPS::fill(cudaPol,verts,project_idx_tag,zs::reinterpret_bits<T>((int)-1));
        
        auto ktBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,kverts,ktris,wrapv<3>{},(T)0.0,"x");
        ktBvh.build(cudaPol,bvs);

        auto kinInCollisionEps = get_input2<float>("kinInColEps");
        auto kinOutCollisionEps = get_input2<float>("kinOutColEps");
        auto thickness = kinInCollisionEps + kinOutCollisionEps;

        // evaluate nodal-wise normal of kboundary

        // if(!ktris.hasProperty("nrm"))
        //     ktris.append_channels(cudaPol,{{"nrm",3}});
        // cudaPol(zs::range(ktris.size()),
        //     [ktris = proxy<space>({},ktris),
        //         kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kti) {
        //     auto ktri = ktris.template pack<3>("inds",kti).reinterpret_bits(int_c);
        //     auto kv0 = kverts.template pack<3>("x",ktri[0]);
        //     auto kv1 = kverts.template pack<3>("x",ktri[1]);
        //     auto kv2 = kverts.template pack<3>("x",ktri[2]);

        //     auto e01 = kv1 - kv0;
        //     auto e02 = kv2 - kv0;

        //     auto nrm = e01.cross(e02);
        //     auto nrm_norm = nrm.norm();
        //     if(nrm_norm < 1e-8)
        //         nrm = zs::vec<T,3>::zeros();
        //     else
        //         nrm = nrm / nrm_norm;

        //     ktris.tuple(dim_c<3>,"nrm",kti) = nrm;
        // });  

        // if(!kverts.hasProperty("nrm"))
        //     kverts.append_channels(cudaPol,{{"nrm",3}});
        // TILEVEC_OPS::fill(cudaPol,kverts,"nrm",(T)0.0);
        // cudaPol(zs::range(ktris.size()),[
        //         ktris = proxy<space>({},ktris),
        //         kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kti) mutable {
        //     auto ktri = ktris.pack(dim_c<3>,"inds",kti).reinterpret_bits(int_c);
        //     auto nrm = ktris.pack(dim_c<3>,"nrm",kti);
        //     for(int i = 0;i != 3;++i)
        //         for(int d = 0;d != 3;++d)
        //             atomic_add(exec_cuda,&kverts("nrm",d,ktri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
        // });
        // cudaPol(zs::range(kverts.size()),[kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kvi) mutable {
        //     auto nrm = kverts.pack(dim_c<3>,"nrm",kvi);
        //     nrm = nrm / (nrm.norm() + (T)1e-6);
        //     kverts.tuple(dim_c<3>,"nrm",kvi) = nrm;
        // });   

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
        if(!verts.hasProperty("nrm"))
            verts.append_channels(cudaPol,{{"nrm",3}});
        TILEVEC_OPS::fill(cudaPol,verts,"nrm",(T)0.0);     
        cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            auto nrm = tris.pack(dim_c<3>,"nrm",ti);
            for(int i = 0;i != 3;++i)
                for(int d = 0;d != 3;++d)
                    atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
        });           
        cudaPol(zs::range(verts.size()),[verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
            auto nrm = verts.pack(dim_c<3>,"nrm",vi);
            nrm = nrm / (nrm.norm() + (T)1e-6);
            verts.tuple(dim_c<3>,"nrm",vi) = nrm;
        });   


        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            ktBvh = proxy<space>(ktBvh),
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            project_pos_tag = zs::SmallString(project_pos_tag),
            project_nrm_tag = zs::SmallString(project_nrm_tag),
            project_idx_tag = zs::SmallString(project_idx_tag),
            project_bary_tag = zs::SmallString(project_bary_tag),
            kinInCollisionEps = kinInCollisionEps,
            align_direction = align_direction,
            kinOutCollisionEps = kinOutCollisionEps,
            thickness = thickness] ZS_LAMBDA(int vi) mutable {
                if(verts.hasProperty("is_surf"))
                    if(verts("is_surf",vi) < (T)0.5)
                        return;
                if(verts.hasProperty("k_active"))// static unbind
                    if(verts("k_active",vi) < (T)0.5)
                        return;
                // if(verts.hasProperty("k_fail"))// dynamic unbind
                //     if(verts("k_fail",vi) > (T)0.5)
                        // return;
                auto p = verts.pack(dim_c<3>,"x",vi);
                auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};

                auto min_dist = std::numeric_limits<T>::infinity();
                int min_tri_idx = -1;
                auto min_bary = vec3::zeros();
                auto min_collision_eps = (T)0;

                auto pnrm = verts.pack(dim_c<3>,"nrm",vi);

                auto process_potential_closest_tris = [&](int kti) {
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti).reinterpret_bits(int_c);
                    if(kverts.hasProperty("k_active"))
                        for(int i = 0;i != 3;++i)
                            if(kverts("k_active",ktri[i]) < (T)0.5)
                                return;
                    auto kv0 = kverts.pack(dim_c<3>,"x",ktri[0]);
                    auto kv1 = kverts.pack(dim_c<3>,"x",ktri[1]);
                    auto kv2 = kverts.pack(dim_c<3>,"x",ktri[2]);

                    vec3 bary{};
                    vec3 project_bary{};
                    T distance = LSL_GEO::pointTriangleDistance(kv0,kv1,kv2,p,bary,project_bary);
                    if(distance > min_dist)
                        return;


                    auto seg = p - kv0;
                    auto knrm = ktris.pack(dim_c<3>,"nrm",kti);
                    auto dist = seg.dot(knrm);

                    auto collisionEps = dist > 0 ? kinOutCollisionEps : kinInCollisionEps;
                    if(distance > collisionEps)
                        return;
                    

                    auto align = knrm.dot(pnrm);
                    if(align < 0.96 && align_direction && dist < 0){
                        // printf("failed of %d %d due to aligh = %f\n",vi,kti,(float)align);
                        return;
                    }
                    if(align > -0.96 && !align_direction && dist < 0){
                        // printf("failed of %d %d due to aligh = %f\n",vi,kti,(float)align);
                        return;
                    }

                    auto bary_sum = fabs(bary[0]) + fabs(bary[1]) + fabs(bary[2]);
                    if(bary_sum > 1.1)
                        return;
                    else {
                        auto ntris = ktris.pack(dim_c<3>,"ff_inds",kti).reinterpret_bits(int_c);

                        for(int i = 0;i != 3;++i){
                            auto nti = ntris[i];
                            auto edge_normal = vec3::zeros();
                            if(nti < 0){
                                edge_normal = knrm;
                            }else {
                                edge_normal = ktris.pack(dim_c<3>,"nrm",nti) + knrm;
                                edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                            }
                            auto ke0 = kverts.pack(dim_c<3>,"x",ktri[(i + 0) % 3]);
                            auto ke1 = kverts.pack(dim_c<3>,"x",ktri[(i + 1) % 3]);  
                            auto ke10 = ke1 - ke0;
                            auto bisector_normal = edge_normal.cross(ke10).normalized();

                            seg = p - kverts.pack(dim_c<3>,"x",ktri[(i + 0) % 3]);
                            if(bisector_normal.dot(seg) < 0)
                                return;
                        }
                    }

                    min_dist = distance;
                    min_tri_idx = kti;
                    min_bary = project_bary;
                    min_collision_eps = collisionEps;
                };
                ktBvh.iter_neighbors(bv,process_potential_closest_tris);

                if(min_tri_idx == -1)
                    return;
                auto closest_ktri = ktris.pack(dim_c<3>,"inds",min_tri_idx).reinterpret_bits(int_c);
                if(kverts.hasProperty("k_fail"))
                    for(int i = 0;i != 3;++i)
                        if(kverts("k_fail",closest_ktri[i]) > (T)0.5)
                            return;

                auto ori_bary = min_bary;

                min_bary[0] = min_bary[0] < 0 ? (T)0 : min_bary[0];
                min_bary[1] = min_bary[1] < 0 ? (T)0 : min_bary[1];
                min_bary[2] = min_bary[2] < 0 ? (T)0 : min_bary[2];
                min_bary = min_bary/min_bary.sum();
                if(min_bary[0] < 0 || min_bary[1] < 0 || min_bary[2] < 0)
                    printf("invalid min_bary[%f %f %f]\n",
                        (float)min_bary[0],
                        (float)min_bary[1],
                        (float)min_bary[2]);
                if((zs::abs(min_bary[0]) + zs::abs(min_bary[1]) + zs::abs(min_bary[2])) > 1.1)
                    printf("invalid min_bary[%f %f %f]\n",
                        (float)min_bary[0],
                        (float)min_bary[1],
                        (float)min_bary[2]);

                if((zs::abs(min_bary[0] + min_bary[1] + min_bary[2] - 1.0)) > 0.1)
                    printf("invalid min_bary[%f %f %f]\n",
                        (float)min_bary[0],
                        (float)min_bary[1],
                        (float)min_bary[2]);

                auto project_kv = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    project_kv += kverts.pack(dim_c<3>,"x",closest_ktri[i]) * min_bary[i];
                auto project_knrm = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    project_knrm += kverts.pack(dim_c<3>,"nrm",closest_ktri[i]) * min_bary[i];
                project_knrm /= (project_knrm.norm() + 1e-6);

                // printf("find closest pairs : %d %d\n",vi,min_tri_idx);

                // printf("vert[%d] = %f %f %f closest to ktri[%d] = %f %f %f bary = %f %f %f\n",vi,
                //     (float)p[0],(float)p[1],(float)p[2],
                //     min_tri_idx,
                //     (float)project_kv[0],(float)project_kv[1],(float)project_kv[2],
                //     (float)ori_bary[0],(float)ori_bary[1],(float)ori_bary[2]
                // );

                verts.tuple(dim_c<3>,project_pos_tag,vi) = project_kv;

                auto distance = (verts.pack(dim_c<3>,"x",vi) - project_kv).norm();
                if(distance > kinInCollisionEps)
                    printf("find invalid distance %f > %f : %f\n",(float)distance,(float)kinInCollisionEps,(float)min_collision_eps);

                verts.tuple(dim_c<3>,project_nrm_tag,vi) = project_knrm;
                verts.tuple(dim_c<3>,project_bary_tag,vi) = min_bary;
                verts(project_idx_tag,vi) = reinterpret_bits<T>(min_tri_idx);
        });

        set_output("zsparticles",zsparticles);
        set_output("kboundary",kboundary);
    }
};

ZENDEFNODE(ZSSurfaceClosestPoints, {
                                  {
                                    "zsparticles",
                                    "kboundary",
                                    {"float","kinInColEps","0.001"},
                                    {"float","kinOutColEps","0.001"},
                                  },
                                  {"zsparticles","kboundary"},
                                  {
                                    {"bool","align_direction","1"},
                                    {"string","project_pos_tag","project_pos_tag"},
                                    {"string","project_nrm_tag","project_nrm_tag"},
                                    {"string","project_idx_tag","project_idx_tag"},
                                    {"string","project_bary_tag","project_bary_tag"}
                                  },
                                  {"ZSGeometry"}});



struct ZSVisualizeClosestPoints : zeno::INode {
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

        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();       

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        const auto& verts = zsparticles->getParticles();
        const auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();
        
        const auto& kverts = kboundary->getParticles();
        const auto& ktris = kboundary->getQuadraturePoints();

        auto project_pos_tag = get_param<std::string>("project_pos_tag");
        auto project_nrm_tag = get_param<std::string>("project_nrm_tag");
        auto project_idx_tag = get_param<std::string>("project_idx_tag");

        dtiles_t verts_buffer{verts.get_allocator(),{
            {"x",3},
            {"xp",3},
            {"nrm",3}
        },verts.size()/* + tris.size()*/};

        cudaPol(zs::range(verts.size()),[
                verts = proxy<space>({},verts),
                verts_buffer = proxy<space>({},verts_buffer),
                project_idx_tag = zs::SmallString(project_idx_tag),
                project_pos_tag = zs::SmallString(project_pos_tag),
                project_nrm_tag = zs::SmallString(project_nrm_tag)
            ] ZS_LAMBDA(int vi) mutable {
                verts_buffer.tuple(dim_c<3>,"x",vi) = verts.pack(dim_c<3>,"x",vi);
                auto pidx = reinterpret_bits<int>(verts(project_idx_tag,vi));
                verts_buffer.tuple(dim_c<3>,"nrm",vi) = vec3::zeros();
                if(pidx < 0)
                    verts_buffer.tuple(dim_c<3>,"xp",vi) = verts.pack(dim_c<3>,"x",vi);
                else{
                    verts_buffer.tuple(dim_c<3>,"xp",vi) = verts.pack(dim_c<3>,project_pos_tag,vi);
                    verts_buffer.tuple(dim_c<3>,"nrm",vi) = verts.pack(dim_c<3>,project_nrm_tag,vi);
                }
        });

        // int offset = verts.size();
        // cudaPol(zs::range(tris.size()),[
        //         verts = proxy<space>({},verts),
        //         tris = proxy<space>({},tris),
        //         offset = offset,
        //         verts_buffer = proxy<space>({},verts_buffer),
        //         project_idx_tag = zs::SmallString(project_idx_tag),
        //         project_pos_tag = zs::SmallString(project_pos_tag),
        //         project_nrm_tag = zs::SmallString(project_nrm_tag)
        //     ] ZS_LAMBDA(int ti) mutable {
        //         auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
        //         auto p = vec3::zeros();
        //         for(int i = 0;i != 3;++i)
        //             p += verts.pack(dim_c<3>,"x",tri[i])/(T)3.0;
        //         verts_buffer.tuple(dim_c<3>,"x",ti + offset) = p;
        //         auto pidx = reinterpret_bits<int>(tris(project_idx_tag,ti));
        //         verts_buffer.tuple(dim_c<3>,"nrm",ti + offset) = vec3::zeros();
        //         if(pidx < 0)
        //             verts_buffer.tuple(dim_c<3>,"xp",ti + offset) = p;
        //         else{
        //             verts_buffer.tuple(dim_c<3>,"xp",ti + offset) = tris.pack(dim_c<3>,project_pos_tag,ti);
        //             verts_buffer.tuple(dim_c<3>,"nrm",ti + offset) = tris.pack(dim_c<3>,project_nrm_tag,ti);
        //         }
        // });

        verts_buffer = verts_buffer.clone({zs::memsrc_e::host});

        auto closest_points_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& pverts = closest_points_vis->verts;
        auto& plines = closest_points_vis->lines;
        pverts.resize(verts_buffer.size() * 2);
        plines.resize(verts_buffer.size());


        ompPol(zs::range(verts_buffer.size()),
            [verts_buffer = proxy<omp_space>({},verts_buffer),
                &pverts,&plines] (int vi) mutable {
            pverts[vi * 2 + 0] = verts_buffer.pack(dim_c<3>,"x",vi).to_array();
            pverts[vi * 2 + 1] = verts_buffer.pack(dim_c<3>,"xp",vi).to_array();
            plines[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
        });     

        auto nrm_scale = get_input2<float>("nrm_scale");
        auto normal_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& nverts = normal_vis->verts;
        auto& nlines = normal_vis->lines;
        nverts.resize(verts_buffer.size() * 2);
        nlines.resize(verts_buffer.size());
        auto& nclrs = nverts.add_attr<zeno::vec3f>("clr");
        int nm_verts = verts.size();
        ompPol(zs::range(verts_buffer.size()),
            [verts_buffer = proxy<omp_space>({},verts_buffer),
                &nverts,&nlines,&nclrs,nrm_scale = nrm_scale,nm_verts = nm_verts] (int vi) mutable {
            nverts[vi * 2 + 0] = verts_buffer.pack(dim_c<3>,"x",vi).to_array();
            auto ep = verts_buffer.pack(dim_c<3>,"nrm",vi) * nrm_scale + verts_buffer.pack(dim_c<3>,"x",vi);
            nverts[vi * 2 + 1] = ep.to_array();
            nlines[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
            if(vi < nm_verts){
                nclrs[vi * 2 + 0] = zeno::vec3f(1.0f,1.0f,1.0f);
                nclrs[vi * 2 + 1] = zeno::vec3f(1.0f,1.0f,1.0f);
            }else{
                nclrs[vi * 2 + 0] = zeno::vec3f(1.0f,1.0f,0.0f);
                nclrs[vi * 2 + 1] = zeno::vec3f(1.0f,1.0f,0.0f);
            }
        });     


        set_output("closest_vis",std::move(closest_points_vis)); 
        set_output("normal_vis",std::move(normal_vis));   
    }
};

ZENDEFNODE(ZSVisualizeClosestPoints, {{"zsparticles","kboundary",{"float","nrm_scale","1.0"}},
                                  {"closest_vis","normal_vis"},
                                  {
                                    {"string","project_pos_tag","project_pos_tag"},
                                    {"string","project_nrm_tag","project_nrm_tag"},
                                    {"string","project_idx_tag","project_idx_tag"},
                                  },
                                  {"ZSGeometry"}});




struct ZSSurfaceClosestTris : zeno::INode {
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

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();

        auto& kverts = kboundary->getParticles();
        auto& ktris = kboundary->getQuadraturePoints();

        // auto max_nm_binder = get_input2<int>("nm_max_binders");
        // auto project_pos_tag = get_param<std::string>("project_pos_tag");
        // auto project_nrm_tag = get_param<std::string>("project_nrm_tag");
        // the id of the vertex on kboundary
        auto project_idx_tag = get_param<std::string>("project_idx_tag");
        // auto project_bary_tag = get_param<std::string>("project_bary_tag");
        auto align_direction = get_param<bool>("align_direction");

        // if(!verts)
        if(!tris.hasProperty(project_idx_tag))
            tris.append_channels(cudaPol,{{project_idx_tag,1}});

        auto cnorm = compute_average_edge_length(cudaPol,verts,"x",tris);
        cnorm *= 2;

        if(!kverts.hasProperty("inds")) {
            kverts.append_channels(cudaPol,{{"inds",1}});
            cudaPol(zs::range(kverts.size()),
                [kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kvi) mutable {
                    kverts("inds",kvi) = reinterpret_bits<T>(kvi);
            });
        }

        TILEVEC_OPS::fill(cudaPol,tris,project_idx_tag,zs::reinterpret_bits<T>((int)-1));
        auto kpBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,kverts,kverts,wrapv<1>{},(T)cnorm,"x");
        kpBvh.build(cudaPol,bvs);

        auto kinInCollisionEps = get_input2<float>("kinInColEps");
        auto kinOutCollisionEps = get_input2<float>("kinOutColEps");
        auto thickness = kinInCollisionEps + kinOutCollisionEps;
        thickness *= (T)2;

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

        if(!verts.hasProperty("nrm"))
            verts.append_channels(cudaPol,{{"nrm",3}});
        TILEVEC_OPS::fill(cudaPol,verts,"nrm",(T)0.0);
        #if 1
        cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            auto nrm = tris.pack(dim_c<3>,"nrm",ti);
            for(int i = 0;i != 3;++i)
                for(int d = 0;d != 3;++d)
                    atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
        });  
        #else
        cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            bool is_active_tri = true;
            if(verts.hasProperty("k_active")){
                for(int i = 0;i != 3;++i)
                    if(verts("k_active",tri[i]) < (T)0.5)
                        is_active_tri = false;
            }

            auto nrm = tris.pack(dim_c<3>,"nrm",ti);
            for(int i = 0;i != 3;++i){
                if(is_active_tri || (verts.hasProperty("k_active") && verts("k_active",tri[i]) < (T)0.5))
                    for(int d = 0;d != 3;++d)
                        atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
            }
        });   
        #endif  

        cudaPol(zs::range(verts.size()),[verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
            auto nrm = verts.pack(dim_c<3>,"nrm",vi);
            nrm = nrm / (nrm.norm() + (T)1e-6);
            verts.tuple(dim_c<3>,"nrm",vi) = nrm;
        });   

        // make sure the kboundary has normal
        if(!kverts.hasProperty("nrm")) {
            fmt::print(fg(fmt::color::red),"the input kboundary should have nodal normal\n");
            throw std::runtime_error("the input kboundary should have nodal normal");
        }  

        // for each triangle, find the closest point
        cudaPol(zs::range(tris.size()),[
                verts = proxy<space>({},verts),
                tris = proxy<space>({},tris),
                kpBvh = proxy<space>(kpBvh),
                kverts = proxy<space>({},kverts),
                project_idx_tag = zs::SmallString(project_idx_tag),
                kinInCollisionEps = kinInCollisionEps,
                kinOutCollisionEps = kinOutCollisionEps,
                align_direction = align_direction,
                thickness = thickness] ZS_LAMBDA(int ti) mutable {
            auto tp = vec3::zeros();
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            if(verts.hasProperty("k_active"))
                for(int i = 0;i != 3;++i)
                    if(verts("k_active",tri[i]) < (T)0.5)
                        return;
            for(int i = 0;i != 3;++i)
                tp += verts.pack(dim_c<3>,"x",tri[i]) / (T)3.0;
            auto bv = bv_t{get_bounding_box(tp - thickness,tp + thickness)};
            
            auto min_dist = std::numeric_limits<T>::infinity();
            int min_kp_idx = -1;

            auto tnrm = vec3::zeros();
            for(int i = 0;i != 3;++i)
                tnrm += verts.pack(dim_c<3>,"nrm",tri[i]);
            tnrm /= tnrm.norm();

            vec3 tvs[3] = {};
            for(int i = 0;i != 3;++i)
                tvs[i] = verts.pack(dim_c<3>,"x",tri[i]);

            auto ntris = tris.pack(dim_c<3>,"ff_inds",ti).reinterpret_bits(int_c);
            vec3 bnrms[3] = {};
            for(int i = 0;i != 3;++i){
                auto nti = ntris[i];
                auto edge_normal = vec3::zeros();
                if(nti < 0)
                    edge_normal = tnrm;
                else{
                    edge_normal = tnrm + tris.pack(dim_c<3>,"nrm",nti);
                    edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                }
                auto e01 = tvs[(i + 1) % 3] - tvs[(i + 0) % 3];
                bnrms[i] = edge_normal.cross(e01).normalized();
            }

            auto process_potential_closest_point = [&](int kpi) {
                if(kverts.hasProperty("k_active"))
                    if(kverts("k_active",kpi) < (T)0.5)
                        return;
                auto kp = kverts.pack(dim_c<3>,"x",kpi);

                vec3 bary{};
                vec3 project_bary{};

                T distance = (kp - tp).norm();
                if(distance > min_dist)
                    return;

                T pt_distance = LSL_GEO::pointTriangleDistance(tvs[0],tvs[1],tvs[2],kp,bary,project_bary);
                auto seg = tvs[0] - kp;
                auto kpnrm = kverts.pack(dim_c<3>,"nrm",kpi);
                auto dist = seg.dot(tnrm);

                auto collisionEps = dist > 0 ? kinOutCollisionEps : kinInCollisionEps;
                if(pt_distance > collisionEps)
                    return;
                
                auto align = kpnrm.dot(tnrm);
                if(align < 0.68 && align_direction && dist < 0)
                    return;
                if(align > -0.68 && !align_direction && dist < 0)
                    return;
                // if(align < 0.5 && align_direction && dist < 0)
                //     return;
                // if(align > -0.5 && !align_direction && dist < 0)
                //     return;
                
                auto bary_sum = fabs(bary[0]) + fabs(bary[1]) + fabs(bary[2]);
                if(bary_sum > 1.01)
                    return;
                else{
                    for(int i = 0;i != 3;++i) {
                        seg = kp - tvs[i];
                        if(bnrms[i].dot(seg) < 0)
                            return;
                    }
                }

                min_dist = distance;
                min_kp_idx = kpi;
            };
            kpBvh.iter_neighbors(bv,process_potential_closest_point);
            
            if(min_kp_idx == -1)
                return;
            tris(project_idx_tag,ti) = reinterpret_bits<T>(min_kp_idx); 
        });
        
        set_output("zsparticles",zsparticles);
        set_output("kboundary",kboundary);
    }
};

ZENDEFNODE(ZSSurfaceClosestTris, {
                                  {
                                    "zsparticles",
                                    "kboundary",
                                    {"float","kinInColEps","0.001"},
                                    {"float","kinOutColEps","0.001"},
                                  },
                                  {"zsparticles","kboundary"},
                                  {
                                    {"bool","align_direction","1"},
                                    {"string","project_idx_tag","project_idx_tag"}
                                  },
                                  {"ZSGeometry"}});

struct VisualizeClosestTris : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec2 = zs::vec<T,2>;
    using vec3 = zs::vec<T, 3>;
    using vec3i = zs::vec<Ti,3>;
    using mat3 = zs::vec<T, 3, 3>;
    using mat9 = zs::vec<T,9,9>;
    using mat12 = zs::vec<T,12,12>;

    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();

        auto& kverts = kboundary->getParticles();
        auto& ktris = kboundary->getQuadraturePoints();

        // auto max_nm_binder = get_input2<int>("nm_max_binders");
        // auto project_pos_tag = get_param<std::string>("project_pos_tag");
        // auto project_nrm_tag = get_param<std::string>("project_nrm_tag");
        // the id of the vertex on kboundary
        auto project_idx_tag = get_param<std::string>("project_idx_tag");
        auto kinOutCollisionEps = get_input2<float>("kin_out_collision_eps");

        dtiles_t verts_buffer{tris.get_allocator(),{
            {"x",3},
            {"xp",3},
            {"nrm",3},
            {"inds",3},
            {"grad",9}
        },tris.size()};    

        dtiles_t force_buffer{verts.get_allocator(),{
            {"force",3},
            {"x",3}
        },verts.size()};

        TILEVEC_OPS::copy(cudaPol,tris,"inds",verts_buffer,"inds");
        TILEVEC_OPS::copy(cudaPol,verts,"x",force_buffer,"x");
        TILEVEC_OPS::fill(cudaPol,verts_buffer,"grad",(T)0.0);
        TILEVEC_OPS::fill(cudaPol,force_buffer,"force",(T)0.0);

        cudaPol(zs::range(tris.size()),[
            verts_buffer = proxy<space>({},verts_buffer),
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            kinOutCollisionEps = kinOutCollisionEps,
            project_idx_tag = zs::SmallString(project_idx_tag)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                auto tp = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    tp += verts.pack(dim_c<3>,"x",tri[i]) / (T)3.0;
                
                verts_buffer.tuple(dim_c<3>,"x",ti) = tp;
                verts_buffer.tuple(dim_c<3>,"nrm",ti) = vec3::zeros();
                auto kp_idx = reinterpret_bits<int>(tris(project_idx_tag,ti));
                if(kp_idx < 0)
                    verts_buffer.tuple(dim_c<3>,"xp",ti) = tp;
                else{
                    auto kp = kverts.pack(dim_c<3>,"x",kp_idx);
                    auto tnrm = tris.pack(dim_c<3>,"nrm",ti);
                    verts_buffer.tuple(dim_c<3>,"xp",ti) = kp;
                    verts_buffer.tuple(dim_c<3>,"nrm",ti) = tnrm;

                    vec3 vs[4] = {};
                    auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                    vs[0] = kp;
                    for(int i = 0;i != 3;++i)
                        vs[i + 1] = verts.pack(dim_c<3>,"x",tri[i]);

                    vec3 e[3] = {};
                    e[0] = vs[3] - vs[2];
                    e[1] = vs[0] - vs[2];
                    e[2] = vs[1] - vs[2];

                    auto n = e[2].cross(e[0]);
                    n = n/(n.norm() + 1e-6);                    

                    T springLength = e[1].dot(n) - kinOutCollisionEps;
                    auto gvf = zs::vec<T,9>::zeros();
                    if(springLength < (T)0){
                        auto gvf_v12 = COLLISION_UTILS::springLengthGradient(vs,e,n);
                        if(isnan(gvf_v12.norm()))
                            printf("nan gvf detected at %d %f %f\n",ti,gvf_v12.norm(),n.norm());
                        for(int i = 0;i != 9;++i)
                            gvf[i] = gvf_v12[i + 3];
                    }
                    auto stiffness = (T)1.0;         
                    auto g = -stiffness * springLength * gvf;
                    // auto H = stiffness * zs::dyadic_prod(gvf, gvf);
                    verts_buffer.tuple(dim_c<9>,"grad",ti) = g;
                }

        });

        TILEVEC_OPS::assemble(cudaPol,verts_buffer,"grad","inds",force_buffer,"force");

        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();     

        verts_buffer = verts_buffer.clone({zs::memsrc_e::host});
        force_buffer = force_buffer.clone({zs::memsrc_e::host});
        auto closest_points_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& pverts = closest_points_vis->verts;
        auto& plines = closest_points_vis->lines;
        pverts.resize(verts_buffer.size()  * 2);
        plines.resize(verts_buffer.size());

        ompPol(zs::range(verts_buffer.size()),[
            verts_buffer = proxy<omp_space>({},verts_buffer),
            &pverts,&plines] (int vi) mutable {
                pverts[vi * 2 + 0] = verts_buffer.pack(dim_c<3>,"x",vi).to_array();
                pverts[vi * 2 + 1] = verts_buffer.pack(dim_c<3>,"xp",vi).to_array();
                plines[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
        });

        auto nrm_scale = get_input2<float>("nrm_scale");
        auto normal_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& nverts = normal_vis->verts;
        auto& nlines = normal_vis->lines;
        nverts.resize(verts_buffer.size() * 2);
        nlines.resize(verts_buffer.size());
        ompPol(zs::range(verts_buffer.size()),[
            verts_buffer = proxy<omp_space>({},verts_buffer),
            &nverts,&nlines,nrm_scale = nrm_scale] (int vi) mutable {
                nverts[vi * 2 + 0] = verts_buffer.pack(dim_c<3>,"x",vi).to_array();
                auto ep = verts_buffer.pack(dim_c<3>,"nrm",vi) * nrm_scale + verts_buffer.pack(dim_c<3>,"x",vi);
                nverts[vi * 2 + 1] = ep.to_array();
                nlines[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
        });

        auto force_scale = get_input2<float>("force_scale");
        auto force_vis = std::make_shared<zeno::PrimitiveObject>();
        auto& fverts = force_vis->verts;
        auto& flines = force_vis->lines;
        fverts.resize(2 * verts.size());
        flines.resize(verts.size());
        ompPol(zs::range(verts.size()),[
            force_buffer = proxy<omp_space>({},force_buffer),
            &fverts,&flines,force_scale = force_scale] (int vi) mutable {
                fverts[vi * 2 + 0] = force_buffer.pack(dim_c<3>,"x",vi).to_array();
                auto ep = force_buffer.pack(dim_c<3>,"force",vi) * force_scale + force_buffer.pack(dim_c<3>,"x",vi);
                fverts[vi * 2 + 1] = ep.to_array();
                flines[vi] = zeno::vec2i{vi * 2 + 0,vi * 2 + 1};
        });

        set_output("closest_vis",std::move(closest_points_vis));
        set_output("normal_vis",std::move(normal_vis));
        set_output("force_vis",std::move(force_vis));
    }    
};

ZENDEFNODE(VisualizeClosestTris, {{"zsparticles","kboundary",
                                        {"float","nrm_scale","1.0"},
                                        {"float","force_scale","1.0"},
                                        {"float","kin_out_collision_eps","0.001"}
                                  },
                                  {"closest_vis","normal_vis","force_vis"},
                                  {
                                    {"string","project_idx_tag","project_idx_tag"},
                                  },
                                  {"ZSGeometry"}});

struct ZSCalcSurfaceNormal : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec2 = zs::vec<T,2>;
    using vec3 = zs::vec<T, 3>;
    using vec3i = zs::vec<Ti,3>;
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

        auto zsparticles = get_input<ZenoParticles>("zsparticles");

        auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();


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
        if(!verts.hasProperty("nrm"))
            verts.append_channels(cudaPol,{{"nrm",3}});
        TILEVEC_OPS::fill(cudaPol,verts,"nrm",(T)0.0);     
        cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            bool is_active_tri = true;
            if(verts.hasProperty("k_active")){
                for(int i = 0;i != 3;++i)
                    if(verts("k_active",tri[i]) < (T)0.5)
                        is_active_tri = false;
            }

            if(is_active_tri) {
                auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                for(int i = 0;i != 3;++i){
                    for(int d = 0;d != 3;++d)
                        atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
                }
            }
        });           
        cudaPol(zs::range(verts.size()),[verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
            auto nrm = verts.pack(dim_c<3>,"nrm",vi);
            nrm = nrm / (nrm.norm() + (T)1e-6);
            verts.tuple(dim_c<3>,"nrm",vi) = nrm;
        });   

        int nm_iterations = get_param<int>("nm_smooth_iters");
        for(int i = 0;i != nm_iterations;++i) {
            TILEVEC_OPS::fill(cudaPol,tris,"nrm",(T)0.0);  
            cudaPol(zs::range(tris.size()),
                        [tris = proxy<space>({},tris),
                            verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) {
                        auto tri = tris.template pack<3>("inds",ti).reinterpret_bits(int_c);
                        auto nrm = vec3::zeros();
                        for(int i = 0;i != 3;++i)
                            nrm += verts.pack(dim_c<3>,"nrm",tri[i]);
                        nrm = nrm / (nrm.norm() + (T)1e-6);
                        tris.tuple(dim_c<3>,"nrm",ti) = nrm;
                    });
            TILEVEC_OPS::fill(cudaPol,verts,"nrm",(T)0.0);     
            cudaPol(zs::range(tris.size()),[
                    tris = proxy<space>({},tris),
                    verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                auto nrm = tris.pack(dim_c<3>,"nrm",ti);
                for(int i = 0;i != 3;++i)
                    for(int d = 0;d != 3;++d)
                        atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
            });           
            cudaPol(zs::range(verts.size()),[verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
                auto nrm = verts.pack(dim_c<3>,"nrm",vi);
                nrm = nrm / (nrm.norm() + (T)1e-6);
                verts.tuple(dim_c<3>,"nrm",vi) = nrm;
            });               
        }


        set_output("zsparticles",zsparticles);
    }
        
};


ZENDEFNODE(ZSCalcSurfaceNormal, {{"zsparticles"},
                                  {"zsparticles"},
                                  {
                                    {"int","nm_smooth_iters","0"}
                                  },
                                  {"ZSGeometry"}});

// for each points of the zsparticles, find the closest three points on the kbonudary
struct ZSSurfaceClosestPointsGrp : zeno::INode {
    using T = float;
    using Ti = int;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec2 = zs::vec<T,2>;
    using vec3 = zs::vec<T, 3>;
    using vec3i = zs::vec<Ti,3>;
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

        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto kboundary = get_input<ZenoParticles>("kboundary");

        auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();

        auto& kverts = kboundary->getParticles();
        auto& ktris = kboundary->getQuadraturePoints();

        auto project_pos_tag = get_param<std::string>("project_pos_tag");
        auto project_nrm_tag = get_param<std::string>("project_nrm_tag");
        auto project_idx_tag = get_param<std::string>("project_idx_tag");
        auto align_direction = get_param<bool>("align_direction");


        if(!verts.hasProperty(project_pos_tag) || !verts.hasProperty(project_nrm_tag) || !verts.hasProperty(project_idx_tag)) {
            verts.append_channels(cudaPol,{
                {project_pos_tag,3},// the idx of triangle of kboudary
                {project_nrm_tag,3},
                {project_idx_tag,1}
            });
        }

        if(!tris.hasProperty(project_pos_tag) || !tris.hasProperty(project_nrm_tag) || !tris.hasProperty(project_idx_tag)) {
            tris.append_channels(cudaPol,{
                {project_pos_tag,3},// the idx of triangle of kboudary
                {project_nrm_tag,3},
                {project_idx_tag,1}
            });
        }

        TILEVEC_OPS::fill(cudaPol,verts,project_idx_tag,zs::reinterpret_bits<T>((int)-1));
        TILEVEC_OPS::fill(cudaPol,tris,project_idx_tag,zs::reinterpret_bits<T>((int)-1));
        auto ktBvh = bvh_t{};
        auto bvs = retrieve_bounding_volumes(cudaPol,kverts,ktris,wrapv<3>{},(T)0.0,"x");
        ktBvh.build(cudaPol,bvs);

        auto kinInCollisionEps = get_input2<float>("kinInColEps");
        auto kinOutCollisionEps = get_input2<float>("kinOutColEps");
        auto thickness = kinInCollisionEps + kinOutCollisionEps;  

        if(!ktris.hasProperty("nrm"))
            ktris.append_channels(cudaPol,{{"nrm",3}});      

        if(!ktris.hasProperty("nrm"))
            ktris.append_channels(cudaPol,{{"nrm",3}});
        cudaPol(zs::range(ktris.size()),
            [ktris = proxy<space>({},ktris),
                kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kti) {
            auto ktri = ktris.template pack<3>("inds",kti).reinterpret_bits(int_c);
            auto kv0 = kverts.template pack<3>("x",ktri[0]);
            auto kv1 = kverts.template pack<3>("x",ktri[1]);
            auto kv2 = kverts.template pack<3>("x",ktri[2]);

            auto e01 = kv1 - kv0;
            auto e02 = kv2 - kv0;

            auto nrm = e01.cross(e02);
            auto nrm_norm = nrm.norm();
            if(nrm_norm < 1e-8)
                nrm = zs::vec<T,3>::zeros();
            else
                nrm = nrm / nrm_norm;

            ktris.tuple(dim_c<3>,"nrm",kti) = nrm;
        });  

        if(!kverts.hasProperty("nrm"))
            kverts.append_channels(cudaPol,{{"nrm",3}});
        TILEVEC_OPS::fill(cudaPol,kverts,"nrm",(T)0.0);
        cudaPol(zs::range(ktris.size()),[
                ktris = proxy<space>({},ktris),
                kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kti) mutable {
            auto ktri = ktris.pack(dim_c<3>,"inds",kti).reinterpret_bits(int_c);
            auto nrm = ktris.pack(dim_c<3>,"nrm",kti);
            for(int i = 0;i != 3;++i)
                for(int d = 0;d != 3;++d)
                    atomic_add(exec_cuda,&kverts("nrm",d,ktri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
        });
        cudaPol(zs::range(kverts.size()),[kverts = proxy<space>({},kverts)] ZS_LAMBDA(int kvi) mutable {
            auto nrm = kverts.pack(dim_c<3>,"nrm",kvi);
            nrm = nrm / (nrm.norm() + (T)1e-6);
            kverts.tuple(dim_c<3>,"nrm",kvi) = nrm;
        });   

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
        if(!verts.hasProperty("nrm"))
            verts.append_channels(cudaPol,{{"nrm",3}});
        TILEVEC_OPS::fill(cudaPol,verts,"nrm",(T)0.0);     
        cudaPol(zs::range(tris.size()),[
                tris = proxy<space>({},tris),
                verts = proxy<space>({},verts)] ZS_LAMBDA(int ti) mutable {
            auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
            auto nrm = tris.pack(dim_c<3>,"nrm",ti);
            for(int i = 0;i != 3;++i)
                for(int d = 0;d != 3;++d)
                    atomic_add(exec_cuda,&verts("nrm",d,tri[i]),nrm[d]/*/(T)kverts("valence",ktri[i])*/);
        });           
        cudaPol(zs::range(verts.size()),[verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) mutable {
            auto nrm = verts.pack(dim_c<3>,"nrm",vi);
            nrm = nrm / (nrm.norm() + (T)1e-6);
            verts.tuple(dim_c<3>,"nrm",vi) = nrm;
        });   


        cudaPol(zs::range(verts.size()),[
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            ktBvh = proxy<space>(ktBvh),
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            project_pos_tag = zs::SmallString(project_pos_tag),
            project_nrm_tag = zs::SmallString(project_nrm_tag),
            project_idx_tag = zs::SmallString(project_idx_tag),
            kinInCollisionEps = kinInCollisionEps,
            kinOutCollisionEps = kinOutCollisionEps,
            align_direction = align_direction,
            thickness = thickness] ZS_LAMBDA(int vi) mutable {
                if(verts.hasProperty("is_surf"))
                    if(verts("is_surf",vi) < (T)0.5)
                        return;
                if(verts.hasProperty("k_active"))// static unbind
                    if(verts("k_active",vi) < (T)0.5)
                        return;
                auto p = verts.pack(dim_c<3>,"x",vi);
                auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};
                auto min_dists = vec3::uniform(std::numeric_limits<T>::infinity());
                auto min_tri_indices = vec3i::uniform(-1);
                int nm_valid_tri_found = 0;
                
                vec3 min_bary[3] = {};
                auto pnrm = verts.pack(dim_c<3>,"nrm",vi);

                auto process_potential_closest_tris = [&](int kti) {
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti).reinterpret_bits(int_c);
                    if(kverts.hasProperty("k_active"))
                        for(int i = 0;i != 3;++i)
                            if(kverts("k_active",ktri[i]) < (T)0.5)
                                return;
                    auto kv0 = kverts.pack(dim_c<3>,"x",ktri[0]);
                    auto kv1 = kverts.pack(dim_c<3>,"x",ktri[1]);
                    auto kv2 = kverts.pack(dim_c<3>,"x",ktri[2]);     

                    vec3 bary{};
                    T distance = LSL_GEO::pointTriangleDistance(kv0,kv1,kv2,p,bary);
                    if(nm_valid_tri_found < 3 && distance > min_dists[nm_valid_tri_found])
                        return;
                    if(nm_valid_tri_found >= 3 && distance > min_dists[2])
                        return;
                     
                    auto seg = p - kv0;
                    auto knrm = ktris.pack(dim_c<3>,"nrm",kti);
                    auto dist = seg.dot(knrm);   

                    auto align = knrm.dot(pnrm);
                    if(pnrm.dot(knrm) < (T)0.5 && align_direction && dist < 0)
                        return;
                    if(pnrm.dot(knrm) > (T)-0.5 && !align_direction && dist < 0)
                        return;

                    auto collisionEps = dist > 0 ? kinOutCollisionEps : kinInCollisionEps;
                    if(distance > collisionEps)
                        return;          

                    if(kverts.hasProperty("k_fail"))
                        for(int i = 0;i != 3;++i)
                            if(kverts("k_fail",ktri[i]) > (T)0.5)
                                return;                    

                    auto ntris = ktris.pack(dim_c<3>,"ff_inds",kti).reinterpret_bits(int_c);
                    for(int i = 0;i != 3;++i){
                        auto nti = ntris[i];
                        auto edge_normal = vec3::zeros();
                        if(nti < 0){
                            edge_normal = knrm;
                        }else {
                            edge_normal = ktris.pack(dim_c<3>,"nrm",nti) + knrm;
                            edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                        }
                        auto ke0 = kverts.pack(dim_c<3>,"x",ktri[(i + 0) % 3]);
                        auto ke1 = kverts.pack(dim_c<3>,"x",ktri[(i + 1) % 3]);  
                        auto ke10 = ke1 - ke0;
                        auto bisector_normal = edge_normal.cross(ke10).normalized();

                        seg = p - kverts.pack(dim_c<3>,"x",ktri[(i + 0) % 3]);
                        if(bisector_normal.dot(seg) < 0)
                            return;
                    }    

                    auto insert_idx = nm_valid_tri_found < 3 ? nm_valid_tri_found : 2;
                    min_tri_indices[insert_idx] = kti;     
                    min_dists[insert_idx] = distance;
                    bary[0] = bary[0] < 0 ? (T)0 : bary[0];
                    bary[1] = bary[1] < 0 ? (T)0 : bary[1];
                    bary[2] = bary[2] < 0 ? (T)0 : bary[2];
                    bary = bary/bary.sum();
                    min_bary[insert_idx] = bary;

                    // sort the found_tris by distance
                    for(int i = insert_idx;i != 0;--i)
                            if(min_dists[i] < min_dists[i-1]){
                                auto tri_idx_tmp = min_tri_indices[i];
                                min_tri_indices[i] = min_tri_indices[i-1];
                                min_tri_indices[i-1] = tri_idx_tmp;

                                auto bary_tmp = min_bary[i];
                                min_bary[i] = min_bary[i-1];
                                min_bary[i-1] = bary_tmp;

                                auto dist_tmp = min_dists[i];
                                min_dists[i] = min_dists[i-1];
                                min_dists[i-1] = dist_tmp;                                
                            }

                    ++nm_valid_tri_found;
                };

                ktBvh.iter_neighbors(bv,process_potential_closest_tris); 
                if(nm_valid_tri_found == 0)
                    return;

                auto project_kv = vec3::zeros();
                auto project_knrm = vec3::zeros();
                int nm_insertion = 0;
                for(int i = 0;i != 3;++i)
                    if(min_tri_indices[i] >= 0)
                        ++nm_insertion;
                for(int ti = 0;ti != nm_insertion;++ti){
                    if(min_tri_indices[ti] > ktris.size())
                        printf("invalid min_tris_indices[%d] : %d ; nm : %d %d all : %d %d %d\n",ti,min_tri_indices[ti],nm_valid_tri_found,nm_insertion,
                            min_tri_indices[0],min_tri_indices[1],min_tri_indices[2]);
                    if(min_tri_indices[ti] < 0)
                        break;
                    auto min_tri = ktris.pack(dim_c<3>,"inds",min_tri_indices[ti]).reinterpret_bits(int_c);
                    for(int i = 0;i != 3;++i)
                        project_kv += kverts.pack(dim_c<3>,"x",min_tri[i]) * min_bary[ti][i] / (T)nm_insertion;
                    for(int i = 0;i != 3;++i)
                        project_knrm += kverts.pack(dim_c<3>,"nrm",min_tri[i]) * min_bary[ti][i];
                }      
                project_knrm /= (project_knrm.norm() + 1e-6);    
                if(nm_insertion == 3) {
                    vec3 pv[3] = {};
                    for(int ti = 0;ti != 3;++ti){
                        pv[ti] = vec3::zeros();
                        auto min_tri = ktris.pack(dim_c<3>,"inds",min_tri_indices[ti]).reinterpret_bits(int_c);
                        for(int i = 0;i != 3;++i)
                            pv[ti] += kverts.pack(dim_c<3>,"x",min_tri[i]) * min_bary[ti][i];
                    }
                    auto pv01 = pv[1] - pv[0];
                    auto pv02 = pv[2] - pv[0];
                    auto test_nrm = pv01.cross(pv02);
                    test_nrm = test_nrm / (test_nrm.norm() + 1e-6);
                    if(test_nrm.dot(project_knrm) < 0)
                        test_nrm = -test_nrm;
                    project_knrm = test_nrm;
                }

                verts.tuple(dim_c<3>,project_pos_tag,vi) = project_kv;
                verts.tuple(dim_c<3>,project_nrm_tag,vi) = project_knrm;
                verts(project_idx_tag,vi) = reinterpret_bits<T>(min_tri_indices[0]);                                           
        });



        cudaPol(zs::range(tris.size()),[
            verts = proxy<space>({},verts),
            tris = proxy<space>({},tris),
            ktBvh = proxy<space>(ktBvh),
            kverts = proxy<space>({},kverts),
            ktris = proxy<space>({},ktris),
            project_pos_tag = zs::SmallString(project_pos_tag),
            project_nrm_tag = zs::SmallString(project_nrm_tag),
            project_idx_tag = zs::SmallString(project_idx_tag),
            kinInCollisionEps = kinInCollisionEps,
            kinOutCollisionEps = kinOutCollisionEps,
            align_direction = align_direction,
            thickness = thickness] ZS_LAMBDA(int ti) mutable {
                auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
                for(int i = 0;i != 3;++i) {
                    if(verts.hasProperty("k_active"))// static unbind
                        if(verts("k_active",tri[i]) < (T)0.5)
                            return;
                    if(verts.hasProperty("is_surf"))
                        if(verts("is_surf",tri[i]) < (T)0.5)
                            return;                    
                }
                auto p = vec3::zeros();
                for(int i = 0;i != 3;++i)
                    p += verts.pack(dim_c<3>,"x",tri[i])/(T)3.0;
                auto bv = bv_t{get_bounding_box(p - thickness,p + thickness)};
                auto min_dists = vec3::uniform(std::numeric_limits<T>::infinity());
                auto min_tri_indices = vec3i::uniform(-1);
                int nm_valid_tri_found = 0;
                
                vec3 min_bary[3] = {};
                auto cnrm = tris.pack(dim_c<3>,"nrm",ti);

                auto process_potential_closest_tris = [&](int kti) {
                    auto ktri = ktris.pack(dim_c<3>,"inds",kti).reinterpret_bits(int_c);
                    if(kverts.hasProperty("k_active"))
                        for(int i = 0;i != 3;++i)
                            if(kverts("k_active",ktri[i]) < (T)0.5)
                                return;
                    auto kv0 = kverts.pack(dim_c<3>,"x",ktri[0]);
                    auto kv1 = kverts.pack(dim_c<3>,"x",ktri[1]);
                    auto kv2 = kverts.pack(dim_c<3>,"x",ktri[2]);     

                    vec3 bary{};
                    T distance = LSL_GEO::pointTriangleDistance(kv0,kv1,kv2,p,bary);
                    if(nm_valid_tri_found < 3 && distance > min_dists[nm_valid_tri_found])
                        return;
                    if(nm_valid_tri_found >= 3 && distance > min_dists[2])
                        return;
                     
                    auto seg = p - kv0;
                    auto knrm = ktris.pack(dim_c<3>,"nrm",kti);
                    auto dist = seg.dot(knrm);   

                    auto align = knrm.dot(cnrm);
                    if(cnrm.dot(knrm) < (T)0.5 && align_direction && dist < 0)
                        return;
                    if(cnrm.dot(knrm) > (T)-0.5 && !align_direction && dist < 0)
                        return;

                    auto collisionEps = dist > 0 ? kinOutCollisionEps : kinInCollisionEps;
                    if(distance > collisionEps)
                        return;          

                    if(kverts.hasProperty("k_fail"))
                        for(int i = 0;i != 3;++i)
                            if(kverts("k_fail",ktri[i]) > (T)0.5)
                                return;                    

                    auto ntris = ktris.pack(dim_c<3>,"ff_inds",kti).reinterpret_bits(int_c);
                    for(int i = 0;i != 3;++i){
                        auto nti = ntris[i];
                        auto edge_normal = vec3::zeros();
                        if(nti < 0){
                            edge_normal = knrm;
                        }else {
                            edge_normal = ktris.pack(dim_c<3>,"nrm",nti) + knrm;
                            edge_normal = edge_normal/(edge_normal.norm() + (T)1e-6);
                        }
                        auto ke0 = kverts.pack(dim_c<3>,"x",ktri[(i + 0) % 3]);
                        auto ke1 = kverts.pack(dim_c<3>,"x",ktri[(i + 1) % 3]);  
                        auto ke10 = ke1 - ke0;
                        auto bisector_normal = edge_normal.cross(ke10).normalized();

                        seg = p - kverts.pack(dim_c<3>,"x",ktri[(i + 0) % 3]);
                        if(bisector_normal.dot(seg) < 0)
                            return;
                    }    

                    auto insert_idx = nm_valid_tri_found < 3 ? nm_valid_tri_found : 2;
                    min_tri_indices[insert_idx] = kti;     
                    min_dists[insert_idx] = distance;
                    bary[0] = bary[0] < 0 ? (T)0 : bary[0];
                    bary[1] = bary[1] < 0 ? (T)0 : bary[1];
                    bary[2] = bary[2] < 0 ? (T)0 : bary[2];
                    bary = bary/bary.sum();
                    min_bary[insert_idx] = bary;

                    // sort the found_tris by distance
                    for(int i = insert_idx;i != 0;--i)
                            if(min_dists[i] < min_dists[i-1]){
                                auto tri_idx_tmp = min_tri_indices[i];
                                min_tri_indices[i] = min_tri_indices[i-1];
                                min_tri_indices[i-1] = tri_idx_tmp;

                                auto bary_tmp = min_bary[i];
                                min_bary[i] = min_bary[i-1];
                                min_bary[i-1] = bary_tmp;

                                auto dist_tmp = min_dists[i];
                                min_dists[i] = min_dists[i-1];
                                min_dists[i-1] = dist_tmp;                                
                            }

                    ++nm_valid_tri_found;
                };

                ktBvh.iter_neighbors(bv,process_potential_closest_tris); 
                if(nm_valid_tri_found == 0)
                    return;

                auto project_kv = vec3::zeros();
                auto project_knrm = vec3::zeros();
                int nm_insertion = 0;
                for(int i = 0;i != 3;++i)
                    if(min_tri_indices[i] >= 0)
                        ++nm_insertion;
                for(int ti = 0;ti != nm_insertion;++ti){
                    if(min_tri_indices[ti] > ktris.size())
                        printf("invalid min_tris_indices[%d] : %d ; nm : %d %d all : %d %d %d\n",ti,min_tri_indices[ti],nm_valid_tri_found,nm_insertion,
                            min_tri_indices[0],min_tri_indices[1],min_tri_indices[2]);
                    if(min_tri_indices[ti] < 0)
                        break;
                    auto min_tri = ktris.pack(dim_c<3>,"inds",min_tri_indices[ti]).reinterpret_bits(int_c);
                    for(int i = 0;i != 3;++i)
                        project_kv += kverts.pack(dim_c<3>,"x",min_tri[i]) * min_bary[ti][i] / (T)nm_insertion;
                    for(int i = 0;i != 3;++i)
                        project_knrm += kverts.pack(dim_c<3>,"nrm",min_tri[i]) * min_bary[ti][i];
                }      
                project_knrm /= (project_knrm.norm() + 1e-6);    
                if(nm_insertion == 3) {
                    vec3 pv[3] = {};
                    for(int ti = 0;ti != 3;++ti){
                        pv[ti] = vec3::zeros();
                        auto min_tri = ktris.pack(dim_c<3>,"inds",min_tri_indices[ti]).reinterpret_bits(int_c);
                        for(int i = 0;i != 3;++i)
                            pv[ti] += kverts.pack(dim_c<3>,"x",min_tri[i]) * min_bary[ti][i];
                    }
                    auto pv01 = pv[1] - pv[0];
                    auto pv02 = pv[2] - pv[0];
                    auto test_nrm = pv01.cross(pv02);
                    test_nrm = test_nrm / (test_nrm.norm() + 1e-6);
                    if(test_nrm.dot(project_knrm) < 0)
                        test_nrm = -test_nrm;
                    project_knrm = test_nrm;
                }

                tris.tuple(dim_c<3>,project_pos_tag,ti) = project_kv;
                // printf("project_kv : %f %f %f\n",(float)project_kv[0],(float)project_kv[1],(float)project_kv[2]);
                tris.tuple(dim_c<3>,project_nrm_tag,ti) = project_knrm;
                tris(project_idx_tag,ti) = reinterpret_bits<T>(min_tri_indices[0]);                                           
        });

        set_output("zsparticles",zsparticles);
        set_output("kboundary",kboundary);        
    }    
};


ZENDEFNODE(ZSSurfaceClosestPointsGrp, {
                                  {
                                    "zsparticles",
                                    "kboundary",
                                    {"float","kinInColEps","0.001"},
                                    {"float","kinOutColEps","0.001"},
                                  },
                                  {"zsparticles","kboundary"},
                                  {
                                    {"string","project_pos_tag","project_pos_tag"},
                                    {"string","project_nrm_tag","project_nrm_tag"},
                                    {"string","project_idx_tag","project_idx_tag"},
                                    {"bool","align_direction","1"},
                                  },
                                  {"ZSGeometry"}});

};