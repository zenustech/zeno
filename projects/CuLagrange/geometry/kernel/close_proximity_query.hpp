#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bvh.hpp"

#include <iostream>

#include "geo_math.hpp"

namespace zeno {

    template<typename Pol,
        typename PosTileVec,
        typename T = typename PosTileVec::value_type,
        typename CloseProximityHash>
    void retrieve_intersected_sphere_pairs(Pol& pol,
        const PosTileVec& verts,  const zs::SmallString& xtag,  const T& uniform_radius,     const zs::SmallString& variable_radius_attr,
        const PosTileVec& nverts, const zs::SmallString& nxtag, const T& nei_uniform_radius, const zs::SmallString& nei_variable_radius_attr,
        CloseProximityHash& res) {
            using namespace zs;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            constexpr auto space = Pol::exec_tag::value;
            Vector<bv_t> bvs{verts.get_allocator(),0};
            if(verts.hasProperty(variable_radius_attr))
                retrieve_bounding_volumes(pol,verts,bvs,variable_radius_attr,xtag);
            else
                retrieve_bounding_volumes(pol,verts,bvs,uniform_radius,xtag);

            
            ZenoParticles::lbvh_t bvh{};
            bvh.build(pol,bvs);
            
            res.reset(pol,true);
            pol(zs::range(nverts.size()),[
                res = proxy<space>(res),
                bvh = proxy<space>(bvh),
                bvs = proxy<space>(bvs),
                nverts = proxy<space>({},nverts),
                nxtag = nxtag,
                nei_variable_radius_attr = nei_variable_radius_attr,
                has_nei_variable_radius = nverts.hasProperty(nei_variable_radius_attr),
                uniform_radius = uniform_radius,
                verts = proxy<space>({},verts),
                xtag = xtag,
                variable_radius_attr = variable_radius_attr,
                has_variable_radius = verts.hasProperty(variable_radius_attr),
                nei_uniform_radius = nei_uniform_radius] ZS_LAMBDA(auto nvi) mutable {
                    auto nx = nverts.pack(dim_c<3>,nxtag,nvi);
                    auto nradius = nei_uniform_radius;
                    if(has_nei_variable_radius)
                        nradius = nverts(nei_variable_radius_attr,nvi);
                    bv_t nbv{get_bounding_box(nx,nx)};
                    nbv._min -= nradius;
                    nbv._max += nradius;

                    auto process_potential_intersected_pairs = [&](auto vi) mutable {
                        auto x = verts.pack(dim_c<3>,xtag,vi);
                        auto dist = (x - nx).norm();
                        auto radius = uniform_radius;
                        if(has_variable_radius)
                            radius = verts(variable_radius_attr,vi);
                        if(dist < radius + nradius)
                            res.insert(zs::vec<int,2>{vi,nvi});
                    };

                    bvh.iter_neighbors(nbv,process_potential_intersected_pairs);
            });
    }

    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename PTHashMap,
        typename TriBvh,
        typename T = typename PosTileVec::value_type>
    void detect_PKT_close_proximity(Pol& pol,
        const PosTileVec& verts,const zs::SmallString& xtag,
        const PosTileVec& kverts,const zs::SmallString& kxtag,
        const TriTileVec& ktris,
        const T& thickness,
        const TriBvh& ktriBvh,
        PTHashMap& csPKT,
        const zs::SmallString& group_id,
        bool find_closest_ktri = false) {
            using namespace zs;
            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            // constexpr auto exec_tag = wrapv<space>{};
            using vec3 = zs::vec<T,3>;
            using vec4 = zs::vec<T,4>;
            using vec4i = zs::vec<int,4>;
            constexpr auto eps = (T)1e-6;

            csPKT.reset(pol,true);

            pol(zs::range(verts.size()),[
                find_closest_ktri = find_closest_ktri,
                xtag = xtag,
                group_id = group_id,
                verts = proxy<space>({},verts),
                kxtag = kxtag,
                kverts = proxy<space>({},kverts),
                ktris = proxy<space>({},ktris),
                thickness = thickness,
                thickness2 = thickness * thickness,
                eps = eps,
                ktriBvh = proxy<space>(ktriBvh),
                csPKT = proxy<space>(csPKT)] ZS_LAMBDA(int vi) mutable {
                    auto p = verts.pack(dim_c<3>,xtag,vi);
                    auto bv = bv_t{get_bounding_box(p - thickness/(T)2,p + thickness/(T)2)};

                    int closest_kti = -1;
                    T min_dist2 = std::numeric_limits<T>::max();

                    auto do_close_proximity_detection = [&](int kti) {
                        auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);

                        if(verts.hasProperty(group_id) && kverts.hasProperty(group_id)) {
                            for(int i = 0;i != 3;++i)
                                if(zs::abs(verts(group_id,vi) - kverts(group_id,ktri[i])) > 0.5)
                                    return;
                        }

                        vec3 kts[3] = {};
                        for(int i = 0;i != 3;++i)
                            kts[i] = kverts.pack(dim_c<3>,kxtag,ktri[i]);
                        
                        auto ktnrm = LSL_GEO::facet_normal(kts[0],kts[1],kts[2]);
                        auto project_dist = zs::abs((p - kts[0]).dot(ktnrm));
                        if(project_dist > thickness)
                            return;

                        vec3 bary{};

                        LSL_GEO::get_triangle_vertex_barycentric_coordinates(kts[0],kts[1],kts[2],p,bary);
                        for(int i = 0;i != 3;++i) {
                            bary[i] = bary[i] > 1 + eps ? 1 : bary[i];
                            bary[i] = bary[i] < -eps ? 0 : bary[i];
                        }

                        bary /= (bary[0] + bary[1] + bary[2]);

                        auto pr = p;
                        for(int i = 0;i != 3;++i)
                            pr -= bary[i] * kts[i];

                        auto dist2 = pr.l2NormSqr();

                        if(dist2 > thickness2)
                            return;

                        if(dist2 < min_dist2) {
                            min_dist2 = dist2;
                            closest_kti = kti;
                        }

                        if(!find_closest_ktri)
                            csPKT.insert(zs::vec<int,2>{vi,kti});
                    };
                    ktriBvh.iter_neighbors(bv,do_close_proximity_detection);


                    if(find_closest_ktri && closest_kti >= 0)
                        csPKT.insert(zs::vec<int,2>{vi,closest_kti});
            });
        }

    template<typename Pol,
        typename PosTileVec,
        typename TriTileVec,
        typename PTHashMap,
        typename TriBvh,
        typename T = typename PosTileVec::value_type>
    void detect_PKT_close_proximity_with_two_sided_path(Pol& pol,
        const PosTileVec& verts,const zs::SmallString& xtag,
        const PosTileVec& kverts,const zs::SmallString& kxtag,const zs::SmallString& kvtag,
        const TriTileVec& ktris,
        const T& thickness,
        const TriBvh& ktriBvh,
        PTHashMap& csPKT,
        const zs::SmallString& group_id,
        bool find_closest_ktri = false) {
            using namespace zs;
            constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
            using bv_t = typename ZenoParticles::lbvh_t::Box;
            // constexpr auto exec_tag = wrapv<space>{};
            using vec3 = zs::vec<T,3>;
            using vec4 = zs::vec<T,4>;
            using vec4i = zs::vec<int,4>;
            constexpr auto eps = (T)1e-6;

            csPKT.reset(pol,true);

            pol(zs::range(verts.size()),[
                find_closest_ktri = find_closest_ktri,
                xtag = xtag,
                group_id = group_id,
                verts = proxy<space>({},verts),
                kxtag = kxtag,
                kverts = proxy<space>({},kverts),
                ktris = proxy<space>({},ktris),
                thickness = thickness,
                thickness2 = thickness * thickness,
                eps = eps,
                ktriBvh = proxy<space>(ktriBvh),
                csPKT = proxy<space>(csPKT)] ZS_LAMBDA(int vi) mutable {
                    auto p = verts.pack(dim_c<3>,xtag,vi);
                    auto bv = bv_t{get_bounding_box(p - thickness/(T)2,p + thickness/(T)2)};

                    int closest_kti = -1;
                    T min_dist2 = std::numeric_limits<T>::max();

                    auto do_close_proximity_detection = [&](int kti) {
                        auto ktri = ktris.pack(dim_c<3>,"inds",kti,int_c);

                        if(verts.hasProperty(group_id) && kverts.hasProperty(group_id)) {
                            for(int i = 0;i != 3;++i)
                                if(zs::abs(verts(group_id,vi) - kverts(group_id,ktri[i])) > 0.5)
                                    return;
                        }

                        vec3 kts[3] = {};
                        for(int i = 0;i != 3;++i)
                            kts[i] = kverts.pack(dim_c<3>,kxtag,ktri[i]);
                        
                        auto ktnrm = LSL_GEO::facet_normal(kts[0],kts[1],kts[2]);
                        auto project_dist = zs::abs((p - kts[0]).dot(ktnrm));
                        if(project_dist > thickness)
                            return;

                        vec3 bary{};

                        LSL_GEO::get_triangle_vertex_barycentric_coordinates(kts[0],kts[1],kts[2],p,bary);
                        for(int i = 0;i != 3;++i) {
                            bary[i] = bary[i] > 1 + eps ? 1 : bary[i];
                            bary[i] = bary[i] < -eps ? 0 : bary[i];
                        }

                        bary /= (bary[0] + bary[1] + bary[2]);

                        auto pr = p;
                        for(int i = 0;i != 3;++i)
                            pr -= bary[i] * kts[i];

                        auto dist2 = pr.l2NormSqr();

                        if(dist2 > thickness2)
                            return;

                        if(dist2 < min_dist2) {
                            min_dist2 = dist2;
                            closest_kti = kti;
                        }

                        if(!find_closest_ktri)
                            csPKT.insert(zs::vec<int,2>{vi,kti});
                    };
                    ktriBvh.iter_neighbors(bv,do_close_proximity_detection);


                    if(find_closest_ktri && closest_kti >= 0)
                        csPKT.insert(zs::vec<int,2>{vi,closest_kti});
            });
        }

};