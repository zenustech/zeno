#include "Structures.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>

namespace zeno {

template <typename IV>
static zs::Vector<zs::AABBBox<3, float>>
retrieve_bounding_volumes(zs::OmpExecutionPolicy &pol, const std::vector<vec3f> &pos, const std::vector<IV> &eles, float thickness) {
    using namespace zs;
    using T = float;
    using bv_t = AABBBox<3, T>;
    constexpr auto space = execspace_e::openmp;
    constexpr auto edim = std::tuple_size_v<IV>;
    zs::Vector<bv_t> ret{eles.size()};
    pol(range(eles.size()), [&, bvs = proxy<space>(ret), ts_c = wrapv<edim>{}, thickness](int i) {
        using vec3 = zs::vec<float, 3>;
        auto inds = eles[i];
        auto x0 = vec3::from_array(pos[inds[0]]);
        bv_t bv{x0 - thickness, x0 + thickness};
        for (int d = 1; d < RM_CVREF_T(ts_c)::value; ++d) {
            merge(bv, vec3::from_array(pos[inds[d]]) - thickness);
            merge(bv, vec3::from_array(pos[inds[d]]) + thickness);
        }
        bvs[i] = bv;
    });
    return ret;
}

static zs::Vector<zs::AABBBox<3, float>>
retrieve_bounding_volumes(zs::OmpExecutionPolicy &pol, const std::vector<vec3f> &pos, float thickness) {
    using namespace zs;
    using T = float;
    using bv_t = AABBBox<3, T>;
    constexpr auto space = execspace_e::openmp;
    zs::Vector<bv_t> ret{pos.size()};
    pol(range(pos.size()), [&, bvs = proxy<space>(ret), thickness](int i) {
        using vec3 = zs::vec<float, 3>;
        auto x0 = vec3::from_array(pos[i]);
        bv_t bv{x0 - thickness, x0 + thickness};
        bvs[i] = bv;
    });
    return ret;
}

struct PrimitiveProject : INode {
    virtual void apply() override {
        using bvh_t = zs::LBvh<3, int, float>;
        using bv_t = typename bvh_t::Box;

        auto prim = get_input<PrimitiveObject>("prim");
        auto targetPrim = get_input<PrimitiveObject>("targetPrim");
        auto limit = get_input2<float>("limit");
        auto nrmAttr = get_input2<std::string>("nrmAttr");
        auto side = get_input2<std::string>("side");

        int sideNo = 0;
        if (side == "closest")
            sideNo = 0;
        else if (side == "farthest")
            sideNo = 1;

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        auto &pos = prim->attr<zeno::vec3f>("pos");
        const auto &targetPos = targetPrim->attr<zeno::vec3f>("pos");
        const auto &tris = targetPrim->tris.values;

        bvh_t targetBvh;
        auto tBvs = retrieve_bounding_volumes(pol, targetPos, tris, 0.f);
        targetBvh.build(pol, tBvs);

        /// @note cut off hits farther than distance [limit]
        if (limit <= 0)
            limit = std::numeric_limits<float>::infinity();

        auto const &nrm = prim->attr<vec3f>(nrmAttr);

        pol(range(pos.size()), [&, bvh = proxy<space>(targetBvh), sideNo](size_t i) {
            using vec3 = zs::vec<float, 3>;
            auto ro = vec3::from_array(pos[i]);
            auto rd = vec3::from_array(nrm[i]).normalized();
            float dist{0};
            if (sideNo == 1) {
                bvh.ray_intersect(ro, rd, [&](int triNo) {
                    auto tri = tris[triNo];
                    auto t0 = vec3::from_array(targetPos[tri[0]]);
                    auto t1 = vec3::from_array(targetPos[tri[1]]);
                    auto t2 = vec3::from_array(targetPos[tri[2]]);
                    if (auto d = ray_tri_intersect(ro, rd, t0, t1, t2); d < limit && d > dist) {
                        dist = d;
                    }
                });
            } else if (sideNo == 0) {
                bvh.ray_intersect(ro, rd, [&](int triNo) {
                    auto tri = tris[triNo];
                    auto t0 = vec3::from_array(targetPos[tri[0]]);
                    auto t1 = vec3::from_array(targetPos[tri[1]]);
                    auto t2 = vec3::from_array(targetPos[tri[2]]);
                    if (auto d = ray_tri_intersect(ro, rd, t0, t1, t2); d < limit && (d < dist || dist == 0)) {
                        dist = d;
                    }
                });
            }
            pos[i] = (ro + dist * rd).to_array();
        });

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveProject, {
                                 {
                                     {"PrimitiveObject", "prim"},
                                     {"PrimitiveObject", "targetPrim"},
                                     {"string", "nrmAttr", "nrm"},
                                     {"float", "limit", "0"},
                                     {"enum closest farthest", "side", "farthest"},
                                 },
                                 {
                                     {"PrimitiveObject", "prim"},
                                 },
                                 {},
                                 {"primitive"},
                             });

#if 1
struct QueryClosestPrimitive : zeno::INode {
    using zsbvh_t = ZenoLinearBvh;
    using bvh_t = zsbvh_t::lbvh_t;
    using bv_t = bvh_t::Box;

    struct KVPair {
        float dist;
        int pid;
        bool operator<(const KVPair &o) const noexcept {
            return dist < o.dist;
        }
    };
    void apply() override {
        auto targetPrim = get_input<PrimitiveObject>("targetPrim");
        auto &userData = targetPrim->userData();
        auto bvhTag = get_input2<std::string>("bvh_tag");
        auto zsbvh = std::dynamic_pointer_cast<zsbvh_t>(userData.get(bvhTag));
        bvh_t &lbvh = zsbvh->get();

        auto line = std::make_shared<PrimitiveObject>();

        using Ti = typename bvh_t::index_type;
        auto pol = zs::omp_exec();
        Ti pid = 0;
        Ti bvhId = -1;
        float dist = std::numeric_limits<float>::max();
        if (has_input<PrimitiveObject>("prim")) {
            auto prim = get_input<PrimitiveObject>("prim");

            auto idTag = get_input2<std::string>("idTag");
            auto distTag = get_input2<std::string>("distTag");
            auto weightTag = get_input2<std::string>("weightTag");

            auto &bvhids = prim->add_attr<float>(idTag);
            auto &dists = prim->add_attr<float>(distTag);
            auto &ws = prim->add_attr<zeno::vec3f>(weightTag);

            std::vector<KVPair> kvs(prim->size());
            std::vector<Ti> ids(prim->size(), -1);
            pol(zs::range(prim->size()), [&, lbvh = zs::proxy<zs::execspace_e::openmp>(lbvh), et = zsbvh->et](int i) {
                using vec3 = zs::vec<float, 3>;
                kvs[i].dist = zs::limits<float>::max();
                kvs[i].pid = i;
                auto pi = vec3::from_array(prim->verts[i]);
                lbvh.find_nearest(pi, [&ids, &kvs, &pi, &targetPrim, i, et](int j, float &dist, int &idx) {
                    float d = zs::limits<float>::max();
                    if (et == ZenoLinearBvh::point) {
                        d = zs::dist_pp(pi, vec3::from_array(targetPrim->verts[j]));
                    } else if (et == ZenoLinearBvh::curve) {
                        auto line = targetPrim->lines[j];
                        d = zs::dist_pe_unclassified(pi, vec3::from_array(targetPrim->verts[line[0]]),
                                                     vec3::from_array(targetPrim->verts[line[1]]));
                    } else if (et == ZenoLinearBvh::surface) {
                        auto tri = targetPrim->tris[j];
                        d = zs::dist_pt(pi, vec3::from_array(targetPrim->verts[tri[0]]),
                                        vec3::from_array(targetPrim->verts[tri[1]]),
                                        vec3::from_array(targetPrim->verts[tri[2]]));
                    } else if (et == ZenoLinearBvh::tet) {
                        throw std::runtime_error("tet distance query not implemented yet!");
                    }
                    if (d < dist) {
                        dist = d;
                        idx = j;
                        ids[i] = j;
                        kvs[i].dist = d;
                    }
                });
                // record info as attribs
                bvhids[i] = ids[i];
                dists[i] = kvs[i].dist;
            });

            KVPair mi{std::numeric_limits<float>::max(), -1};
// ref:
// https://stackoverflow.com/questions/28258590/using-openmp-to-get-the-index-of-minimum-element-parallelly
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp declare reduction(minimum:KVPair                                   \
                              : omp_out = omp_in < omp_out ? omp_in : omp_out) \
    initializer(omp_priv = KVPair{std::numeric_limits <float>::max(), -1})
#pragma omp parallel for reduction(minimum : mi)
#endif
#endif
            for (Ti i = 0; i < kvs.size(); ++i) {
                if (kvs[i].dist < mi.dist)
                    mi = kvs[i];
            }
            pid = mi.pid;
            dist = mi.dist;
            bvhId = ids[pid];
            line->verts.push_back(prim->verts[pid]);
#if 0
      fmt::print("done nearest reduction. dist: {}, bvh[{}] (of {})-prim[{}]"
                 "(of {})\n",
                 dist, bvhId, lbvh->getNumLeaves(), pid, prim->size());
#endif
        } else if (has_input<NumericObject>("prim")) {
            auto p = get_input<NumericObject>("prim")->get<vec3f>();
            using vec3 = zs::vec<float, 3>;
            auto pi = vec3::from_array(p);
            auto lbvhv = zs::proxy<zs::execspace_e::host>(lbvh);
            lbvhv.find_nearest(pi, [&, et = zsbvh->et](int j, float &dist_, int &idx) {
                using vec3 = zs::vec<float, 3>;
                float d = zs::limits<float>::max();
                if (et == ZenoLinearBvh::point) {
                    d = zs::dist_pp(pi, vec3::from_array(targetPrim->verts[j]));
                } else if (et == ZenoLinearBvh::curve) {
                    auto line = targetPrim->lines[j];
                    d = zs::dist_pe_unclassified(pi, vec3::from_array(targetPrim->verts[line[0]]),
                                                 vec3::from_array(targetPrim->verts[line[1]]));
                } else if (et == ZenoLinearBvh::surface) {
                    auto tri = targetPrim->tris[j];
                    d = zs::dist_pt(pi, vec3::from_array(targetPrim->verts[tri[0]]),
                                    vec3::from_array(targetPrim->verts[tri[1]]),
                                    vec3::from_array(targetPrim->verts[tri[2]]));
                } else if (et == ZenoLinearBvh::tet) {
                    throw std::runtime_error("tet distance query not implemented yet!");
                }
                if (d < dist_) {
                    dist_ = d;
                    idx = j;

                    bvhId = j;
                    dist = d;
                }
            });
            line->verts.push_back(p);
        } else
            throw std::runtime_error("unknown primitive kind (only supports "
                                     "PrimitiveObject and NumericObject::vec3f).");

        // line->verts.push_back(lbvh->retrievePrimitiveCenter(bvhId, w));
        // line->lines.push_back({0, 1});

        set_output("primid", std::make_shared<NumericObject>(pid));
        set_output("bvh_primid", std::make_shared<NumericObject>(bvhId));
        set_output("dist", std::make_shared<NumericObject>(dist));
        // set_output("bvh_prim", lbvh->retrievePrimitive(bvhId));
        set_output("segment", std::move(line));
    }
};

ZENDEFNODE(QueryClosestPrimitive, {
                                      {{"prim"},
                                       {"prim", "targetPrim"},
                                       {"string", "idTag", "bvh_id"},
                                       {"string", "distTag", "bvh_dist"},
                                       {"string", "weightTag", "bvh_ws"},
                                       {"string", "bvh_tag", "bvh"}},
                                      {{"NumericObject", "primid"},
                                       {"NumericObject", "bvh_primid"},
                                       {"NumericObject", "dist"},
                                       {"PrimitiveObject", "bvh_prim"},
                                       {"PrimitiveObject", "segment"}},
                                      {},
                                      {"zenofx"},
                                  });
#endif

struct EmbedPrimitiveBvh : zeno::INode {
  virtual void apply() override {
    using zsbvh_t = ZenoLinearBvh;
    using bvh_t = zsbvh_t::lbvh_t;
    using bv_t = bvh_t::Box;

    auto prim = get_input<zeno::PrimitiveObject>("prim");
    auto &userData = prim->userData();
    float thickness =
        has_input("thickness")
            ? get_input<zeno::NumericObject>("thickness")->get<float>()
            : 0.f;
    auto primType = get_input2<std::string>("prim_type");
    auto bvhTag = get_input2<std::string>("bvh_tag");

    auto pol = zs::omp_exec();

    zs::Vector<bv_t> bvs;
    std::shared_ptr<zsbvh_t> zsbvh;
    ZenoLinearBvh::element_e et = ZenoLinearBvh::point;
    if (primType == "point") {
        bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), thickness);
        et = ZenoLinearBvh::point;
    } else if (primType == "line") {
        bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), prim->lines.values, thickness);
        et = ZenoLinearBvh::curve;
    } else if (primType == "tri") {
        bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), prim->tris.values, thickness);
        et = ZenoLinearBvh::surface;
    } else if (primType == "quad") {
        bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), prim->quads.values, thickness);
        et = ZenoLinearBvh::tet;
    }
    if (!userData.has(bvhTag)) {    // build
        zsbvh = std::make_shared<zsbvh_t>();
        zsbvh->et = et; 
        bvh_t &bvh = zsbvh->get();
        bvh.build(pol, bvs);
        userData.set(bvhTag, zsbvh);
    } else {    // refit
        zsbvh = std::dynamic_pointer_cast<zsbvh_t>(userData.get(bvhTag));
        zsbvh->et = et; 
        bvh_t &bvh = zsbvh->get();
        bvh.refit(pol, bvs);
    }
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(EmbedPrimitiveBvh,
           {
               {{"PrimitiveObject", "prim"}, {"float", "thickness", "0"}, {"enum point line tri quad", "prim_type", "auto"}, {"string", "bvh_tag", "bvh"}},
               {{"PrimitiveObject", "prim"}},
               {},
               {"zenofx"},
           });


} // namespace zeno