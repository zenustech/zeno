#include <zeno/core/INode.h>
#include <zeno/core/defNode.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include "bvh.h"
#include "distanceQueries.h"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/resource/Filesystem.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <chrono>
#include <zeno/utils/log.h>

namespace zeno {

///
/// ref: https://github.com/ingowald/closestSurfacePointQueries
///
struct QueryNearestPoints : INode {
    void apply() override {
        auto points = get_input<PrimitiveObject>("points");
        auto prim = get_input<PrimitiveObject>("target_prim");
        auto idTag = get_input2<std::string>("idTag");
        auto distTag = get_input2<std::string>("distTag");
        auto cpTag = get_input2<std::string>("closestPointTag");

        using v3f = ospcommon::vec3f;
        using v3i = ospcommon::vec3i;

        static_assert(sizeof(v3f) == sizeof(zeno::vec3f), "v3f, vec3f size mismatch!");
        static_assert(sizeof(v3i) == sizeof(zeno::vec3i), "v3i, vec3i size mismatch!");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &pos = points->attr<zeno::vec3f>("pos");
        auto &ids = points->add_attr<int>(idTag);
        auto &dists = points->add_attr<float>(distTag);
        auto &cps = points->add_attr<zeno::vec3f>(cpTag);

        auto &vertices = prim->verts.values;

#if 0
        std::vector<v3i> vertIndex(prim->size());
        pol(enumerate(vertIndex), [](int id, v3i &v) { v = v3i{id, id, id}; });

        ///
        /// begin routine
        ///
        auto begin = std::chrono::system_clock::now();
        // create the actual scene:
        distance_query_scene scene =
            rtdqNewTriangleMeshfi(&vertices[0][0], &vertices[0][1], &vertices[0][2], 3, &vertIndex[0].x,
                                  &vertIndex[0].y, &vertIndex[0].z, 3, vertIndex.size());
        auto done_build = std::chrono::system_clock::now();

        // perform the queries - all together, in a single thread
        rtdqComputeClosestPointsfi(scene, &cps[0][0], &cps[0][1], &cps[0][2], 3, &dists[0], 1, &ids[0], 1, &pos[0][0],
                                   &pos[0][1], &pos[0][2], 3, points->size());
        auto done_all = std::chrono::system_clock::now();

        std::chrono::duration<double> buildTime = done_build - begin;
        std::chrono::duration<double> queryTime = done_all - done_build;
        std::cout << "time to build tree " << buildTime.count() << "s" << std::endl;
        std::cout << "time to query " << points->size() << " points: " << queryTime.count() << "s" << std::endl;
        std::cout << "(this is " << (queryTime.count() / points->size()) << " seconds/prim)" << std::endl;

        rtdqDestroy(scene);
#endif
        ///
        /// done routine
        ///

#if 1
        using Box = AABBBox<3, float>;
        Box gbv;
        CppTimer timer;
        timer.tick();
        Vector<float> ret{1};
        Vector<float> gmins{vertices.size()}, gmaxs{vertices.size()};
        for (int d = 0; d != 3; ++d) {
            pol(enumerate(gmins, gmaxs), [&vertices, d] ZS_LAMBDA(int i, float &gmin, float &gmax) {
                gmin = vertices[i][d];
                gmax = vertices[i][d];
            });
            reduce(pol, std::begin(gmins), std::end(gmins), std::begin(ret), limits<float>::max(), getmin<float>{});
            gbv._min[d] = ret.getVal();
            reduce(pol, std::begin(gmaxs), std::end(gmaxs), std::begin(ret), limits<float>::lowest(), getmax<float>{});
            gbv._max[d] = ret.getVal();
        }
        int axis = 0; // x-axis by default
        auto dis = gbv._max[0] - gbv._min[0];
        for (int d = 1; d < 3; ++d) {
            if (auto tmp = gbv._max[d] - gbv._min[d]; tmp > dis) { // select the longest orientation
                dis = tmp;
                axis = d;
            }
        }
        timer.tock("done prep for ordering");
        fmt::print("box: {}, {}, {} - {}, {}, {}. pick axis [{}]\n", gbv._min[0], gbv._min[1], gbv._min[2], gbv._max[0],
                   gbv._max[1], gbv._max[2], axis);

        std::vector<float> keys(vertices.size());
        std::vector<int> indices(vertices.size());
        pol(enumerate(keys, indices), [&vertices, axis](int id, float &key, int &idx) {
            key = vertices[id][axis];
            idx = id;
        });
        //
        timer.tick();
#if 1
        merge_sort_pair(pol, std::begin(keys), std::begin(indices), vertices.size(), std::less<float>{});
#else
        std::vector<float> sortedKeys(vertices.size());
        std::vector<int> sortedIndices(vertices.size());
        radix_sort_pair(pol, std::begin(keys), std::begin(indices), std::begin(sortedKeys), std::begin(sortedIndices),
                        vertices.size());
        keys = std::move(sortedKeys);
        indices = std::move(sortedIndices);
#endif
        timer.tock(fmt::format("sort {} points", vertices.size()));

        {
            int cnt = 0;
            for (int i = 0; i < vertices.size() - 1; ++i) {
                if ((keys[i] >= limits<float>::epsilon() || keys[i] <= -limits<float>::epsilon()) &&
                    (keys[i + 1] >= limits<float>::epsilon() || keys[i + 1] <= -limits<float>::epsilon()))
                    if (keys[i] > keys[i + 1]) {
                        printf("order is wrong at [%d] ... %e, %e...\n", i, keys[i], keys[i + 1]);
                        cnt++;
                    }
                if (cnt >= 1000)
                    break;
            }
        }

        timer.tick();
        std::vector<zeno::vec3f> xs(vertices.size());
        pol(zip(indices, xs), [&vertices](int oid, zeno::vec3f &p) { p = vertices[oid]; });
        timer.tock(fmt::format("reorder {} points", vertices.size()));

        timer.tick();
        std::vector<int> locs(pos.size());
        auto locate = [&keys](float v) -> int {
            int left = 0, right = keys.size();
            while (left < right) {
                auto mid = left + (right - left) / 2;
                if (keys[mid] > v)
                    right = mid;
                else
                    left = mid + 1;
            }
            if (left < keys.size()) {
                if (keys[left] > v)
                    left--;
            } else
                left = keys.size() - 1;
            // left could be -1
            return left;
        };
        pol(zip(pos, locs), [&locate, axis](const zeno::vec3f &xi, int &loc) { loc = locate(xi[axis]); });
        timer.tock("find st locations");

        timer.tick();
        pol(enumerate(pos, locs, dists, ids, cps),
            [&xs, &indices, axis](int i, const auto &xi, const int loc, float &dist, int &id, vec3f &cp) {
                int l = loc + 1;
                float d2 = limits<float>::max();
                int j = -1;
                int cnt = 0;
                while (l < xs.size() && cnt++ < 128) {
                    if (zs::sqr(xs[l][axis] - xi[axis]) > d2)
                        break;
                    if (auto tmp = lengthSquared(xs[l] - xi); tmp < d2) {
                        d2 = tmp;
                        j = l;
                    }
                    l++;
                }
                cnt = 0;
                l = loc;
                while (l >= 0 && cnt++ < 128) {
                    if (zs::sqr(xi[axis] - xs[l][axis]) > d2)
                        break;
                    if (auto tmp = lengthSquared(xs[l] - xi); tmp < d2) {
                        d2 = tmp;
                        j = l;
                    }
                    l--;
                }
                if (j != -1) {
                    dist = std::sqrt(d2);
                    id = indices[j];
                    cp = xs[j];
                } else {
                    dist = d2;
                    id = j;
                    cp = xi;
                }
            });
        timer.tock("initial nearest distance (radius) estimate");

#if 0
        Box bv{zs::vec<float, 3>::constant(-1), zs::vec<float, 3>::constant(1)};
        auto p = zs::vec<float, 3>::zeros();
        p[1] = 1.5;
        fmt::print(fg(fmt::color::red), "distance: {}\n", distance(p, bv));
#endif

        timer.tick();
        using bvh_t = LBvh<3>;
        bvh_t bvh;
        zs::Vector<Box> bvs(vertices.size());
        pol(enumerate(bvs), [&vertices](int i, Box &bv) {
            auto x = zeno::vec_to_other<zs::vec<float, 3>>(vertices[i]);
            bv = Box{x, x};
        });
        // bvh.buildRefit(pol, bvs);
        bvh.build(pol, bvs);
        timer.tock("build bvh");

        timer.tick();
        // keys.resize(pos.size());
        // indices.resize(pos.size());
        std::vector<u32> ks(pos.size());
        std::vector<int> is(pos.size());
        pol(enumerate(ks, is), [&pos, gbv, axis](int i, u32 &key, int &idx) {
            auto p = vec_to_other<zs::vec<float, 3>>(pos[i]);
            for (int d = 0; d != 3; ++d) {
                if (p[d] < gbv._min[d])
                    p[d] = gbv._min[d];
                else if (p[d] > gbv._max[d])
                    p[d] = gbv._max[d];
            }
            auto coord = gbv.getUniformCoord(p).template cast<f32>();
            key = morton_code<3>(coord);
            idx = i;
        });
        merge_sort_pair(pol, std::begin(ks), std::begin(is), pos.size(), std::less<u32>{});
        timer.tock("sort query points");

        timer.tick();
        pol(is, [&vertices, &pos, &dists, &ids, &cps, bvh = proxy<space>(bvh)](int qid) {
            float &dist = dists[qid];
            int &id = ids[qid];
            zeno::vec3f &cp = cps[qid];
            auto p = pos[qid];
            auto x = zeno::vec_to_other<zs::vec<float, 3>>(p);
            auto [j, d] = bvh.find_nearest_point(x, dist * dist, id, wrapv<true>{});
            // dist = (x - zeno::vec_to_other<zs::vec<float, 3>>(vertices[j])).norm();
            dist = d;
            id = j;
            cp = vertices[j];
        });
        timer.tock("query nearest point");

#if 0
        pol(enumerate(pos, dists, ids, cps), [&pos, &vertices, &locs, &xs, &indices, bvh = proxy<space>(bvh), axis](
                                                 int i, const zeno::vec3f &p, float &dist, int &id, zeno::vec3f &cp) {
            auto target = vertices[id];
            if (auto d = zeno::length(p - target); std::abs(d - dist) > limits<float>::epsilon())
                fmt::print("actual dist {}, cp ({}, {}, {}); calced dist {}, cp ({}, {}, {}). \n", d, target[0],
                           target[1], target[2], dist, cp[0], cp[1], cp[2]);
            const int loc = locs[i];
            const auto dist2 = dist * dist;
            {
                auto xi = pos[i];
                auto key = xi[axis];
                int l = loc + 1;
                while (l < xs.size() && zs::sqr(xs[l][axis] - key) < dist2) {
                    if (auto d2 = zeno::lengthSquared(xs[l] - xi); std::sqrt(d2) + limits<float>::epsilon() < dist) {
                        fmt::print("[{}] found nearer pair! real id should be {} ({}), not {} ({})\n", i, indices[l],
                                   std::sqrt(d2), id, std::sqrt(dist2));
                        return;
                    }
                    l++;
                }
                l = loc;
                while (l >= 0 && zs::sqr(xs[l][axis] - key) < dist2) {
                    if (auto d2 = zeno::lengthSquared(xs[l] - xi); std::sqrt(d2) + limits<float>::epsilon() < dist) {
                        fmt::print("[{}] found nearer pair! real id should be {} ({}), not {} ({})\n", i, indices[l],
                                   std::sqrt(d2), id, dist);
                        return;
                    }
                    l--;
                }
            }
        });
#endif
#if 0
        {
            std::vector<int> ids(pos.size());
            std::vector<float> dists(pos.size());
            std::vector<zeno::vec3f> cps(pos.size());
            timer.tick();
            pol(zip(range(pos.size()), locs),
                [&locs, &xs, &vertices, &indices, &pos, &ids, &dists, &cps, axis](int i, const int loc) {
                    float dist2 = limits<float>::max();
                    int id = -1;
                    auto xi = pos[i];
                    auto key = xi[axis];
                    int l = loc + 1;
                    while (l < xs.size() && zs::sqr(std::abs(xs[l][axis] - key)) < dist2) {
                        if (auto d2 = zeno::lengthSquared(xs[l] - xi); d2 < dist2) {
                            dist2 = d2;
                            id = indices[l];
                        }
                        l++;
                    }
                    l = loc;
                    while (l >= 0 && zs::sqr(std::abs(xs[l][axis] - key)) < dist2) {
                        if (auto d2 = zeno::lengthSquared(xs[l] - xi); d2 < dist2) {
                            dist2 = d2;
                            id = indices[l];
                        }
                        l--;
                    }
                    ids[i] = id;
                    dists[i] = std::sqrt(dist2);
                    if (id != -1)
                        cps[i] = vertices[id];
                });
            timer.tock(fmt::format("query nearest points for {} points", pos.size()));
        }
#endif
#endif

        set_output("points", points);
    }
};

ZENDEFNODE(QueryNearestPoints, {/* inputs: */
                                {
                                    {"PrimitiveObject", "points", ""},
                                    {"PrimitiveObject", "target_prim", ""},
                                    {"string", "idTag", "bvh_id"},
                                    {"string", "distTag", "bvh_dist"},
                                    {"string", "closestPointTag", "cp"},
                                },
                                /* outputs: */
                                {{"PrimitiveObject", "points", ""}},
                                /* params: */
                                {},
                                /* category: */
                                {"query"}});

} // namespace zeno