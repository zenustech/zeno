#include "Structures.hpp"
#include "zensim/container/Bcht.hpp"
#include "zensim/container/Bht.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/graph/Coloring.hpp"
#include "zensim/graph/ConnectedComponents.hpp"
#include "zensim/math/matrix/SparseMatrixOperations.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/zeno.h>

namespace zeno {

template <typename IV>
static zs::Vector<zs::AABBBox<3, float>> retrieve_bounding_volumes(zs::OmpExecutionPolicy &pol,
                                                                   const std::vector<vec3f> &pos,
                                                                   const std::vector<IV> &eles, float thickness) {
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

static zs::Vector<zs::AABBBox<3, float>> retrieve_bounding_volumes(zs::OmpExecutionPolicy &pol,
                                                                   const std::vector<vec3f> &pos, float thickness) {
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

struct PrimitiveConnectedComponents : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        auto &pos = prim->attr<zeno::vec3f>("pos");
        const auto &lines = prim->lines.values;
        const auto &tris = prim->tris.values;
        const auto &quads = prim->quads.values;

        using IV = zs::vec<int, 2>;
        zs::bcht<IV, int, true, zs::universal_hash<IV>, 16> tab{lines.size() * 2 + tris.size() * 3 + quads.size() * 4};
        std::vector<int> is, js;
        auto buildTopo = [&](const auto &eles) mutable {
            pol(range(eles), [tab = view<execspace_e::openmp>(tab)](const auto &ele) mutable {
                using eleT = RM_CVREF_T(ele);
                constexpr int codim = is_same_v<eleT, zeno::vec2i> ? 2 : (is_same_v<eleT, zeno::vec3i> ? 3 : 4);
                for (int i = 0; i < codim; ++i) {
                    auto a = ele[i];
                    auto b = ele[(i + 1) % codim];
                    if (a > b)
                        std::swap(a, b);
                    tab.insert(IV{a, b});
                    if constexpr (codim == 2)
                        break;
                }
            });
        };
        buildTopo(lines);
        buildTopo(tris);
        buildTopo(quads);

        auto numEntries = tab.size();
        is.resize(numEntries);
        js.resize(numEntries);

        pol(zip(is, js, range(tab._activeKeys)), [](int &i, int &j, const auto &ij) {
            i = ij[0];
            j = ij[1];
        });

        /// @note doublets (wihtout value) to csr matrix
        zs::SparseMatrix<int, true> spmat{(int)pos.size(), (int)pos.size()};
        spmat.build(pol, (int)pos.size(), (int)pos.size(), range(is), range(js), true_c);

#if 0
        fmt::print("@@@@@@@@@@@@@@@@@\n");
        spmat.print();
        fmt::print("@@@@@@@@@@@@@@@@@\n\n");
#endif

        /// @note update fathers of each vertex
        auto &fas = prim->add_attr<int>("fa");
        union_find(pol, spmat, range(fas));

#if 0
        fmt::print("$$$$$$$$$$$$$$$$$\n");
        for (int i = 0; i < pos.size(); ++i) {
            fmt::print("vert [{}]\'s father {}\n", i, fas[i]);
        }
        fmt::print("$$$$$$$$$$$$$$$$$\n\n");
#endif

        /// @note update ancestors, discretize connected components
        zs::bcht<int, int, true, zs::universal_hash<int>, 16> vtab{pos.size()};

#if 0
        fmt::print("????init??table??\n");
        for (int i = 0; i != vtab.size(); ++i)
            fmt::print("[{}]-th set representative: {}\n", i, vtab._activeKeys[i]);
        fmt::print("?????????????????\n\n");
#endif

        pol(range(pos.size()), [&fas, vtab = view<space>(vtab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            vtab.insert(fa);
        });

        auto &setids = prim->add_attr<int>("set");
        pol(range(pos.size()), [&fas, &setids, vtab = view<space>(vtab)](int vi) mutable {
            auto ancestor = fas[vi];
            auto setNo = vtab.query(ancestor);
            setids[vi] = setNo;
        });
        auto numSets = vtab.size();
        fmt::print("{} disjoint sets in total.\n", numSets);

        auto outPrim = std::make_shared<PrimitiveObject>();
        outPrim->resize(pos.size());
        auto id = get_input2<int>("set_index");
        int setSize = 0;
        for (int i = 0; i != pos.size(); ++i)
            if (setids[i] == id)
                outPrim->attr<zeno::vec3f>("pos")[setSize++] = pos[i];
        outPrim->resize(setSize);

#if 0
        fmt::print("?????????????????\n");
        for (int i = 0; i != numSets; ++i)
            fmt::print("[{}]-th set representative: {}\n", i, vtab._activeKeys[i]);
        fmt::print("?????????????????\n\n");

        fmt::print("=================\n");
        for (int i = 0; i < pos.size(); ++i) {
            fmt::print("vert [{}] belong to set {}\n", i, setids[i]);
        }
        fmt::print("=================\n\n");
#endif

        set_output("prim", std::move(outPrim));
    }
};

ZENDEFNODE(PrimitiveConnectedComponents, {
                                             {{"PrimitiveObject", "prim"}, {"int", "set_index", "0"}},
                                             {
                                                 {"PrimitiveObject", "prim"},
                                             },
                                             {},
                                             {"zs_query"},
                                         });

struct ParticleCluster : zeno::INode {
    virtual void apply() override {
        using zsbvh_t = ZenoLinearBvh;
        using bvh_t = zsbvh_t::lbvh_t;
        using bv_t = bvh_t::Box;

        auto pars = get_input<zeno::PrimitiveObject>("pars");
        float dist = get_input2<float>("dist");
        float uvDist = get_input2<float>("uv_dist");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = zs::omp_exec();

        zs::Vector<bv_t> bvs;
        const auto &pos = pars->attr<vec3f>("pos");
        auto &clusterId = pars->add_attr<float>("tag");

        std::shared_ptr<zsbvh_t> zsbvh;
        ZenoLinearBvh::element_e et = ZenoLinearBvh::point;
        bvs = retrieve_bounding_volumes(pol, pars->attr<vec3f>("pos"), 0);
        et = ZenoLinearBvh::point;
        zsbvh = std::make_shared<zsbvh_t>();
        zsbvh->et = et;
        bvh_t &bvh = zsbvh->get();
        bvh.build(pol, bvs);

        std::vector<std::vector<int>> neighbors(pos.size());
        pol(range(pos.size()), [bvh = proxy<space>(bvh), &pos, &neighbors, dist](int vi) mutable {
            const auto &p = pos[vi];
            bv_t pb{vec_to_other<zs::vec<float, 3>>(p - dist), vec_to_other<zs::vec<float, 3>>(p + dist)};
            bvh.iter_neighbors(pb, [&](int vj) { neighbors[vi].push_back(vj); });
        });

        /// @brief uv
        if (pars->has_attr("uv") && uvDist > zs::limits<float>::epsilon() * 10) {
            const auto &uv = pars->attr<vec2i>("uv");
            pol(range(pos.size()), [&neighbors, &uv, uvDist2 = uvDist * uvDist](int vi) mutable {
                int n = neighbors[vi].size();
                int nExcl = 0;
                for (int i = 0; i != n - nExcl;) {
                    auto vj = neighbors[vi][i];
                    if (lengthSquared(uv[vj] - uv[vi]) < uvDist2) {
                        i++;
                    } else {
                        nExcl++;
                        std::swap(neighbors[vi][i], neighbors[vi][n - nExcl]);
                    }
                }
                neighbors[vi].resize(n - nExcl);
            });
        }

        std::vector<int> numNeighbors(pos.size() + 1);
        pol(zip(numNeighbors, neighbors), [](auto &n, const std::vector<int> &neis) { n = neis.size(); });

        SparseMatrix<int, true> spmat(pos.size(), pos.size());
        spmat._ptrs.resize(pos.size() + 1);
        exclusive_scan(pol, std::begin(numNeighbors), std::end(numNeighbors), std::begin(spmat._ptrs));

        auto numEntries = spmat._ptrs[pos.size()];
        spmat._inds.resize(numEntries);

        pol(range(pos.size()),
            [&neighbors, inds = view<space>(spmat._inds), offsets = view<space>(spmat._ptrs)](int vi) {
                auto offset = offsets[vi];
                for (int vj : neighbors[vi])
                    inds[offset++] = vj;
            });
        ///
        std::vector<int> fas(pos.size());
        union_find(pol, spmat, range(fas));
        /// @note update ancestors, discretize connected components
        zs::bcht<int, int, true, zs::universal_hash<int>, 16> vtab{pos.size()};
        pol(range(pos.size()), [&fas, vtab = view<space>(vtab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            vtab.insert(fa);
        });
        auto &clusterids = pars->add_attr<float>(get_input2<std::string>("cluster_tag"));
        pol(range(pos.size()), [&clusterids, &fas, vtab = view<space>(vtab)](int vi) mutable {
            auto ancestor = fas[vi];
            auto clusterNo = vtab.query(ancestor);
            clusterids[vi] = clusterNo;
        });
        auto numClusters = vtab.size();
        fmt::print("{} clusters in total.\n", numClusters);

        set_output("num_clusters", std::make_shared<NumericObject>((int)numClusters));
        set_output("pars", std::move(pars));
    }
};

ZENDEFNODE(ParticleCluster, {
                                {
                                    {"PrimitiveObject", "pars"},
                                    {"float", "dist", "1"},
                                    {"float", "uv_dist", "0"},
                                    {"string", "cluster_tag", "cluster_index"},
                                },
                                {{"PrimitiveObject", "pars"}, {"NumericObject", "num_clusters"}},
                                {},
                                {"zs_geom"},
                            });

struct ParticleSegmentation : zeno::INode {
    virtual void apply() override {
        using zsbvh_t = ZenoLinearBvh;
        using bvh_t = zsbvh_t::lbvh_t;
        using bv_t = bvh_t::Box;

        auto pars = get_input<zeno::PrimitiveObject>("pars");
        float dist = get_input2<float>("dist");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = zs::omp_exec();

        zs::Vector<bv_t> bvs;
        const auto &pos = pars->attr<vec3f>("pos");
        auto &clusterId = pars->add_attr<float>("tag");

        std::shared_ptr<zsbvh_t> zsbvh;
        ZenoLinearBvh::element_e et = ZenoLinearBvh::point;
        bvs = retrieve_bounding_volumes(pol, pars->attr<vec3f>("pos"), 0);
        et = ZenoLinearBvh::point;
        zsbvh = std::make_shared<zsbvh_t>();
        zsbvh->et = et;
        bvh_t &bvh = zsbvh->get();
        bvh.build(pol, bvs);

        // exclusion topo
        std::vector<std::vector<int>> neighbors(pos.size());
        std::vector<std::vector<int>> distNeighbors(pos.size()); // for clustering later
        pol(range(pos.size()), [bvh = proxy<space>(bvh), &pos, &neighbors, &distNeighbors, dist](int vi) mutable {
            const auto &p = pos[vi];
            bv_t pb{vec_to_other<zs::vec<float, 3>>(p - 2 * dist), vec_to_other<zs::vec<float, 3>>(p + 2 * dist)};
            bvh.iter_neighbors(pb, [&](int vj) {
                if (vi == vj)
                    return;
                if (auto d2 = lengthSquared(pos[vi] - pos[vj]); d2 < (dist + dist) * (dist + dist)) {
                    neighbors[vi].push_back(vj);
                    if (d2 < dist * dist)
                        distNeighbors[vi].push_back(vj);
                }
            });
        });

        std::vector<int> numNeighbors(pos.size() + 1);
        pol(zip(numNeighbors, neighbors), [](auto &n, const std::vector<int> &neis) { n = neis.size(); });

        SparseMatrix<int, true> spmat(pos.size(), pos.size());
        spmat._ptrs.resize(pos.size() + 1);
        exclusive_scan(pol, std::begin(numNeighbors), std::end(numNeighbors), std::begin(spmat._ptrs));

        auto numEntries = spmat._ptrs[pos.size()];
        spmat._inds.resize(numEntries);

        pol(range(pos.size()),
            [&neighbors, inds = view<space>(spmat._inds), offsets = view<space>(spmat._ptrs)](int vi) {
                auto offset = offsets[vi];
                for (int vj : neighbors[vi])
                    inds[offset++] = vj;
            });

        /// maximum coloring
        std::vector<u32> weights(pos.size());
        {
            bht<int, 1, int> tab{spmat.get_allocator(), pos.size() * 2};
            tab.reset(pol, true);
            pol(enumerate(weights), [tab1 = proxy<space>(tab)](int seed, u32 &w) mutable {
                using tab_t = RM_CVREF_T(tab);
                std::mt19937 rng;
                rng.seed(seed);
                u32 v = rng() % (u32)4294967291u;
                // prevent weight duplications
                while (tab1.insert(v) != tab_t::sentinel_v)
                    v = rng() % (u32)4294967291u;
                w = v;
            });
        }
        auto &colors = pars->add_attr<float>("colors"); // 0 by default
        auto ncolors = maximum_independent_sets(pol, spmat, weights, colors);

        std::vector<int> maskOut(pos.size());
        int clusterNo = 0;
        auto &clusterids = pars->add_attr<float>(get_input2<std::string>("segment_tag"));
        for (int color = 1; color <= ncolors; ++color) {
            pol(range(pos.size()), [&](int vi) {
                if (colors[vi] != color)
                    return;
                if (atomic_cas(exec_omp, &maskOut[vi], 0, 1) != 0)
                    return;
                auto no = atomic_add(exec_omp, &clusterNo, 1);
                clusterids[vi] = no;
                for (int vj : distNeighbors[vi]) {
                    if (vi == vj)
                        continue;
#if 0
                    if (colors[vj] == color)
                        fmt::print("this cannot be happening! vi [{}] clr {}, while nei vj [{}] clr {}\n", vi,
                                   colors[vi], vj, colors[vj]);
#endif
                    /// use 'cas' in case points at the boundary got inserted into adjacent clusters
                    if (atomic_cas(exec_omp, &maskOut[vj], 0, 1) == 0) {
                        clusterids[vj] = no;
                    }
                }
            });
        }
        fmt::print("{} colors {} clusters.\n", ncolors, clusterNo);

#if 1
        if (get_input2<bool>("paint_color")) {
            auto &clrs = pars->add_attr<vec3f>("clr");
            pol(range(pos.size()), [&](int vi) {
                std::mt19937 rng;
                rng.seed(clusterids[vi]);
                u32 r = rng() % 256u;
                u32 g = rng() % 256u;
                u32 b = rng() % 256u;
                zeno::vec3f clr{1.f * r / 256.f, 1.f * g / 256.f, 1.f * b / 256.f};
                clrs[vi] = clr;
            });
        }
#endif

#if 0
        std::vector<int> nPerCluster(clusterNo);
        int nTotal = 0;
        pol(range(pos.size()), [&](int vi) {
            int clusterNo = (int)clusterids[vi];
            atomic_add(exec_omp, &nPerCluster[clusterNo], 1);
            atomic_add(exec_omp, &nTotal, 1);
        });
        if (nTotal != pos.size())
            throw std::runtime_error("some particles might be duplicated!");
#endif

        set_output("num_segments", std::make_shared<NumericObject>((int)clusterNo));
        set_output("pars", std::move(pars));
    }
};

ZENDEFNODE(ParticleSegmentation, {
                                     {
                                         {"PrimitiveObject", "pars"},
                                         {"float", "dist", "1"},
                                         {"string", "segment_tag", "segment_index"},
                                         {"bool", "paint_color", "1"},
                                     },
                                     {{"PrimitiveObject", "pars"}, {"NumericObject", "num_segments"}},
                                     {},
                                     {"zs_geom"},
                                 });

struct PrimitiveBFS : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        auto &pos = prim->attr<zeno::vec3f>("pos");
        const auto &lines = prim->lines.values;
        const auto &tris = prim->tris.values;
        const auto &quads = prim->quads.values;

        using IV = zs::vec<int, 2>;
        zs::bcht<IV, int, true, zs::universal_hash<IV>, 16> tab{lines.size() * 2 + tris.size() * 3 + quads.size() * 4};
        std::vector<int> is, js;
        auto buildTopo = [&](const auto &eles) mutable {
            pol(range(eles), [tab = view<execspace_e::openmp>(tab)](const auto &ele) mutable {
                using eleT = RM_CVREF_T(ele);
                constexpr int codim = is_same_v<eleT, zeno::vec2i> ? 2 : (is_same_v<eleT, zeno::vec3i> ? 3 : 4);
                for (int i = 0; i < codim; ++i) {
                    auto a = ele[i];
                    auto b = ele[(i + 1) % codim];
                    if (a > b)
                        std::swap(a, b);
                    tab.insert(IV{a, b});
                    if constexpr (codim == 2)
                        break;
                }
            });
        };
        buildTopo(lines);
        buildTopo(tris);
        buildTopo(quads);

        auto numEntries = tab.size();
        is.resize(numEntries);
        js.resize(numEntries);

        pol(zip(is, js, range(tab._activeKeys)), [](int &i, int &j, const auto &ij) {
            i = ij[0];
            j = ij[1];
        });

        /// @note doublets (wihtout value) to csr matrix
        zs::SparseMatrix<int, true> spmat{(int)pos.size(), (int)pos.size()};
        spmat.build(pol, (int)pos.size(), (int)pos.size(), range(is), range(js), true_c);
        spmat._vals.resize(spmat.nnz());
        /// finalize connectivity graph
        pol(spmat._vals, [](int &v) { v = 1; });
        puts("done connectivity graph build");

        auto id = get_input2<int>("vert_index");
        if (id >= pos.size())
            id = 0;

        std::vector<int> maskOut(pos.size());
        std::vector<int> levelQueue(pos.size());
        std::vector<int> nextLevel(pos.size());
        auto &bfslevel = prim->add_attr<int>("bfs_level");
        // initial mask
        maskOut[id] = 1;
        levelQueue[id] = 1;
        pol(bfslevel, [](int &l) { l = -1; });
        bfslevel[id] = 0;

        puts("done init");
        // bfs
        int iter = 0;
        auto allZerosUpdateQueue = [&nextLevel, &levelQueue, &maskOut, &bfslevel, &pol](int iter) -> bool {
            bool pred = true;
            pol(zip(nextLevel, levelQueue, maskOut, bfslevel), [&](int &v, int &vnext, int &mask, int &level) {
                if (v == 1) {
                    pred = false;
                    vnext = 1;
                    mask = 1;
                    level = iter;
                }
            });
            return pred;
        };
        for (iter++;; ++iter) {
            spmv_mask(pol, spmat, levelQueue, maskOut, nextLevel, wrapv<semiring_e::boolean>{});
            if (allZerosUpdateQueue(iter))
                break;
        }
        fmt::print("{} bfs levels in total.\n", iter);

        auto lid = get_input2<int>("level_index");
        auto outPrim = std::make_shared<PrimitiveObject>();
        outPrim->resize(pos.size());
        int setSize = 0;
        for (int i = 0; i != pos.size(); ++i)
            if (bfslevel[i] == lid)
                outPrim->attr<zeno::vec3f>("pos")[setSize++] = pos[i];
        outPrim->resize(setSize);

        set_output("prim", std::move(outPrim));
    }
};

ZENDEFNODE(PrimitiveBFS, {
                             {{"PrimitiveObject", "prim"}, {"int", "vert_index", "0"}, {"int", "level_index", "0"}},
                             {
                                 {"PrimitiveObject", "prim"},
                             },
                             {},
                             {"zs_query"},
                         });

struct PrimitiveColoring : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        auto &pos = prim->attr<zeno::vec3f>("pos");
        const auto &lines = prim->lines.values;
        const auto &tris = prim->tris.values;
        const auto &quads = prim->quads.values;

        using IV = zs::vec<int, 2>;
        zs::bcht<IV, int, true, zs::universal_hash<IV>, 16> tab{lines.size() * 2 + tris.size() * 3 + quads.size() * 4};
        std::vector<int> is, js;
        auto buildTopo = [&](const auto &eles) mutable {
            pol(range(eles), [tab = view<execspace_e::openmp>(tab)](const auto &ele) mutable {
                using eleT = RM_CVREF_T(ele);
                constexpr int codim = is_same_v<eleT, zeno::vec2i> ? 2 : (is_same_v<eleT, zeno::vec3i> ? 3 : 4);
                for (int i = 0; i < codim; ++i) {
                    auto a = ele[i];
                    auto b = ele[(i + 1) % codim];
                    if (a > b)
                        std::swap(a, b);
                    tab.insert(IV{a, b});
                    if constexpr (codim == 2)
                        break;
                }
            });
        };
        buildTopo(lines);
        buildTopo(tris);
        buildTopo(quads);

        auto numEntries = tab.size();
        is.resize(numEntries);
        js.resize(numEntries);

        pol(zip(is, js, range(tab._activeKeys)), [](int &i, int &j, const auto &ij) {
            i = ij[0];
            j = ij[1];
        });
        {
            // test if self connectivity matters
            auto offset = is.size();
            is.resize(offset + pos.size());
            js.resize(offset + pos.size());
            pol(range(pos.size()), [&is, &js, offset](int i) {
                is[offset + i] = i;
                js[offset + i] = i;
            });
        }

        /// @note doublets (wihtout value) to csr matrix
        zs::SparseMatrix<u32, true> spmat{(int)pos.size(), (int)pos.size()};
        spmat.build(pol, (int)pos.size(), (int)pos.size(), range(is), range(js), true_c);
        spmat._vals.resize(spmat.nnz());
        /// finalize connectivity graph
        pol(spmat._vals, [](u32 &v) { v = 1; });
        puts("done connectivity graph build");

        auto &colors = prim->add_attr<float>("colors"); // 0 by default
        // init weights
        std::vector<u32> minWeights(pos.size());
        std::vector<u32> weights(pos.size());
        {
            bht<int, 1, int> tab{spmat.get_allocator(), pos.size() * 2};
            tab.reset(pol, true);
            pol(enumerate(weights), [tab1 = proxy<space>(tab)](int seed, u32 &w) mutable {
                using tab_t = RM_CVREF_T(tab);
                std::mt19937 rng;
                rng.seed(seed);
                u32 v = rng() % (u32)4294967291u;
                // prevent weight duplications
                while (tab1.insert(v) != tab_t::sentinel_v)
                    v = rng() % (u32)4294967291u;
                w = v;
            });
        }
        std::vector<int> maskOut(pos.size());

        puts("done init");
#if 1
        auto iter = fast_independent_sets(pol, spmat, weights, colors);
        {
            fmt::print("{} colors by fast.\n", iter);
            using T = typename RM_CVREF_T(colors)::value_type;
            zs::Vector<int> correct{spmat.get_allocator(), 1};
            correct.setVal(1);
            pol(range(spmat.outerSize()),
                [&colors, spmat = proxy<space>(spmat), correct = proxy<space>(correct)](int i) mutable {
                    auto color = colors[i];
                    if (color == limits<T>::max()) {
                        correct[0] = 0;
                        printf("node [%d]: %f. not colored!\n", i, (float)color);
                        return;
                    }
                    auto &ap = spmat._ptrs;
                    auto &aj = spmat._inds;
                    for (int k = ap[i]; k < ap[i + 1]; k++) {
                        int j = aj[k];
                        if (j == i)
                            continue;
                        if (colors[j] == color) {
                            correct[0] = 0;
                            printf("node [%d]: %f, neighbor node [%d]: %f, conflict!\n", i, (float)color, j,
                                   (float)colors[j]);
                            return;
                        }
                    }
                });
            if (correct.getVal() == 0) {
                throw std::runtime_error("coloring is wrong!\n");
            }
        }
        auto iterRef = maximum_independent_sets(pol, spmat, weights, colors);
        {
            fmt::print("{} colors by maximum\n", iterRef);
            using T = typename RM_CVREF_T(colors)::value_type;
            zs::Vector<int> correct{spmat.get_allocator(), 1};
            correct.setVal(1);
            pol(range(spmat.outerSize()),
                [&colors, spmat = proxy<space>(spmat), correct = proxy<space>(correct)](int i) mutable {
                    auto color = colors[i];
                    if (color == limits<T>::max()) {
                        correct[0] = 0;
                        printf("node [%d]: %f. not colored!\n", i, (float)color);
                        return;
                    }
                    auto &ap = spmat._ptrs;
                    auto &aj = spmat._inds;
                    for (int k = ap[i]; k < ap[i + 1]; k++) {
                        int j = aj[k];
                        if (j == i)
                            continue;
                        if (colors[j] == color) {
                            correct[0] = 0;
                            printf("node [%d]: %f, neighbor node [%d]: %f, conflict!\n", i, (float)color, j,
                                   (float)colors[j]);
                            return;
                        }
                    }
                });
            if (correct.getVal() == 0) {
                throw std::runtime_error("coloring is wrong!\n");
            }
        }
        puts("all right!");
#else
        // bfs
        int iter = 0;
        auto update = [&](int iter) -> bool {
            bool done = true;
            pol(zip(weights, minWeights, maskOut, colors), [&](u32 &w, u32 &mw, int &mask, float &color) {
                //if (w < mw && mask == 0)
                if (w < mw && mw != limits<u32>::max()) {
                    done = false;
                    mask = 1;
                    color = iter;
                    w = limits<u32>::max();
                }
            });
            return done;
        };
        for (iter++;; ++iter) {
            // fmt::print("iterating {}-th time\n", iter);
            spmv_mask(pol, spmat, weights, maskOut, minWeights, wrapv<semiring_e::min_times>{});
            if (update(iter))
                break;
        }
#endif
        fmt::print("{} colors in total.\n", iter);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveColoring, {
                                  {{"PrimitiveObject", "prim"}},
                                  {
                                      {"PrimitiveObject", "prim"},
                                  },
                                  {},
                                  {"zs_query"},
                              });

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
                                 {"zs_query"},
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
            std::vector<KVPair> out(1);
            reduce(pol, std::begin(kvs), std::end(kvs), std::begin(out), KVPair{std::numeric_limits<float>::max(), -1},
                   [](const auto &a, const auto &b) {
                       if (a < b)
                           return a;
                       else
                           return b;
                   });
#if 0
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
#endif
            mi = out[0];

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
                                      {"zs_query"},
                                  });
#endif

struct FollowUpReferencePrimitive : INode {
    void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        auto idTag = get_input2<std::string>("idTag");
        auto wsTag = get_input2<std::string>("weightTag");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        auto &ids = prim->attr<float>(idTag);
        auto &ws = prim->attr<zeno::vec3f>(wsTag);
        auto refPrim = get_input2<PrimitiveObject>("ref_surf_prim");
        auto &refPos = refPrim->attr<vec3f>("pos");
        auto &refTris = refPrim->tris.values;
        auto pol = zs::omp_exec();
        pol(zs::range(prim->size()), [&](int i) {
            int triNo = ids[i];
            auto w = ws[i];
            auto tri = refTris[triNo];
            pos[i] = w[0] * refPos[tri[0]] + w[1] * refPos[tri[1]] + w[2] * refPos[tri[2]];
        });
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(FollowUpReferencePrimitive, {
                                           {{"PrimitiveObject", "prim"},
                                            {"PrimitiveObject", "ref_surf_prim"},
                                            {"string", "idTag", "bvh_id"},
                                            {"string", "weightTag", "bvh_ws"}},
                                           {{"PrimitiveObject", "prim"}},
                                           {},
                                           {"zs_geom"},
                                       });

struct EmbedPrimitiveBvh : zeno::INode {
    virtual void apply() override {
        using zsbvh_t = ZenoLinearBvh;
        using bvh_t = zsbvh_t::lbvh_t;
        using bv_t = bvh_t::Box;

        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &userData = prim->userData();
        float thickness = has_input("thickness") ? get_input<zeno::NumericObject>("thickness")->get<float>() : 0.f;
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
        if (!userData.has(bvhTag)) { // build
            zsbvh = std::make_shared<zsbvh_t>();
            zsbvh->et = et;
            bvh_t &bvh = zsbvh->get();
            bvh.build(pol, bvs);
            userData.set(bvhTag, zsbvh);
        } else { // refit
            zsbvh = std::dynamic_pointer_cast<zsbvh_t>(userData.get(bvhTag));
            zsbvh->et = et;
            bvh_t &bvh = zsbvh->get();
            bvh.refit(pol, bvs);
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(EmbedPrimitiveBvh, {
                                  {{"PrimitiveObject", "prim"},
                                   {"float", "thickness", "0"},
                                   {"enum point line tri quad", "prim_type", "auto"},
                                   {"string", "bvh_tag", "bvh"}},
                                  {{"PrimitiveObject", "prim"}},
                                  {},
                                  {"zs_accel"},
                              });

struct EmbedPrimitiveSpatialHash : zeno::INode {
    virtual void apply() override {
        using zssh_t = ZenoSpatialHash;
        using sh_t = zssh_t::sh_t;
        using bv_t = sh_t::bv_t;

        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &userData = prim->userData();
        auto sideLength = get_input2<float>("side_length");
        float thickness = has_input("thickness") ? get_input<zeno::NumericObject>("thickness")->get<float>() : 0.f;
        auto primType = get_input2<std::string>("prim_type");
        auto shTag = get_input2<std::string>("spatial_hash_tag");

        auto pol = zs::omp_exec();

        zs::Vector<bv_t> bvs;
        std::shared_ptr<zssh_t> zssh;
        ZenoSpatialHash::element_e et = ZenoSpatialHash::point;
        if (primType == "point") {
            bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), thickness);
            et = ZenoSpatialHash::point;
        } else if (primType == "line") {
            bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), prim->lines.values, thickness);
            et = ZenoSpatialHash::curve;
        } else if (primType == "tri") {
            bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), prim->tris.values, thickness);
            et = ZenoSpatialHash::surface;
        } else if (primType == "quad") {
            bvs = retrieve_bounding_volumes(pol, prim->attr<vec3f>("pos"), prim->quads.values, thickness);
            et = ZenoSpatialHash::tet;
        }
        if (!userData.has(shTag)) { // build
            zssh = std::make_shared<zssh_t>();
            zssh->et = et;
            sh_t &sh = zssh->get();
            sh.build(pol, sideLength, bvs);
            userData.set(shTag, zssh);
        } else { // refit
            zssh = std::dynamic_pointer_cast<zssh_t>(userData.get(shTag));
            zssh->et = et;
            sh_t &sh = zssh->get();
            sh.build(pol, sideLength, bvs);
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(EmbedPrimitiveSpatialHash, {
                                          {{"PrimitiveObject", "prim"},
                                           {"float", "side_length", "1"},
                                           {"float", "thickness", "0"},
                                           {"enum point line tri quad", "prim_type", "auto"},
                                           {"string", "spatial_hash_tag", "sh"}},
                                          {{"PrimitiveObject", "prim"}},
                                          {},
                                          {"zs_accel"},
                                      });

} // namespace zeno