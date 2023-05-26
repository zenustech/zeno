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
#include <queue>
#include <zeno/ListObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
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

#if 0
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
#else

struct PrimitiveConnectedComponents : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &verts = prim->verts;
        const auto &pos = verts.values;

        const auto &tris = prim->tris.values;
        const bool hasTris = tris.size() > 0;
        // const bool hasTriUV = tris.has_attr<vec3f>("uv0") && tris.has_attr<vec3f>("uv1") && tris.has_attr<vec3f>("uv2");
        // const auto &uvs = prim->uvs;

        const auto &loops = prim->loops;
        const auto &polys = prim->polys;
        const bool hasLoops = polys.size() > 1;

        std::size_t expectedLinks = hasTris ? tris.size() * 3 : (polys.values.back()[0] + polys.values.back()[1]);

        if ((hasTris ^ hasLoops) == 0)
            throw std::runtime_error("The input mesh must either own active triangle topology or loop topology.");

        // for island partition
        std::vector<int> elementMarks(hasTris ? tris.size() : polys.size());
        std::vector<int> elementOffsets(elementMarks.size());

        using IV = zs::vec<int, 2>;
        zs::bht<int, 2, int, 16> tab{expectedLinks};
        std::vector<int> is, js;

        if (hasTris) {
            auto &eles = tris;
            pol(range(eles), [tab = view<execspace_e::openmp>(tab)](const auto &ele) mutable {
                using eleT = RM_CVREF_T(ele);
                for (int i = 0; i < 3; ++i) {
                    auto a = ele[i];
                    auto b = ele[(i + 1) % 3];
                    if (a > b)
                        std::swap(a, b);
                    tab.insert(IV{a, b});
                }
            });
        } else {
            pol(range(polys), [&, tab = view<execspace_e::openmp>(tab)](const auto &poly) mutable {
                auto offset = poly[0];
                auto size = poly[1];
                for (int i = 0; i < size; ++i) {
                    auto a = loops[offset + i];
                    auto b = loops[offset + (i + 1) % size];
                    if (a > b)
                        std::swap(a, b);
                    tab.insert(IV{a, b});
                }
            });
        }

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

        /// @note update fathers of each vertex
        std::vector<int> fas(pos.size());
        union_find(pol, spmat, range(fas));

        /// @note update ancestors, discretize connected components
        zs::bht<int, 1, int, 16> vtab{pos.size()};

        pol(range(pos.size()), [&fas, vtab = view<space>(vtab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            vtab.insert(fa);
        });

        auto numSets = vtab.size();
        fmt::print("{} disjoint sets in total.\n", numSets);

        std::vector<int> invMap(numSets);
        std::vector<std::pair<int, int>> kvs(numSets);
        auto keys = vtab._activeKeys;
        pol(enumerate(keys, kvs), [](int id, int key, std::pair<int, int> &kv) { kv = std::make_pair(key, id); });
        struct {
            constexpr bool operator()(const std::pair<int, int> &a, const std::pair<int, int> &b) const {
                return a.first < b.first;
            }
        } lessOp;
        std::sort(kvs.begin(), kvs.end(), lessOp);
        pol(enumerate(kvs), [&invMap](int no, auto kv) { invMap[kv.second] = no; });

        /// @brief compute the set index of each vertex and calculate the size of each set
        auto &setids = prim->add_attr<int>("set");
        std::vector<int> vertexCounts(numSets), vertexOffsets(numSets);
        pol(range(pos.size()), [&fas, &setids, &vertexCounts, &invMap, vtab = view<space>(vtab)](int vi) mutable {
            auto ancestor = fas[vi];
            auto setNo = vtab.query(ancestor);
            auto dst = invMap[setNo];
            setids[vi] = dst;
            atomic_add(exec_omp, &vertexCounts[dst], 1);
        });
        exclusive_scan(pol, std::begin(vertexCounts), std::end(vertexCounts), std::begin(vertexOffsets));

        auto outPrims = std::make_shared<ListObject>();
        outPrims->arr.resize(numSets);

        std::vector<int> preserveMarks(pos.size()), preserveOffsets(pos.size());
        for (int setNo = 0; setNo != numSets; ++setNo) {
            auto primIsland = std::make_shared<PrimitiveObject>();
            outPrims->arr[setNo] = primIsland;
            /// @brief comptact vertices
            primIsland->resize(vertexCounts[setNo]);
            // add custom vert attributes
            verts.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                primIsland->verts.add_attr<T>(key);
            });
            auto &posI = primIsland->attr<vec3f>("pos");
            // mark
            pol(zip(preserveMarks, setids), [&](int &mark, int setId) { mark = setId == setNo; });
            exclusive_scan(pol, std::begin(preserveMarks), std::end(preserveMarks), std::begin(preserveOffsets));
            // verts
            pol(range(pos.size()), [&](int vi) {
                if (preserveMarks[vi]) {
                    auto dst = preserveOffsets[vi];
                    posI[dst] = pos[vi];
                    // other vertex attributes
                    for (auto &[key, arr] : primIsland->verts.attrs) {
                        auto const &k = key;
                        match(
                            [&k, &verts, vi, dst](auto &arr)
                                -> std::enable_if_t<variant_contains<RM_CVREF_T(arr[0]), AttrAcceptAll>::value> {
                                using T = RM_CVREF_T(arr[0]);
                                const auto &srcArr = verts.attr<T>(k);
                                arr[dst] = srcArr[vi];
                            },
                            [](...) {})(arr);
                    }
                }
            });
            if (hasTris) {
                // tris
                pol(range(tris.size()), [&](int ei) {
                    auto tri = tris[ei];
                    elementMarks[ei] = setids[tri[0]] == setNo && setids[tri[1]] == setNo && setids[tri[2]] == setNo;
                });
                exclusive_scan(pol, std::begin(elementMarks), std::end(elementMarks), std::begin(elementOffsets));
                auto triSize = elementOffsets.back() + elementMarks.back();
                auto &triI = primIsland->tris;
                triI.resize(triSize);
                // add custom tris attributes
                prim->tris.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    primIsland->tris.add_attr<T>(key);
                });
                pol(range(tris.size()), [&](int ei) {
                    if (elementMarks[ei]) {
                        auto dst = elementOffsets[ei];
                        for (int d = 0; d != 3; ++d)
                            triI[dst][d] = preserveOffsets[tris[ei][d]];
                        for (auto &[key, arr] : triI.attrs) {
                            auto const &k = key;
                            match(
                                [&k, &tris = prim->tris, ei, dst](auto &arr)
                                    -> std::enable_if_t<variant_contains<RM_CVREF_T(arr[0]), AttrAcceptAll>::value> {
                                    using T = RM_CVREF_T(arr[0]);
                                    const auto &srcArr = tris.attr<T>(k);
                                    arr[dst] = srcArr[ei];
                                },
                                [](...) {})(arr);
                        }
                    }
                });
            } else {
                // loops
                // select polys
                pol(enumerate(polys), [&](int ei, vec2i poly) {
                    int mark = 1;
                    for (int i = 0; i != poly[1]; ++i)
                        if (setids[loops[poly[0] + i]] != setNo) {
                            mark = 0;
                            break;
                        }
                    elementMarks[ei] = mark;
                });
                exclusive_scan(pol, std::begin(elementMarks), std::end(elementMarks), std::begin(elementOffsets));
                auto &polyI = primIsland->polys;
                auto polySize = elementOffsets.back() + elementMarks.back();
                polyI.resize(polySize);
                prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    polyI.add_attr<T>(key);
                });

                std::vector<int> preservedPolySizes(polySize);
                pol(enumerate(polys), [&](int ei, vec2i poly) {
                    if (elementMarks[ei]) {
                        auto dst = elementOffsets[ei];
                        preservedPolySizes[dst] = poly[1];
                    }
                });
                std::vector<int> preservedPolyOffsets(polySize);
                exclusive_scan(pol, std::begin(preservedPolySizes), std::end(preservedPolySizes),
                               std::begin(preservedPolyOffsets));

                auto &loopI = primIsland->loops;
                auto loopSize = preservedPolyOffsets.back() + preservedPolySizes.back();
                loopI.resize(loopSize);
                loops.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    loopI.add_attr<T>(key);
                });
                primIsland->uvs = prim->uvs; // BEWARE: does not remove redundant uvs here

                // write poly.values
                pol(zip(polyI.values, preservedPolyOffsets, preservedPolySizes), [](vec2i &poly, int offset, int size) {
                    poly[0] = offset;
                    poly[1] = size;
                });
                // write poly.attrs
                pol(range(polys.size()), [&](int ei) {
                    if (elementMarks[ei]) {
                        auto dst = elementOffsets[ei];
                        for (auto &[key, arr] : polyI.attrs) {
                            auto const &k = key;
                            match(
                                [&k, &polys, ei, dst](auto &arr)
                                    -> std::enable_if_t<variant_contains<RM_CVREF_T(arr[0]), AttrAcceptAll>::value> {
                                    using T = RM_CVREF_T(arr[0]);
                                    const auto &srcArr = polys.attr<T>(k);
                                    arr[dst] = srcArr[ei];
                                },
                                [](...) {})(arr);
                        }
                    }
                });

                // write loop.values/attrs
                pol(enumerate(polys), [&](int ei, vec2i poly) {
                    if (elementMarks[ei]) {
                        auto dstPolyNo = elementOffsets[ei];
                        auto dstLoopOffset = preservedPolyOffsets[dstPolyNo];
                        for (int i = 0; i != poly[1]; ++i) {
                            auto point = loops[poly[0] + i];
                            loopI.values[dstLoopOffset + i] = preserveOffsets[point];
                        }
                        for (auto &[key, arr] : loopI.attrs) {
                            auto const &k = key;
                            // may include loops["uvs"] (int)
                            match(
                                [&k, &loops, &poly, ei, dstLoopOffset](auto &arr)
                                    -> std::enable_if_t<variant_contains<RM_CVREF_T(arr[0]), AttrAcceptAll>::value> {
                                    using T = RM_CVREF_T(arr[0]);
                                    const auto &srcArr = loops.attr<T>(k);
                                    for (int i = 0; i != poly[1]; ++i) {
                                        arr[dstLoopOffset + i] = srcArr[poly[0] + i];
                                    }
                                },
                                [](...) {})(arr);
                        }
                    }
                });
            }
        }

        set_output("prim_islands", std::move(outPrims));
    }
};

ZENDEFNODE(PrimitiveConnectedComponents, {
                                             {{"PrimitiveObject", "prim"}},
                                             {
                                                 {"ListObject", "prim_islands"},
                                             },
                                             {},
                                             {"zs_geom"},
                                         });

// assuming point uv and triangle topo
struct PrimitiveMarkIslands : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &pos = prim->attr<zeno::vec3f>("pos");

        bool isTris = prim->tris.size() > 0;
        if (isTris) {
            primPolygonate(prim.get(), true);
        }

        const auto &loops = prim->loops;
        const auto &polys = prim->polys;
        using IV = zs::vec<int, 2>;
        zs::bht<int, 2, int, 16> tab{(std::size_t)(polys.values.back()[0] + polys.values.back()[1])};
        std::vector<int> is, js;
        pol(range(polys), [&, tab = view<space>(tab)](const auto &poly) mutable {
            auto offset = poly[0];
            auto size = poly[1];
            for (int i = 0; i < size; ++i) {
                auto a = loops[offset + i];
                auto b = loops[offset + (i + 1) % size];
                if (a > b)
                    std::swap(a, b);
                tab.insert(IV{a, b});
            }
        });

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

        /// @note update fathers of each vertex
        std::vector<int> fas(pos.size());
        union_find(pol, spmat, range(fas));

        /// @note update ancestors, discretize connected components
        zs::bht<int, 1, int, 16> vtab{pos.size()};

        pol(range(pos.size()), [&fas, vtab = view<space>(vtab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            vtab.insert(fa);
        });
        auto numSets = vtab.size();
        fmt::print("{} islands in total.\n", numSets);

        std::vector<int> invMap(numSets);
        std::vector<std::pair<int, int>> kvs(numSets);
        auto keys = vtab._activeKeys;
        pol(enumerate(keys, kvs), [](int id, int key, std::pair<int, int> &kv) { kv = std::make_pair(key, id); });
        struct {
            constexpr bool operator()(const std::pair<int, int> &a, const std::pair<int, int> &b) const {
                return a.first < b.first;
            }
        } lessOp;
        std::sort(kvs.begin(), kvs.end(), lessOp);
        pol(enumerate(kvs), [&invMap](int no, auto kv) { invMap[kv.second] = no; });

        auto &setids = prim->add_attr<int>(get_input2<std::string>("island_tag"));
        pol(range(pos.size()), [&fas, &setids, &invMap, vtab = view<space>(vtab)](int vi) mutable {
            auto ancestor = fas[vi];
            auto setNo = vtab.query(ancestor);
            setids[vi] = invMap[setNo];
        });

        if (isTris) {
            primTriangulate(prim.get(), true, false);
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveMarkIslands, {
                                     {{"PrimitiveObject", "prim"}, {"string", "island_tag", "island_index"}},
                                     {
                                         {"PrimitiveObject", "prim"},
                                     },
                                     {},
                                     {"zs_geom"},
                                 });
#endif

struct PrimitiveReorder : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        bool orderVerts = get_input2<bool>("order_vertices");
        bool orderTris = get_input2<bool>("order_tris");

        using namespace zs;
        using bv_t = zs::AABBBox<3, zs::f32>;
        using zsvec3 = zs::vec<float, 3>;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &verts = prim->verts;
        const auto &pos = verts.values;

        auto &tris = prim->tris.values;

        /// @note bv
        constexpr auto defaultBv =
            bv_t{zsvec3::constant(zs::limits<zs::f32>::max()), zsvec3::constant(zs::limits<zs::f32>::lowest())};
        bv_t gbv;
        if (orderVerts || orderTris) {

            zs::Vector<bv_t> bv{1};
            bv.setVal(defaultBv);

            zs::Vector<zs::f32> X{pos.size()}, Y{pos.size()}, Z{pos.size()};
            zs::Vector<zs::f32> res{6};
            pol(enumerate(X, Y, Z), [&pos] ZS_LAMBDA(int i, zs::f32 &x, zs::f32 &y, zs::f32 &z) {
                auto xn = pos[i];
                x = xn[0];
                y = xn[1];
                z = xn[2];
            });
            zs::reduce(pol, std::begin(X), std::end(X), std::begin(res), zs::limits<zs::f32>::max(), getmin<zs::f32>{});
            zs::reduce(pol, std::begin(X), std::end(X), std::begin(res) + 3, zs::limits<zs::f32>::lowest(),
                       getmax<zs::f32>{});
            zs::reduce(pol, std::begin(Y), std::end(Y), std::begin(res) + 1, zs::limits<zs::f32>::max(),
                       getmin<zs::f32>{});
            zs::reduce(pol, std::begin(Y), std::end(Y), std::begin(res) + 4, zs::limits<zs::f32>::lowest(),
                       getmax<zs::f32>{});
            zs::reduce(pol, std::begin(Z), std::end(Z), std::begin(res) + 2, zs::limits<zs::f32>::max(),
                       getmin<zs::f32>{});
            zs::reduce(pol, std::begin(Z), std::end(Z), std::begin(res) + 5, zs::limits<zs::f32>::lowest(),
                       getmax<zs::f32>{});
            gbv = bv_t{zsvec3{res[0], res[1], res[2]}, zsvec3{res[3], res[4], res[5]}};
        }
        gbv._min -= limits<float>::epsilon() * 16;
        gbv._max += limits<float>::epsilon() * 16;

        /// @note reorder
        struct Mapping {
            zs::Vector<int> originalToOrdered, orderedToOriginal;
        } vertMapping, triMapping;

        if (orderVerts) {
            /// @brief establish vert proximity topo
            RM_CVREF_T(verts) newVerts;

            auto &dsts = vertMapping.originalToOrdered;
            auto &indices = vertMapping.orderedToOriginal;
            dsts.resize(pos.size());
            indices.resize(pos.size());
            zs::Vector<zs::u32> tempBuffer{pos.size() * 2};
            pol(range(pos.size()),
                [dsts = view<space>(dsts), codes = view<space>(tempBuffer), &pos, bv = gbv] ZS_LAMBDA(int i) mutable {
                    auto coord = bv.getUniformCoord(vec_to_other<zsvec3>(pos[i])).template cast<f32>();
                    codes[i] = (zs::u32)morton_code<3>(coord);
                    dsts[i] = i;
                });
            // radix sort
            radix_sort_pair(pol, std::begin(tempBuffer), dsts.begin(), std::begin(tempBuffer) + pos.size(),
                            indices.begin(), pos.size());
            // compute inverse mapping
            pol(range(pos.size()), [dsts = view<space>(dsts), indices = view<space>(indices)] ZS_LAMBDA(int i) mutable {
                dsts[indices[i]] = i;
            });
            // sort vert data
            // alloc all props
            newVerts.resize(verts.size());
            verts.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                newVerts.add_attr<T>(key);
            });
            // reorder
            pol(range(newVerts.size()), [indices = view<space>(indices), &verts, &pos,
                                         &newVerts] ZS_LAMBDA(int i) mutable {
                auto srcNo = indices[i];
                newVerts.values[i] = pos[srcNo];
                for (auto &[key, arr] : newVerts.attrs) {
                    auto const &k = key;
                    match(
                        [&k, &verts, i, srcNo](
                            auto &arr) -> std::enable_if_t<variant_contains<RM_CVREF_T(arr[0]), AttrAcceptAll>::value> {
                            using T = RM_CVREF_T(arr[0]);
                            const auto &srcArr = verts.attr<T>(k);
                            arr[i] = srcArr[srcNo];
                        },
                        [](...) {})(arr);
                }
            });
            verts = std::move(newVerts);
            // update tri indices
            pol(tris, [&dsts](zeno::vec3i &tri) {
                for (auto &v : tri)
                    v = dsts[v];
            });
        }

        if (orderTris) {
            /// @brief map element indices
            const bool hasTris = tris.size() > 0;

            if (hasTris) {
                RM_CVREF_T(prim->tris) newTris;

                auto &dsts = triMapping.originalToOrdered;
                auto &indices = triMapping.orderedToOriginal;
                dsts.resize(tris.size());
                indices.resize(tris.size());
                zs::Vector<zs::u32> tempBuffer{tris.size() * 2};
                pol(range(tris.size()), [dsts = view<space>(dsts), codes = view<space>(tempBuffer), &pos, &tris,
                                         bv = gbv] ZS_LAMBDA(int i) mutable {
                    auto c = zsvec3::zeros();
                    for (auto v : tris[i])
                        c += vec_to_other<zsvec3>(pos[v]);
                    c /= 3;
                    auto coord = bv.getUniformCoord(c).template cast<f32>();
                    codes[i] = (zs::u32)morton_code<3>(coord);
                    dsts[i] = i;
                });
                // radix sort
                radix_sort_pair(pol, std::begin(tempBuffer), dsts.begin(), std::begin(tempBuffer) + tris.size(),
                                indices.begin(), tris.size());
                // compute inverse mapping
                pol(range(pos.size()), [dsts = view<space>(dsts), indices = view<space>(indices)] ZS_LAMBDA(
                                           int i) mutable { dsts[indices[i]] = i; });
                // sort vert data
                // alloc all props
                newTris.resize(tris.size());
                prim->tris.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    newTris.add_attr<T>(key);
                });
                // reorder
                pol(range(newTris.size()),
                    [indices = view<space>(indices), &tris = prim->tris, &newTris] ZS_LAMBDA(int i) mutable {
                        auto srcNo = indices[i];
                        newTris.values[i] = tris.values[srcNo];
                        for (auto &[key, arr] : newTris.attrs) {
                            auto const &k = key;
                            match(
                                [&k, &tris, i, srcNo](auto &arr)
                                    -> std::enable_if_t<variant_contains<RM_CVREF_T(arr[0]), AttrAcceptAll>::value> {
                                    using T = RM_CVREF_T(arr[0]);
                                    const auto &srcArr = tris.attr<T>(k);
                                    arr[i] = srcArr[srcNo];
                                },
                                [](...) {})(arr);
                        }
                    });
                prim->tris = std::move(newTris);
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveReorder,
           {
               {{"PrimitiveObject", "prim"}, {"bool", "order_vertices", "0"}, {"bool", "order_tris", "1"}},
               {
                   {"PrimitiveObject", "prim"},
               },
               {},
               {"zs_geom"},
           });

struct PrimitiveFuse : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        using namespace zs;
        using zsbvh_t = ZenoLinearBvh;
        using bvh_t = zsbvh_t::lbvh_t;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &verts = prim->verts;
        const auto &pos = verts.values;

        /// @brief establish vert proximity topo
        RM_CVREF_T(prim->verts) newVerts;
        auto dist = get_input2<float>("proximity_theshold");
        std::shared_ptr<zsbvh_t> zsbvh;
        ZenoLinearBvh::element_e et = ZenoLinearBvh::point;
        auto bvs = retrieve_bounding_volumes(pol, pos, dist);
        et = ZenoLinearBvh::point;
        zsbvh = std::make_shared<zsbvh_t>();
        zsbvh->et = et;
        bvh_t &bvh = zsbvh->get();
        bvh.build(pol, bvs);

        // exclusion topo
        std::vector<std::vector<int>> neighbors(pos.size());
        pol(range(pos.size()), [bvh = proxy<space>(bvh), &pos, &neighbors, dist2 = dist * dist](int vi) mutable {
            const auto &p = vec_to_other<zs::vec<float, 3>>(pos[vi]);
            bvh.iter_neighbors(p, [&](int vj) {
                if (vi == vj)
                    return;
                if (auto d2 = lengthSquared(pos[vi] - pos[vj]); d2 < dist2)
                    neighbors[vi].push_back(vj);
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
        std::vector<int> fas(pos.size());
        union_find(pol, spmat, range(fas));

        bht<int, 1, int> vtab{pos.size() * 3 / 2};
        pol(range(pos.size()), [&fas, vtab = proxy<space>(vtab)](int vi) mutable {
            auto fa = fas[vi];
            while (fa != fas[fa])
                fa = fas[fa];
            fas[vi] = fa;
            vtab.insert(fa);
            // if (fa > vi)
            //    printf("should not happen!!! fa: %d, self: %d\n", fa, vi);
        });
        if (vtab._buildSuccess.getVal() == 0)
            throw std::runtime_error("PrimitiveFuse hash failed!!");

        /// @brief preserving vertex islands order
        auto numNewVerts = vtab.size();
        std::vector<int> fwdMap(numNewVerts);
        std::vector<std::pair<int, int>> kvs(numNewVerts);
        auto keys = vtab._activeKeys;
        pol(enumerate(keys, kvs), [](int id, auto key, std::pair<int, int> &kv) { kv = std::make_pair(key[0], id); });
        struct {
            constexpr bool operator()(const std::pair<int, int> &a, const std::pair<int, int> &b) const {
                return a.first < b.first;
            }
        } lessOp;
        std::sort(kvs.begin(), kvs.end(), lessOp);
        pol(enumerate(kvs), [&fwdMap](int no, auto kv) { fwdMap[kv.second] = no; });
        //

        newVerts.resize(numNewVerts);
        auto &newPos = newVerts.attr<vec3f>("pos");
        pol(newPos, [](zeno::vec3f &p) { p = vec3f(0, 0, 0); });
        std::vector<int> cnts(numNewVerts);
        pol(range(pos.size()),
            [&cnts, &fas, &newPos, &pos, &fwdMap, vtab = proxy<space>(vtab), tag = wrapv<space>{}](int vi) mutable {
                auto fa = fas[vi];
                auto dst = fwdMap[vtab.query(fa)];
                fas[vi] = dst;
                atomic_add(tag, &cnts[dst], 1);
                for (int d = 0; d != 3; ++d)
                    atomic_add(tag, &newPos[dst][d], pos[vi][d]);
            });
        pol(zip(newPos, cnts), [](zeno::vec3f &p, int sz) { p /= (float)sz; });

        /// @brief map element indices
        auto &tris = prim->tris.values;
        const bool hasTris = tris.size() > 0;

        auto &loops = prim->loops;
        const auto &polys = prim->polys;
        const bool hasLoops = polys.size() > 1;
        if ((hasTris ^ hasLoops) == 0)
            throw std::runtime_error("The input mesh must either own active triangle topology or loop topology.");

        if (hasTris) {
            auto &eles = prim->tris;
            // add custom tris attributes
            verts.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                eles.add_attr<T>(key + "0");
                eles.add_attr<T>(key + "1");
                eles.add_attr<T>(key + "2");
            });
            pol(enumerate(eles.values), [&fas, &verts, &eles](int ei, auto &tri) mutable {
                for (auto &[key, vertArr] : verts.attrs) {
                    auto const &k = key;
                    match(
                        [&k, &eles, &tri, ei](auto &vertArr)
                            -> std::enable_if_t<variant_contains<RM_CVREF_T(vertArr[0]), AttrAcceptAll>::value> {
                            using T = RM_CVREF_T(vertArr[0]);
                            eles.attr<T>(k + "0")[ei] = vertArr[tri[0]];
                            eles.attr<T>(k + "1")[ei] = vertArr[tri[1]];
                            eles.attr<T>(k + "2")[ei] = vertArr[tri[2]];
                        },
                        [](...) {})(vertArr);
                }
                for (auto &e : tri)
                    e = fas[e];
            });
        } else {
            bool uv_exist = prim->uvs.size() > 0 && loops.has_attr("uvs");
            verts.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if (key != "uv")
                    loops.add_attr<T>(key);
                else if (!uv_exist) {
                    loops.add_attr<int>("uvs");
                    prim->uvs.resize(loops.size());
                }
            });
            pol(range(polys), [&fas, &verts, &loops, &prim, uv_exist](const auto &poly) mutable {
                auto offset = poly[0];
                auto size = poly[1];
                for (int i = 0; i < size; ++i) {
                    auto loopI = offset + i;
                    auto ptNo = loops[loopI];

                    for (auto &[key, vertArr] : verts.attrs) {
                        auto const &k = key;
                        auto &lps = loops;
                        if (k == "uv") {
                            if (!uv_exist) {
                                auto &loopUV = loops.attr<int>("uvs");
                                loopUV[loopI] = loopI;
                                auto &uvs = prim->uvs.values;
                                const auto &srcVertUV = std::get<std::vector<vec3f>>(vertArr);
                                auto vertUV = srcVertUV[ptNo];
                                uvs[loopI] = vec2f(vertUV[0], vertUV[1]);
                            }
                        } else {
                            match(
                                [&k, &lps, loopI, ptNo](auto &vertArr)
                                    -> std::enable_if_t<
                                        variant_contains<RM_CVREF_T(vertArr[0]), AttrAcceptAll>::value> {
                                    using T = RM_CVREF_T(vertArr[0]);
                                    lps.attr<T>(k)[loopI] = vertArr[ptNo];
                                },
                                [](...) {})(vertArr);
                        }
                    }

                    loops[loopI] = fas[ptNo];
                }
            });
        }

        /// @brief update verts
        verts = newVerts;
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveFuse, {
                              {{"PrimitiveObject", "prim"}, {"float", "proximity_theshold", "0.00001"}},
                              {
                                  {"PrimitiveObject", "prim"},
                              },
                              {},
                              {"zs_geom"},
                          });

struct ComputeAverageEdgeLength : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto prim = get_input<PrimitiveObject>("prim");
        const auto &pos = prim->attr<vec3f>("pos");

        std::vector<float> els(0);
        std::vector<float> sum(1), minEl(1), maxEl(1);

        if (prim->polys.size()) {
            const auto &loops = prim->loops;
            const auto &polys = prim->polys;
            els.resize(loops.size());
            pol(range(polys), [&pos, &loops, &els](vec2i poly) {
                for (int i = 0; i != poly[1]; ++i) {
                    auto a = loops[poly[0] + i];
                    auto b = loops[poly[0] + (i + 1) % poly[1]];
                    els[poly[0] + i] = length(pos[a] - pos[b]);
                }
            });
        } else {
            auto compute = [&](auto &topos) {
                constexpr int codim = std::tuple_size_v<typename RM_CVREF_T(topos)::value_type>;
                els.resize(topos.size() * codim);
                pol(enumerate(topos), [&pos, &els, codim_c = wrapv<codim>{}](int ei, auto ele) {
                    constexpr int codim = RM_CVREF_T(codim_c)::value;
                    for (int i = 0; i != codim; ++i) {
                        auto a = ele[i];
                        auto b = ele[(i + 1) % codim];
                        els[ei * codim + i] = length(pos[a] - pos[b]);
                    }
                });
            };
            if (prim->quads.size())
                compute(prim->quads);
            else if (prim->tris.size())
                compute(prim->tris);
            else if (prim->lines.size())
                compute(prim->lines);
        }

#if 0
        CppTimer timer;
        timer.tick();
        sum[0] = 0;
        pol(range(els.size()), [&sum, &els](int ei) { atomic_add(exec_omp, &sum[0], els[ei]); });
        timer.tock("naive atomic");
        sum[0] = 0;
        for (auto el : els)
            sum[0] += el;
        fmt::print("deduced init: {}\n", deduce_identity<std::plus<float>, float>());
        fmt::print("ref sum edge lengths: {}, num edges: {}\n", sum[0], els.size());
        auto sz = els.size();
        timer.tick();
        zs::reduce(pol, std::begin(els), std::end(els), std::begin(sum), 0);
        timer.tock("target");
        fmt::print("sum edge lengths: {}, num edges: {}\n", sum[0], els.size());
#endif
        zs::reduce(pol, std::begin(els), std::end(els), std::begin(sum), 0);
        zs::reduce(pol, std::begin(els), std::end(els), std::begin(minEl), zs::limits<float>::max(),
                   zs::getmin<float>{});
        zs::reduce(pol, std::begin(els), std::end(els), std::begin(maxEl), zs::limits<float>::min(),
                   zs::getmax<float>{});

        set_output("prim", prim);
        set_output("average_edge_length", std::make_shared<NumericObject>(sum[0] / els.size()));
        set_output("minimum_edge_length", std::make_shared<NumericObject>(minEl[0]));
        set_output("maximum_edge_length", std::make_shared<NumericObject>(maxEl[0]));
    }
};

ZENDEFNODE(ComputeAverageEdgeLength, {
                                         {
                                             {"PrimitiveObject", "prim"},
                                         },
                                         {
                                             {"PrimitiveObject", "prim"},
                                             {"NumericObject", "average_edge_length"},
                                             {"NumericObject", "minimum_edge_length"},
                                             {"NumericObject", "maximum_edge_length"},
                                         },
                                         {},
                                         {"zs_query"},
                                     });

struct PrimitiveHasUV : INode {
    void apply() override {

        auto prim = get_input<PrimitiveObject>("prim");

        auto ret = std::make_shared<NumericObject>(0);
        if (prim->verts.has_attr("uv"))
            ret = std::make_shared<NumericObject>(1);
        if (prim->polys.size()) {
            if (prim->loops.has_attr("uvs") && prim->uvs.size() > 0)
                ret = std::make_shared<NumericObject>(1);
        } else {
            if (prim->quads.size()) {
                if (prim->quads.has_attr("uv0") && prim->quads.has_attr("uv1") && prim->quads.has_attr("uv2") &&
                    prim->quads.has_attr("uv3"))
                    ret = std::make_shared<NumericObject>(1);
            } else if (prim->tris.size()) {
                if (prim->tris.has_attr("uv0") && prim->tris.has_attr("uv1") && prim->tris.has_attr("uv2"))
                    ret = std::make_shared<NumericObject>(1);
            } else if (prim->lines.size()) {
                if (prim->lines.has_attr("uv0") && prim->lines.has_attr("uv1"))
                    ret = std::make_shared<NumericObject>(1);
            } else if (prim->points.size()) {
                if (prim->points.has_attr("uv0"))
                    ret = std::make_shared<NumericObject>(1);
            }
        }
        set_output("prim", prim);
        set_output("has_uv", ret);
    }
};
ZENDEFNODE(PrimitiveHasUV, {
                               {
                                   {"PrimitiveObject", "prim"},
                               },
                               {
                                   {"PrimitiveObject", "prim"},
                                   {"NumericObject", "has_uv"},
                               },
                               {},
                               {"zs_query"},
                           });

struct SurfacePointsInterpolation : INode {
    void apply() override {
        using namespace zs;
        auto prim = get_input<PrimitiveObject>("prim");
        /// @note assume weight/index tag presence, attr tag can be constructed on-the-fly
        auto attrTag = get_input2<std::string>("attrTag");
        auto weightTag = get_input2<std::string>("weightTag");
        auto indexTag = get_input2<std::string>("indexTag");

        auto &ws = prim->attr<vec3f>(weightTag);
        auto &triInds = prim->attr<float>(indexTag); // this in accordance with pnbvhw.cpp : QueryNearestPrimitive

        const int *vlocked = nullptr;
        if (has_input("vert_exclusion"))
            if (get_input2<bool>("vert_exclusion"))
                if (prim->has_attr("v_feature")) // generated during remesh
                    vlocked = prim->attr<int>("v_feature").data();

        const auto refPrim = get_input<PrimitiveObject>("ref_prim");
        auto refAttrTag = get_input2<std::string>("refAttrTag");

        const auto &refTris = refPrim->tris.values;
        auto doWork = [&](const auto &srcAttr)
            -> std::enable_if_t<variant_contains<RM_CVREF_T(srcAttr[0]), AttrAcceptAll>::value> {
            using T = RM_CVREF_T(srcAttr[0]);
            auto &arr = prim->add_attr<T>(attrTag);
            auto pol = omp_exec();
            pol(enumerate(arr, triInds, ws),
                [&refTris, &srcAttr, vlocked](int vi, T &attr, int refTriNo, const vec3f &w) {
                    if (vlocked)
                        if (vlocked[vi])
                            return; // do not overwrite these marked vertices
                    auto refTri = refTris[refTriNo];
                    attr = w[0] * srcAttr[refTri[0]] + w[1] * srcAttr[refTri[1]] + w[2] * srcAttr[refTri[2]];
                });
        };
        if (attrTag == "pos") {
            doWork(refPrim->attr<vec3f>(refAttrTag));
        } else {
            // auto &dstAttr = prim->attr(attrTag);
            match(doWork, [](...) {})(refPrim->attr(refAttrTag));
        }

        set_output("prim", prim);
    }
};

ZENDEFNODE(SurfacePointsInterpolation, {
                                           {
                                               {"PrimitiveObject", "prim"},
                                               {"string", "attrTag", "pos"},
                                               {"string", "weightTag"},
                                               {"string", "indexTag"},
                                               {"PrimitiveObject", "ref_prim"},
                                               {"string", "refAttrTag", "pos"},
                                               {"bool", "vert_exclusion", "false"},
                                           },
                                           {
                                               {"PrimitiveObject", "prim"},
                                           },
                                           {},
                                           {"zs_geom"},
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
            const auto &uv = pars->attr<vec2f>("uv");
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
        zs::bht<int, 1, int, 16> vtab{pos.size()};
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
        float uvDist2 = zs::sqr(get_input2<float>("uv_dist"));
        const vec2f *uvPtr = nullptr;

        if (pars->has_attr("uv") && std::sqrt(uvDist2) > zs::limits<float>::epsilon() * 10)
            uvPtr = pars->attr<vec2f>("uv").data();

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
        std::vector<int> clusterSize(pos.size());
        int clusterNo = 0;
        auto &clusterids = pars->add_attr<float>(get_input2<std::string>("segment_tag"));
        for (int color = 1; color <= ncolors; ++color) {
            pol(range(pos.size()), [&](int vi) {
                if (colors[vi] != color)
                    return;
                if (atomic_cas(exec_omp, &maskOut[vi], 0, 1) != 0)
                    return;
                auto no = atomic_add(exec_omp, &clusterNo, 1);
                auto cnt = 1;
                clusterids[vi] = no;
                for (int vj : distNeighbors[vi]) {
                    // if (vi == vj)
                    //    continue;
#if 0
                    if (colors[vj] == color)
                        fmt::print("this cannot be happening! vi [{}] clr {}, while nei vj [{}] clr {}\n", vi,
                                   colors[vi], vj, colors[vj]);
#endif
                    /// use 'cas' in case points at the boundary got inserted into adjacent clusters
                    if (uvPtr)
                        if (lengthSquared(uvPtr[vi] - uvPtr[vj]) > uvDist2)
                            continue;
                    if (atomic_cas(exec_omp, &maskOut[vj], 0, 1) == 0) {
                        clusterids[vj] = no;
                        cnt++;
                    }
                }
                clusterSize[no] = cnt;
            });
        }
        clusterSize.resize(clusterNo);

        /// further redistribute particles for more spatial-evenly distributed clusters
        auto npp = get_input2<int>("post_process_cnt");
        while (npp--) {
            std::vector<vec3f> clusterCenters(clusterNo);
            std::vector<vec2f> clusterUVCenters(clusterNo);

            std::memset(clusterCenters.data(), 0, sizeof(vec3f) * clusterNo);
            if (uvPtr)
                std::memset(clusterUVCenters.data(), 0, sizeof(vec2f) * clusterNo);

            pol(range(pos.size()), [&](int vi) {
                auto cno = clusterids[vi];
                const auto &p = pos[vi];
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_omp, &clusterCenters[cno][d], p[d]);
                if (uvPtr) {
                    auto uv = uvPtr[vi];
                    atomic_add(exec_omp, &clusterUVCenters[cno][0], uv[0]);
                    atomic_add(exec_omp, &clusterUVCenters[cno][1], uv[1]);
                }
            });
            pol(range(clusterNo), [&](int cno) { clusterCenters[cno] = clusterCenters[cno] / clusterSize[cno]; });
            if (uvPtr)
                pol(range(clusterNo),
                    [&](int cno) { clusterUVCenters[cno] = clusterUVCenters[cno] / clusterSize[cno]; });

            if (npp)
                std::memset(clusterSize.data(), 0, sizeof(int) * clusterNo);
            bvs = retrieve_bounding_volumes(pol, clusterCenters, 0);
            bvh_t bvh;
            bvh.build(pol, bvs);
            pol(range(pos.size()), [bvh = proxy<space>(bvh), &pos, &clusterids, &clusterCenters, &clusterUVCenters,
                                    &clusterSize, uvPtr, dist, uvDist2](int vi) mutable {
                const auto &p = pos[vi];
                auto cno = clusterids[vi];
                float curDist = length(p - clusterCenters[cno]);
                bv_t pb{vec_to_other<zs::vec<float, 3>>(p - curDist), vec_to_other<zs::vec<float, 3>>(p + curDist)};
                bvh.iter_neighbors(pb, [&](int oCNo) {
                    if (uvPtr) {
                        if (lengthSquared(uvPtr[vi] - clusterUVCenters[oCNo]) > uvDist2)
                            return;
                    }
                    if (auto d = length(p - clusterCenters[oCNo]); d < curDist /* && d < dist*/) {
                        cno = oCNo;
                        curDist = d;
                    }
                });
                clusterids[vi] = cno;
                atomic_add(exec_omp, &clusterSize[cno], 1);
            });
        }
        fmt::print("{} colors {} clusters.\n", ncolors, clusterNo);

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
                                         {"float", "uv_dist", "0"},
                                         {"string", "segment_tag", "segment_index"},
                                         {"int", "post_process_cnt", "0"},
                                         {"bool", "paint_color", "1"},
                                     },
                                     {{"PrimitiveObject", "pars"}, {"NumericObject", "num_segments"}},
                                     {},
                                     {"zs_geom"},
                                 });

struct CollapseClusters : INode {
    void apply() override {
        auto clusters = get_input<zeno::PrimitiveObject>("clusters");
        auto numClusters = get_input2<int>("num_segments");
        auto clusterTag = get_input2<std::string>("segment_tag");
        auto pars = std::make_shared<PrimitiveObject>();
        pars->resize(numClusters);

        std::vector<int> sizes(numClusters, 0);
        using namespace zs;
        {
            constexpr auto space = execspace_e::openmp;
            auto pol = omp_exec();
            const auto &clusterIds = clusters->verts.attr<float>(clusterTag);

            pol(pars->verts.values, [](auto &v) { v = zeno::vec3f(0, 0, 0); });
            pol(zip(clusters->verts.values, clusterIds),
                [&dstPos = pars->verts.values, &sizes](const auto &p, int clusterId) {
                    auto &dst = dstPos[clusterId];
                    for (int d = 0; d != 3; ++d)
                        atomic_add(exec_omp, &dst[d], p[d]);
                    atomic_add(exec_omp, &sizes[clusterId], 1);
                });

            pol(zip(pars->verts.values, sizes), [](auto &p, int sz) {
                if (sz > 0)
                    p /= (float)sz;
                else
                    printf("there exists cluster with no actual particles.\n");
            });
        }
        set_output("pars", std::move(pars));
    }
};
ZENDEFNODE(CollapseClusters, {
                                 {
                                     {"PrimitiveObject", "clusters"},
                                     {"NumericObject", "num_segments"},
                                     {"string", "segment_tag", "segment_index"},
                                 },
                                 {{"PrimitiveObject", "pars"}},
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
        zs::bht<int, 2, int, 16> tab{lines.size() * 2 + tris.size() * 3 + quads.size() * 4};
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
        zs::bht<int, 2, int, 16> tab{lines.size() * 2 + tris.size() * 3 + quads.size() * 4};
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
        std::string distTag = "dist";
        if (has_input("distTag"))
            distTag = get_input2<std::string>("distTag");
        auto &dists = prim->add_attr<float>(distTag);

        float tol = 5e-6f;
        if (has_input("threshold"))
            tol = get_input2<float>("threshold");

        pol(range(pos.size()), [&, bvh = proxy<space>(targetBvh), sideNo](size_t i) {
            using vec3 = zs::vec<float, 3>;
            auto ro = vec3::from_array(pos[i]);
            auto rd = vec3::from_array(nrm[i]).normalized();
            float dist{-1};

            auto robustProcess = [&](auto &f) {
                auto ro0 = ro;
                auto tan = rd.orthogonal().normalized();
                bvh.ray_intersect(ro, rd, f);
                if (dist < -0.5) {
                    ro = ro0 + tan * tol;
                    bvh.ray_intersect(ro, rd, f);
                }
                if (dist < -0.5) {
                    ro = ro0 - tan * tol;
                    bvh.ray_intersect(ro, rd, f);
                }
                if (dist < -0.5) {
                    tan = tan.cross(rd);
                    ro = ro0 + tan * tol;
                    bvh.ray_intersect(ro, rd, f);
                }
                if (dist < -0.5) {
                    ro = ro0 - tan * tol;
                    bvh.ray_intersect(ro, rd, f);
                }
            };
            if (sideNo == 1) { // farthest
                auto f = [&](int triNo) {
                    auto tri = tris[triNo];
                    auto t0 = vec3::from_array(targetPos[tri[0]]);
                    auto t1 = vec3::from_array(targetPos[tri[1]]);
                    auto t2 = vec3::from_array(targetPos[tri[2]]);
                    if (auto d = ray_tri_intersect(ro, rd, t0, t1, t2); d < limit && d > dist) {
                        dist = d;
                    }
                };
                robustProcess(f);
                // bvh.ray_intersect(ro, rd, f);
            } else if (sideNo == 0) { // closest
                auto f = [&](int triNo) {
                    auto tri = tris[triNo];
                    auto t0 = vec3::from_array(targetPos[tri[0]]);
                    auto t1 = vec3::from_array(targetPos[tri[1]]);
                    auto t2 = vec3::from_array(targetPos[tri[2]]);
                    if (auto d = ray_tri_intersect(ro, rd, t0, t1, t2); d < limit && (d < dist || dist < -0.5)) {
                        dist = d;
                    }
                };
                robustProcess(f);
                // bvh.ray_intersect(ro, rd, f);
            }
            if (dist > -0.5)
                pos[i] = (ro + dist * rd).to_array();
            dists[i] = dist;
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
                                     {"string", "distTag", "dist"},
                                     {"float", "threshold", "5e-6"},
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
#pragma omp declare reduction(minimum:KVPair : omp_out = omp_in < omp_out ? omp_in : omp_out) \
    initializer(omp_priv = KVPair{std::numeric_limits<float>::max(), -1})
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

struct KuhnMunkres {
    using T = float; // weight
    int n;
    // std::vector<std::vector<int>> weight;
    std::function<T(int, int)> weight;
    std::queue<int> q;
    std::vector<T> head_l; // mark for the left node, head_l[i] + head_r[j] >= weight[i][j]
    std::vector<T> head_r; // mark for the right node, the same
    std::vector<T> slack;
    std::vector<int> visit_l; // whether left node is in the tree
    std::vector<int> visit_r; // whether right node is in the tree
    std::vector<int> find_l;  // current match for left node
    std::vector<int> find_r;  // current match for right node
    std::vector<int> previous;

    template <typename Func>
    KuhnMunkres(int n, Func &&f)
        : weight{FWD(f)}, n(n), head_l(n), head_r(n), slack(n), visit_l(n), visit_r(n), find_l(n), find_r(n),
          previous(n) {
        // resize the vectors
    }
    int check(int i) {
        visit_l[i] = 1;
        if (find_l[i] != -1) {
            q.push(find_l[i]);
            return visit_r[find_l[i]] = 1;
        }
        while (i != -1) {
            find_l[i] = previous[i];
            std::swap(i, find_r[find_l[i]]);
        }
        return 0;
    }
    void bfs(int s) {
        // initialize
        for (int i = 0; i < n; ++i) {
            slack[i] = std::numeric_limits<T>::max();
            visit_l[i] = visit_r[i] = 0;
        }
        while (!q.empty())
            q.pop();
        q.push(s);
        visit_r[s] = 1;
        T d = 0;
        while (true) {
            while (!q.empty()) {
                for (int i = 0, j = q.front(); i < n; ++i) {
                    d = head_l[i] + head_r[j] - weight(i, j);
                    if (!visit_l[i] && slack[i] >= d) {
                        previous[i] = j;
                        if (d > 0)
                            slack[i] = d;
                        else if (!check(i))
                            return;
                    }
                }
                q.pop();
            }
            d = std::numeric_limits<T>::max();
            for (int i = 0; i < n; ++i)
                if (!visit_l[i] && d > slack[i])
                    d = slack[i];
            for (int i = 0; i < n; ++i) {
                if (visit_l[i])
                    head_l[i] += d;
                else
                    slack[i] -= d;
                if (visit_r[i])
                    head_r[i] -= d;
            }
            for (int i = 0; i < n; ++i)
                if (!visit_l[i] && slack[i] > 0 && !check(i))
                    return;
        }
    }
    void solve() {
        for (int i = 0; i < n; ++i) {
            previous[i] = find_l[i] = find_r[i] = -1;
            head_r[i] = (T)0;
        }
        for (int i = 0; i < n; ++i) {
            head_l[i] = weight(i, 0);
            for (int j = 1; j < n; ++j)
                head_l[i] = std::max(head_l[i], weight(i, j));
        }
        for (int i = 0; i < n; ++i)
            bfs(i);
    }
};

struct KM {
    using T = float; // weight
    static constexpr T inf = std::numeric_limits<float>::max();

    std::function<T(int, int)> w;

    int n, nl, nr, py, x, y, i, j, p;
    std::vector<int> lk, pre;

    T d, ans;
    std::vector<T> lx, ly, slk;

    std::vector<int> vy;

    template <typename Func>
    KM(int n, Func &&f)
        : w{FWD(f)}, nl(n), nr(n), n(n), lk(n + 1), pre(n + 1), lx(n + 1), ly(n + 1), slk(n + 1), vy(n + 1) {
        for (i = 0; i <= n; i++)
            lk[i] = pre[i] = lx[i] = ly[i] = slk[i] = 0;

        for (i = 1; i <= n; i++)
            for (j = 1; j <= n; j++)
                lx[i] = std::max(lx[i], w(i - 1, j - 1));

        for (i = 1; i <= n; i++) {
            for (j = 1; j <= n; j++)
                slk[j] = inf, vy[j] = 0;
            for (lk[py = 0] = i; lk[py]; py = p) {
                vy[py] = 1;
                d = inf;
                x = lk[py];
                for (y = 1; y <= n; y++)
                    if (!vy[y]) {
                        if (lx[x] + ly[y] - w(x - 1, y - 1) < slk[y])
                            slk[y] = lx[x] + ly[y] - w(x - 1, y - 1), pre[y] = py;
                        if (slk[y] < d)
                            d = slk[y], p = y;
                    }
                for (y = 0; y <= n; y++)
                    if (vy[y])
                        lx[lk[y]] -= d, ly[y] += d;
                    else
                        slk[y] -= d;
            }
            for (; py; py = pre[py])
                lk[py] = lk[pre[py]];
        }
        for (ans = 0, i = 1; i <= n; i++) {
            ans += lx[i] + ly[i];
        }
        printf("%f\n", ans);
#if 0
        for (i = 1; i <= nl; i++)
            printf("%d ", lk[i]);
#endif
    }
};

struct AssociateParticles : INode {
    void apply() override {
        auto srcPrim = get_input2<PrimitiveObject>("srcPrim");
        auto dstPrim = get_input2<PrimitiveObject>("dstPrim");
        auto posTag = get_input2<std::string>("target_pos_tag");
        auto indexTag = get_input2<std::string>("target_index_tag");

        auto &dstPos = srcPrim->add_attr<vec3f>(posTag);
        auto &dstIndices = srcPrim->add_attr<int>(indexTag);

        auto n = srcPrim->size();
        const auto &src = srcPrim->attr<vec3f>("pos");
        const auto &dst = dstPrim->attr<vec3f>("pos");
        KuhnMunkres km{(int)n, [&src, &dst](int i, int j) { return std::numeric_limits<float>::max() - length(src[i] - dst[j]); }};
        km.solve();

        if constexpr (false) {
            KM km{(int)n, [&src, &dst](int i, int j) { return std::numeric_limits<float>::max() - length(src[i] - dst[j]); }};
            // fmt::print(fg(fmt::color::green), "new candidate: {}\n", km.solve());
            (void)km;
        }

        float refSum = 0.f;
        for (int i = 0; i != n; ++i)
            refSum += length(src[i] - dst[i]);
        float curSum = 0.f;
        for (int i = 0; i != n; ++i)
            curSum += length(src[i] - dst[km.find_l[i]]);
        fmt::print(fg(fmt::color::red), "ref: {}, calc: {}\n", refSum, curSum);

        auto pol = zs::omp_exec();
        pol(zs::range(n), [&, &dst = dstPrim->attr<vec3f>("pos")](int i) {
            int id = km.find_l[i];
            // int id = km.previous[i];
            dstIndices[i] = id;
            dstPos[i] = dst[id];
        });
        set_output("srcPrim", std::move(srcPrim));
    }
};
ZENDEFNODE(AssociateParticles, {
                                   {{"PrimitiveObject", "srcPrim"},
                                    {"string", "target_pos_tag", "target_pos"},
                                    {"string", "target_index_tag", "target_index"},
                                    {"PrimitiveObject", "dstPrim"}},
                                   {{"PrimitiveObject", "srcPrim"}},
                                   {},
                                   {"zs_geom"},
                               });

struct AssociateParticlesFast : INode {
    void apply() override {
        auto srcPrim = get_input2<PrimitiveObject>("srcPrim");
        auto dstPrim = get_input2<PrimitiveObject>("dstPrim");
        auto posTag = get_input2<std::string>("target_pos_tag");
        auto indexTag = get_input2<std::string>("target_index_tag");

        auto principal = get_input2<zeno::vec3f>("principal_direction");

        auto &dstPos = srcPrim->add_attr<vec3f>(posTag);
        auto &dstIndices = srcPrim->add_attr<int>(indexTag);

        auto n = srcPrim->size();
        const auto &src = srcPrim->attr<vec3f>("pos");
        const auto &dst = dstPrim->attr<vec3f>("pos");

#if 0
        float refSum = 0.f;
        for (int i = 0; i != n; ++i)
            refSum += length(src[i] - dst[i]);
        float curSum = 0.f;
        // for (int i = 0; i != n; ++i)
        //    curSum += length(src[i] - dst[km.find_l[i]]);
        fmt::print(fg(fmt::color::red), "ref: {}, calc: {}\n", refSum, curSum);
#endif

        auto pol = zs::omp_exec();
        std::vector<float> locs(n);
        std::vector<int> srcSortedIndices(n), dstSortedIndices(n);
        auto sortPrim = [&pol, principal, n](const auto &ps, auto &distances, auto &sortedIndices) {
            pol(zs::enumerate(ps, distances, sortedIndices), [principal](int i, const auto &p, auto &dis, auto &id) {
                dis = dot(p, principal);
                id = i;
            });
            merge_sort_pair(pol, std::begin(distances), std::begin(sortedIndices), n);
        };
        sortPrim(src, locs, srcSortedIndices);
        sortPrim(dst, locs, dstSortedIndices);
        pol(zs::range(n), [&](int i) {
            auto srcId = srcSortedIndices[i];
            auto dstId = dstSortedIndices[i];
            dstIndices[srcId] = dstId;
            dstPos[srcId] = dst[dstId];
        });
        set_output("srcPrim", std::move(srcPrim));
    }
};
ZENDEFNODE(AssociateParticlesFast, {
                                       {{"PrimitiveObject", "srcPrim"},
                                        {"string", "target_pos_tag", "target_pos"},
                                        {"string", "target_index_tag", "target_index"},
                                        {"vec3f", "principal_direction", "1, 0, 0"},
                                        {"PrimitiveObject", "dstPrim"}},
                                       {{"PrimitiveObject", "srcPrim"}},
                                       {},
                                       {"zs_geom"},
                                   });

struct RemovePrimitiveTopo : INode {
    void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        auto removeAttr = [](auto &attrVector) { attrVector.clear(); };
        removeAttr(prim->points);
        removeAttr(prim->lines);
        removeAttr(prim->tris);
        removeAttr(prim->quads);
        removeAttr(prim->loops);
        removeAttr(prim->polys);
        removeAttr(prim->edges);
        removeAttr(prim->uvs);
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(RemovePrimitiveTopo, {
                                    {
                                        {"PrimitiveObject", "prim"},
                                    },
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