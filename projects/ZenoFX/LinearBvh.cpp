#if WXHFUCKINGHAPPY // remove this when wxl fixed the compile error
#include "LinearBvh.h"
#include "SpatialUtils.hpp"
#include <algorithm>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <zeno/zeno.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zeno {

    typename LBvh::BvFunc
        LBvh::getBvFunc(const std::shared_ptr<PrimitiveObject>& prim) const {
        constexpr auto ma = std::numeric_limits<float>::max();
        constexpr auto mi = std::numeric_limits<float>::lowest();
        const Box defaultBox{ TV{ma, ma, ma}, TV{mi, mi, mi} };

        std::function<Box(typename LBvh::Ti)> getBv;
        if (eleCategory == element_e::tet)
            getBv = [&quads = prim->quads, &refpos = prim->attr<vec3f>("pos"),
            &defaultBox, this](Ti i)->Box {
            auto quad = quads[i];
            Box bv = defaultBox;
            for (int j = 0; j != 4; ++j) {
                const auto& p = refpos[quad[j]];
                for (int d = 0; d != 3; ++d) {
                    if (p[d] - thickness < bv.first[d])
                        bv.first[d] = p[d] - thickness;
                    if (p[d] + thickness > bv.second[d])
                        bv.second[d] = p[d] + thickness;
                }
            }
            return bv;
        };
        else if (eleCategory == element_e::tri)
            getBv = [&tris = prim->tris, &refpos = prim->attr<vec3f>("pos"),
            &defaultBox, this](Ti i)->Box {
            auto tri = tris[i];
            Box bv = defaultBox;
            for (int j = 0; j != 3; ++j) {
                const auto& p = refpos[tri[j]];
                for (int d = 0; d != 3; ++d) {
                    if (p[d] - thickness < bv.first[d])
                        bv.first[d] = p[d] - thickness;
                    if (p[d] + thickness > bv.second[d])
                        bv.second[d] = p[d] + thickness;
                }
            }
            return bv;
        };
        else if (eleCategory == element_e::line)
            getBv = [&lines = prim->lines, &refpos = prim->attr<vec3f>("pos"),
            &defaultBox, this](Ti i)->Box {
            auto line = lines[i];
            Box bv = defaultBox;
            for (int j = 0; j != 2; ++j) {
                const auto& p = refpos[line[j]];
                for (int d = 0; d != 3; ++d) {
                    if (p[d] - thickness < bv.first[d])
                        bv.first[d] = p[d] - thickness;
                    if (p[d] + thickness > bv.second[d])
                        bv.second[d] = p[d] + thickness;
                }
            }
            return bv;
        };
        else if (eleCategory == element_e::point)
            getBv = [&points = prim->points, &refpos = prim->attr<vec3f>("pos"),
            &defaultBox, this](Ti i)->Box {
            auto point = points[i];
            Box bv = defaultBox;
            const auto& p = refpos[point];
            for (int d = 0; d != 3; ++d) {
                if (p[d] - thickness < bv.first[d])
                    bv.first[d] = p[d] - thickness;
                if (p[d] + thickness > bv.second[d])
                    bv.second[d] = p[d] + thickness;
            }
            return bv;
        };
        return getBv;
    }

    template <LBvh::element_e et>
    void LBvh::build(const std::shared_ptr<PrimitiveObject>& prim, float thickness, element_t<et>) {
        this->primPtr = prim;
        this->thickness = thickness;

        Ti numLeaves = 0; // refpos.size();

        {
            // determine element category
            if constexpr (et == element_e::tet) {
                this->eleCategory = element_e::tet;
                numLeaves = prim->quads.size();
            }
            else if constexpr (et == element_e::tri) {
                this->eleCategory = element_e::tri;
                numLeaves = prim->tris.size();
            }
            else if constexpr (et == element_e::line) {
                this->eleCategory = element_e::line;
                numLeaves = prim->lines.size();
            }
            else if constexpr (et == element_e::point) {
                if (prim->points.size() > 0) {
                    this->eleCategory = element_e::point;
                    numLeaves = prim->points.size();
                }
                else {
                    this->eleCategory = element_e::point;
                    numLeaves = prim->verts.size();
                    prim->points.resize(numLeaves);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                    for (Ti i = 0; i < numLeaves; ++i)
                        prim->points[i] = i;
                }
            }
        }

        const auto& refpos = prim->attr<vec3f>("pos");
        const Ti numNodes = numLeaves + numLeaves - 1;
        sortedBvs.resize(numNodes);
        auxIndices.resize(numNodes);
        levels.resize(numNodes);
        parents.resize(numNodes);
        leafIndices.resize(numLeaves);

        this->getBv = getBvFunc(prim);

        if (numLeaves <= 2) { // edge cases where not enough primitives to form a tree
            for (Ti i = 0; i != numLeaves; ++i) {
                sortedBvs[i] = getBv(i);
                leafIndices[i] = i;
                levels[i] = 0;
                auxIndices[i] = i;
                parents[i] = -1;
            }
            return;
        }

        constexpr int dim = 3;
        constexpr auto ma = std::numeric_limits<float>::max();
        constexpr auto mi = std::numeric_limits<float>::lowest();
        Box wholeBox{ TV{ma, ma, ma}, TV{mi, mi, mi} };

        // fmt::print("wholebox before[{}, {}, {}] - [{}, {}, {}]\n", wholeBox.first[0], wholeBox.first[1], wholeBox.first[2], wholeBox.second[0], wholeBox.second[1], wholeBox.second[2]);
        /// whole box
        TV minVec = { ma, ma, ma };
        TV maxVec = { mi, mi, mi };

        // ref: https://www.openmp.org/spec-html/5.0/openmpsu107.html
        for (int d = 0; d != dim; ++d) {
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp parallel for reduction(min: minVec[d])
#endif
#endif
            for (Ti i = 0; i < refpos.size(); ++i) {
                const auto& p = refpos[i];
                if (p[d] < minVec[d])
                    minVec[d] = p[d];
            }
        }
        for (int d = 0; d != dim; ++d) {
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp parallel for reduction(max: maxVec[d])
#endif
#endif
            for (Ti i = 0; i < refpos.size(); ++i) {
                const auto& p = refpos[i];
                if (p[d] > maxVec[d])
                    maxVec[d] = p[d];
            }
        }
        wholeBox.first = minVec;
        wholeBox.second = maxVec;

        // fmt::print("wholebox after[{}, {}, {}] - [{}, {}, {}]\n", wholeBox.first[0], wholeBox.first[1], wholeBox.first[2], wholeBox.second[0], wholeBox.second[1], wholeBox.second[2]);
        // printf("lbvh bounding box: %f, %f, %f - %f, %f, %f\n", wholeBox.first[0],
        // wholeBox.first[1], wholeBox.first[2], wholeBox.second[0],
        // wholeBox.second[1], wholeBox.second[2]);

        std::vector<std::pair<Tu, Ti>> records(numLeaves); // <mc, id>
        /// morton codes
        auto getMortonCode = [](const TV& p) -> Tu {
            auto expand_bits = [](Tu v) -> Tu { // expands lower 10-bits to 30 bits
                v = (v * 0x00010001u) & 0xFF0000FFu;
                v = (v * 0x00000101u) & 0x0F00F00Fu;
                v = (v * 0x00000011u) & 0xC30C30C3u;
                v = (v * 0x00000005u) & 0x49249249u;
                return v;
            };
            return (expand_bits((Tu)(p[0] * 1024.f)) << (Tu)2) |
                (expand_bits((Tu)(p[1] * 1024.f)) << (Tu)1) |
                expand_bits((Tu)(p[2] * 1024.f));
        };
        {
            const auto lengths = wholeBox.second - wholeBox.first;
            auto getUniformCoord = [&wholeBox, &lengths](const TV& p) {
                // https://newbedev.com/constexpr-variable-captured-inside-lambda-loses-its-constexpr-ness
                constexpr int dim = 3;
                auto offsets = p - wholeBox.first;
                for (int d = 0; d != dim; ++d)
                    offsets[d] = std::clamp(offsets[d], (float)0, lengths[d]) / lengths[d];
                return offsets;
            };
            if constexpr (et == element_e::tet) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti i = 0; i < numLeaves; ++i) {
                    auto quad = prim->quads[i];
                    auto uc = getUniformCoord((refpos[quad[0]] + refpos[quad[1]] +
                        refpos[quad[2]] + refpos[quad[3]]) /
                        4);
                    records[i] = std::make_pair(getMortonCode(uc), i);
                }
            }
            else if constexpr (et == element_e::tri) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti i = 0; i < numLeaves; ++i) {
                    auto tri = prim->tris[i];
                    auto uc = getUniformCoord(
                        (refpos[tri[0]] + refpos[tri[1]] + refpos[tri[2]]) / 3);
                    records[i] = std::make_pair(getMortonCode(uc), i);
                }
            }
            else if constexpr (et == element_e::line) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti i = 0; i < numLeaves; ++i) {
                    auto line = prim->lines[i];
                    auto uc = getUniformCoord((refpos[line[0]] + refpos[line[1]]) / 2);
                    records[i] = std::make_pair(getMortonCode(uc), i);
                }
            }
            else if constexpr (et == element_e::point) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti i = 0; i < numLeaves; ++i) {
                    auto pi = prim->points[i];
                    auto uc = getUniformCoord(refpos[pi]);
                    records[i] = std::make_pair(getMortonCode(uc), i);
                }
            }
        }
        std::sort(std::begin(records), std::end(records));

        std::vector<Tu> splits(numLeaves);
        ///
        constexpr auto numTotalBits = sizeof(Tu) * 8;
        auto clz = [](Tu x) -> Tu {
            static_assert(std::is_same_v<Tu, unsigned int>,
                "Tu should be unsigned int");
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
            return __lzcnt((unsigned int)x);
#elif defined(__clang__) || defined(__GNUC__)
            return __builtin_clz((unsigned int)x);
#endif
        };
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (Ti i = 0; i < numLeaves; ++i) {
            if (i != numLeaves - 1)
                splits[i] = numTotalBits - clz(records[i].first ^ records[i + 1].first);
            else
                splits[i] = numTotalBits + 1;
        }
        ///
        std::vector<Box> leafBvs(numLeaves);
        std::vector<Box> trunkBvs(numLeaves - 1);
        std::vector<Ti> leafLca(numLeaves);
        std::vector<Ti> leafDepths(numLeaves);
        std::vector<Ti> trunkR(numLeaves - 1);
        std::vector<Ti> trunkLc(numLeaves - 1);

        std::vector<std::atomic<Tu>> trunkTopoMarks(numLeaves - 1);
        std::vector<std::atomic<Ti>> trunkBuildFlags(numLeaves - 1);
#if 0
        // already zero-initialized during default ctor
        // https://en.cppreference.com/w/cpp/atomic/atomic/atomic
        // courtesy of pyb
#pragma omp parallel for
        for (Ti i = 0; i < numLeaves - 1; ++i) {
            trunkTopoMarks[i] = 0;
            trunkBuildFlags[i] = 0;
        }
#endif

        {
            std::vector<Ti> trunkL(numLeaves - 1);
            std::vector<Ti> trunkRc(numLeaves - 1);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti idx = 0; idx < numLeaves; ++idx) {
                leafBvs[idx] = getBv(records[idx].second);

                leafLca[idx] = -1, leafDepths[idx] = 1;
                Ti l = idx - 1, r = idx; ///< (l, r]
                bool mark{ false };

                if (l >= 0)
                    mark =
                    splits[l] < splits[r]; ///< true when right child, false otherwise

                int cur = mark ? l : r;
                if (mark)
                    trunkRc[cur] = idx, trunkR[cur] = idx,
                    trunkTopoMarks[cur].fetch_or((Tu)0x00000002u);
                else
                    trunkLc[cur] = idx, trunkL[cur] = idx,
                    trunkTopoMarks[cur].fetch_or((Tu)0x00000001u);

                while (trunkBuildFlags[cur].fetch_add(1) == 1) {
                    { // refit
                        int lc = trunkLc[cur], rc = trunkRc[cur];
                        const auto childMask = trunkTopoMarks[cur] & (Tu)3;
                        const auto& leftBox = (childMask & 1) ? leafBvs[lc] : trunkBvs[lc];
                        const auto& rightBox = (childMask & 2) ? leafBvs[rc] : trunkBvs[rc];
                        Box bv{};
                        for (int d = 0; d != dim; ++d) {
                            bv.first[d] = leftBox.first[d] < rightBox.first[d]
                                ? leftBox.first[d]
                                : rightBox.first[d];
                            bv.second[d] = leftBox.second[d] > rightBox.second[d]
                                ? leftBox.second[d]
                                : rightBox.second[d];
                        }
                        trunkBvs[cur] = bv;
                    }
                    trunkTopoMarks[cur] &= 0x00000007;

                    l = trunkL[cur] - 1, r = trunkR[cur];
                    leafLca[l + 1] = cur, leafDepths[l + 1]++;
                    atomic_thread_fence(std::memory_order_acquire);

                    if (l >= 0)
                        mark =
                        splits[l] < splits[r]; ///< true when right child, false otherwise
                    else
                        mark = false;

                    if (l + 1 == 0 && r == numLeaves - 1) {
                        // trunkPar(cur) = -1;
                        trunkTopoMarks[cur] &= 0xFFFFFFFB;
                        break;
                    }

                    int par = mark ? l : r;
                    // trunkPar(cur) = par;
                    if (mark) {
                        trunkRc[par] = cur, trunkR[par] = r;
                        trunkTopoMarks[par].fetch_and(0xFFFFFFFD);
                        trunkTopoMarks[cur] |= 0x00000004;
                    }
                    else {
                        trunkLc[par] = cur, trunkL[par] = l + 1;
                        trunkTopoMarks[par].fetch_and(0xFFFFFFFE);
                        trunkTopoMarks[cur] &= 0xFFFFFFFB;
                    }
                    cur = par;
                }
            }
        }

        std::vector<Ti> leafOffsets(numLeaves + 1);
        leafOffsets[0] = 0;
        for (Ti i = 1; i <= numLeaves; ++i)
            leafOffsets[i] = leafOffsets[i - 1] + leafDepths[i - 1];
        std::vector<Ti> trunkDst(numLeaves - 1);
        /// compute trunk order
        // [levels], [parents], [trunkDst]
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (Ti i = 0; i < numLeaves; ++i) {
            auto offset = leafOffsets[i];
            parents[offset] = -1;
            for (Ti node = leafLca[i], level = leafDepths[i]; --level;
                node = trunkLc[node]) {
                levels[offset] = level;
                parents[offset + 1] = offset;
                trunkDst[node] = offset++;
            }
        }
        // only left-branch-node's parents are set so far
        // levels store the number of node within the left-child-branch from bottom
        // up starting from 0

        /// reorder trunk
        // [sortedBvs], [auxIndices], [parents]
        // auxIndices here is escapeIndex (for trunk nodes)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (Ti i = 0; i < numLeaves - 1; ++i) {
            const auto dst = trunkDst[i];
            const auto& bv = trunkBvs[i];
            // auto l = trunkL[i];
            auto r = trunkR[i];
            sortedBvs[dst] = bv;
            const auto rb = r + 1;
            if (rb < numLeaves) {
                auto lca = leafLca[rb]; // rb must be in left-branch
                auto brother = (lca != -1 ? trunkDst[lca] : leafOffsets[rb]);
                auxIndices[dst] = brother;
                if (parents[dst] == dst - 1)
                    parents[brother] = dst - 1; // setup right-branch brother's parent
            }
            else
                auxIndices[dst] = -1;
        }

        /// reorder leaf
        // [sortedBvs], [auxIndices], [levels], [parents], [leafIndices]
        // auxIndices here is primitiveIndex (for leaf nodes)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (Ti i = 0; i < numLeaves; ++i) {
            const auto& bv = leafBvs[i];
            // const auto leafDepth = leafDepths[i];

            auto dst = leafOffsets[i + 1] - 1;
            leafIndices[i] = dst;
            sortedBvs[dst] = bv;
            auxIndices[dst] = records[i].second;
            levels[dst] = 0;
            if (parents[dst] == dst - 1)
                parents[dst + 1] = dst - 1; // setup right-branch brother's parent
              // if (leafDepth > 1) parents[dst + 1] = dst - 1;  // setup right-branch
              // brother's parent
        }
    }

    template void
        LBvh::build<LBvh::element_e::point>(const std::shared_ptr<PrimitiveObject>&,
            float, element_t<element_e::point>);
    template void
        LBvh::build<LBvh::element_e::line>(const std::shared_ptr<PrimitiveObject>&,
            float, element_t<element_e::line>);
    template void
        LBvh::build<LBvh::element_e::tri>(const std::shared_ptr<PrimitiveObject>&,
            float, element_t<element_e::tri>);
    template void
        LBvh::build<LBvh::element_e::tet>(const std::shared_ptr<PrimitiveObject>&,
            float, element_t<element_e::tet>);

    void LBvh::build(const std::shared_ptr<PrimitiveObject>& prim,
        float thickness) {
        // determine element category
        if (prim->quads.size() > 0)
            build(prim, thickness, element_c<element_e::tet>);
        else if (prim->tris.size() > 0)
            build(prim, thickness, element_c<element_e::tri>);
        else if (prim->lines.size() > 0)
            build(prim, thickness, element_c<element_e::line>);
        else if (prim->points.size() > 0)
            build(prim, thickness, element_c<element_e::point>);
        else
            build(prim, thickness, element_c<element_e::point>);
    }

    void LBvh::refit() {
        std::shared_ptr<const PrimitiveObject> prim = primPtr.lock();
        if (!prim)
            throw std::runtime_error(
                "the primitive object referenced by lbvh not available anymore");
        const auto& refpos = prim->attr<vec3f>("pos");

        const auto numLeaves = getNumLeaves();
        if (numLeaves <= 2) {
            for (Ti i = 0; i != numLeaves; ++i) {
                sortedBvs[i] = getBv(i);
                leafIndices[i] = i;
                levels[i] = 0;
                auxIndices[i] = i;
                parents[i] = -1;
            }
            return;
        }
        const auto numNodes = numLeaves * 2 - 1;
        std::vector<std::atomic<Ti>> refitFlags(numNodes);
#if 0
        // comment out with the same reason in build
#pragma omp parallel for
        for (Ti i = 0; i < numNodes; ++i)
            refitFlags[i] = 0;
#endif

        {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti nid = 0; nid < numLeaves; ++nid) {
                auto idx = leafIndices[nid];
                sortedBvs[idx] = getBv(auxIndices[idx]);

                auto par = parents[idx];
                while (par != -1) {
                    Ti old{ 0 };
                    if (refitFlags[par].compare_exchange_strong(
                        old, (Ti)1, std::memory_order_acquire))
                        break;
                    auto lc = par + 1;
                    auto rc = levels[lc] == 0 ? lc + 1 : auxIndices[lc];
                    // merge box
                    const auto& leftBox = sortedBvs[lc];
                    const auto& rightBox = sortedBvs[rc];
                    Box bv{};
                    for (int d = 0; d != 3; ++d) {
                        bv.first[d] = leftBox.first[d] < rightBox.first[d]
                            ? leftBox.first[d]
                            : rightBox.first[d];
                        bv.second[d] = leftBox.second[d] > rightBox.second[d]
                            ? leftBox.second[d]
                            : rightBox.second[d];
                    }
                    sortedBvs[par] = bv;
                    atomic_thread_fence(std::memory_order_release);
                    par = parents[par];
                }
            }
        }
    }

    /// nearest primitive
    template <LBvh::element_e et>
    void LBvh::find_nearest(TV const& pos, Ti& id, float& dist,
        element_t<et>) const {
        std::shared_ptr<const PrimitiveObject> prim = primPtr.lock();
        if (!prim)
            throw std::runtime_error(
                "the primitive object referenced by lbvh not available anymore");
        const auto& refpos = prim->attr<vec3f>("pos");

        const Ti numNodes = sortedBvs.size();
        Ti node = 0;
        while (node != -1 && node != numNodes) {
            Ti level = levels[node];
            // level and node are always in sync
            for (; level; --level, ++node)
                if (auto d = distance(sortedBvs[node], pos); d > dist)
                    break;
            // leaf node check
            if (level == 0) {
                const auto eid = auxIndices[node];
                float d = std::numeric_limits<float>::max();
                if constexpr (et == element_e::point)
                    d = length(refpos[prim->points[eid]] - pos);
                else if constexpr (et == element_e::line) {
                    auto line = prim->lines[eid];
                    d = dist_pe(pos, refpos[line[0]], refpos[line[1]]);
                }
                else if constexpr (et == element_e::tri) {
                    auto tri = prim->tris[eid];
                    d = dist_pt(pos, refpos[tri[0]], refpos[tri[1]], refpos[tri[2]]);
                }
                else if constexpr (et == element_e::tet) {
                    auto tet = prim->quads[eid];
                    if (auto dd =
                        dist_pt(pos, refpos[tet[0]], refpos[tet[1]], refpos[tet[2]]);
                        dd < d)
                        d = dd;
                    if (auto dd =
                        dist_pt(pos, refpos[tet[1]], refpos[tet[3]], refpos[tet[2]]);
                        dd < d)
                        d = dd;
                    if (auto dd =
                        dist_pt(pos, refpos[tet[0]], refpos[tet[3]], refpos[tet[2]]);
                        dd < d)
                        d = dd;
                    if (auto dd =
                        dist_pt(pos, refpos[tet[0]], refpos[tet[2]], refpos[tet[3]]);
                        dd < d)
                        d = dd;
                }
                if (d < dist) {
                    id = eid;
                    dist = d;
                }
                node++;
            }
            else // separate at internal nodes
                node = auxIndices[node];
        }
    }

    template void LBvh::find_nearest<LBvh::element_e::point>(
        const LBvh::TV&, LBvh::Ti&, float&,
        typename element_t<element_e::point>) const;
    template void LBvh::find_nearest<LBvh::element_e::line>(
        const LBvh::TV&, LBvh::Ti&, float&,
        typename element_t<element_e::line>) const;
    template void LBvh::find_nearest<LBvh::element_e::tri>(
        const LBvh::TV&, LBvh::Ti&, float&,
        typename element_t<element_e::tri>) const;
    template void LBvh::find_nearest<LBvh::element_e::tet>(
        const LBvh::TV&, LBvh::Ti&, float&,
        typename element_t<element_e::tet>) const;

    void LBvh::find_nearest(TV const& pos, Ti& id, float& dist) const {
        if (eleCategory == element_e::tet)
            find_nearest(pos, id, dist, element_c<element_e::tet>);
        else if (eleCategory == element_e::tri)
            find_nearest(pos, id, dist, element_c<element_e::tri>);
        else if (eleCategory == element_e::line)
            find_nearest(pos, id, dist, element_c<element_e::line>);
        else if (eleCategory == element_e::point)
            find_nearest(pos, id, dist, element_c<element_e::point>);
    }

    std::shared_ptr<PrimitiveObject>
        LBvh::retrievePrimitive(Ti eid) const {
        std::shared_ptr<const PrimitiveObject> prim = primPtr.lock();
        if (!prim)
            throw std::runtime_error(
                "the primitive object referenced by lbvh not available anymore");
        const auto& refpos = prim->attr<vec3f>("pos");

        auto ret = std::make_shared<PrimitiveObject>();
        if (eleCategory == element_e::tet) {
            auto quad = prim->quads[eid];
            ret->quads.push_back({ 0, 1, 2, 3 });
            ret->verts.push_back(refpos[quad[0]]);
            ret->verts.push_back(refpos[quad[1]]);
            ret->verts.push_back(refpos[quad[2]]);
            ret->verts.push_back(refpos[quad[3]]);
        }
        else if (eleCategory == element_e::tri) {
            auto tri = prim->tris[eid];
            ret->tris.push_back({ 0, 1, 2 });
            ret->verts.push_back(refpos[tri[0]]);
            ret->verts.push_back(refpos[tri[1]]);
            ret->verts.push_back(refpos[tri[2]]);
        }
        else if (eleCategory == element_e::line) {
            auto line = prim->lines[eid];
            ret->lines.push_back({ 0, 1 });
            ret->verts.push_back(refpos[line[0]]);
            ret->verts.push_back(refpos[line[1]]);
        }
        else if (eleCategory == element_e::point) {
            auto point = prim->points[eid];
            ret->points.push_back(0);
            ret->verts.push_back(refpos[point]);
        }
        return ret;
    }

    vec3f LBvh::retrievePrimitiveCenter(Ti eid) const {
        std::shared_ptr<const PrimitiveObject> prim = primPtr.lock();
        if (!prim)
            throw std::runtime_error(
                "the primitive object referenced by lbvh not available anymore");
        const auto& refpos = prim->attr<vec3f>("pos");

        vec3f ret;
        if (eleCategory == element_e::tet) {
            auto quad = prim->quads[eid];
            ret = (refpos[quad[0]] + refpos[quad[1]] + refpos[quad[2]] +
                refpos[quad[3]]) /
                4;
        }
        else if (eleCategory == element_e::tri) {
            auto tri = prim->tris[eid];
            ret = (refpos[tri[0]] + refpos[tri[1]] + refpos[tri[2]]) / 3;
        }
        else if (eleCategory == element_e::line) {
            auto line = prim->lines[eid];
            ret = (refpos[line[0]] + refpos[line[1]]) / 2;
        }
        else if (eleCategory == element_e::point) {
            auto point = prim->points[eid];
            ret = refpos[point];
        }
        return ret;
    }

} // namespace zeno
#endif
