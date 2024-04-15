#include <functional>
#include <limits>
#include <unordered_map>
#include <zeno/para/parallel_for.h> // enable by -DZENO_PARALLEL_STL:BOOL=ON
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/core/INode.h>
#include <zeno/zeno.h>
#if defined(_OPENMP)
#define WXL 1
#else
#define WXL 0
#endif
#if WXL
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <atomic>
#endif

namespace zeno {
namespace {

template <class Cond>
static float tri_intersect(Cond cond, vec3f const &ro, vec3f const &rd, vec3f const &v0, vec3f const &v1,
                           vec3f const &v2) {
    const float eps = 1e-6f;
    vec3f u = v1 - v0;
    vec3f v = v2 - v0;
    vec3f n = cross(u, v);
    float b = dot(n, rd);
    if (std::abs(b) > eps) {
        float a = dot(n, v0 - ro);
        float r = a / b;
        if (cond(r)) {
            vec3f ip = ro + r * rd;
            float uu = dot(u, u);
            float uv = dot(u, v);
            float vv = dot(v, v);
            vec3f w = ip - v0;
            float wu = dot(w, u);
            float wv = dot(w, v);
            float d = uv * uv - uu * vv;
            float s = uv * wv - vv * wu;
            float t = uv * wu - uu * wv;
            d = 1.0f / d;
            s *= d;
            t *= d;
            if (-eps <= s && s <= 1 + eps && -eps <= t && s + t <= 1 + eps * 2)
                return r;
        }
    }
    return std::numeric_limits<float>::infinity();
}

/// ref: An Efficient and Robust Ray-Box Intersection Algorithm, 2005
static bool ray_box_intersect(vec3f const &ro, vec3f const &rd, std::pair<vec3f, vec3f> const &box) {
    vec3f invd{1 / rd[0], 1 / rd[1], 1 / rd[2]};
    int sign[3] = {invd[0] < 0, invd[1] < 0, invd[2] < 0};
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = ((sign[0] ? box.second : box.first)[0] - ro[0]) * invd[0];
    tmax = ((sign[0] ? box.first : box.second)[0] - ro[0]) * invd[0];
    tymin = ((sign[1] ? box.second : box.first)[1] - ro[1]) * invd[1];
    tymax = ((sign[1] ? box.first : box.second)[1] - ro[1]) * invd[1];
    if (tmin > tymax || tymin > tmax)
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    tzmin = ((sign[2] ? box.second : box.first)[2] - ro[2]) * invd[2];
    tzmax = ((sign[2] ? box.first : box.second)[2] - ro[2]) * invd[2];
    if (tmin > tzmax || tzmin > tmax)
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    return tmax >= 0.f;
}

struct BVH { // TODO: WXL please complete this to accel up
    PrimitiveObject const *prim{};
#if WXL
    using TV = vec3f;
    using Box = std::pair<TV, TV>;
    using Ti = int;
    static constexpr Ti threshold = 128;
    using Tu = std::make_unsigned_t<Ti>;
    std::vector<Box> sortedBvs;
    std::vector<Ti> auxIndices, levels, parents, leafIndices;
#endif

    void build(PrimitiveObject const *prim) {
        this->prim = prim;
#if WXL
        const auto &verts = prim->verts;
        const auto &tris = prim->tris;
        if (tris.size() >= threshold) {
            const Ti numLeaves = tris.size();
            const Ti numTrunk = numLeaves - 1;
            const Ti numNodes = numLeaves + numTrunk;
            /// utilities
            auto getbv = [&verts, &tris](int tid) -> Box {
                auto ind = tris[tid];
                Box bv = std::make_pair(verts[ind[0]], verts[ind[0]]);
                for (int i = 1; i != 3; ++i) {
                    const auto &v = verts[ind[i]];
                    for (int d = 0; d != 3; ++d) {
                        if (v[d] < bv.first[d])
                            bv.first[d] = v[d];
                        if (v[d] > bv.second[d])
                            bv.second[d] = v[d];
                    }
                }
                return bv;
            };
            auto getMortonCode = [](const TV &p) -> Tu {
                auto expand_bits = [](Tu v) -> Tu { // expands lower 10-bits to 30 bits
                    v = (v * 0x00010001u) & 0xFF0000FFu;
                    v = (v * 0x00000101u) & 0x0F00F00Fu;
                    v = (v * 0x00000011u) & 0xC30C30C3u;
                    v = (v * 0x00000005u) & 0x49249249u;
                    return v;
                };
                return (expand_bits((Tu)(p[0] * 1024.f)) << (Tu)2) | (expand_bits((Tu)(p[1] * 1024.f)) << (Tu)1) |
                       expand_bits((Tu)(p[2] * 1024.f));
            };
            auto clz = [](Tu x) -> Tu {
                static_assert(std::is_same_v<Tu, unsigned int>, "Tu should be unsigned int");
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
                return __lzcnt((unsigned int)x);
#elif defined(__clang__) || defined(__GNUC__)
                return __builtin_clz((unsigned int)x);
#endif
            };

            /// total box
            constexpr auto ma = std::numeric_limits<float>::max();
            constexpr auto mi = std::numeric_limits<float>::lowest();
            Box wholeBox{TV{ma, ma, ma}, TV{mi, mi, mi}};
            TV minVec = {ma, ma, ma};
            TV maxVec = {mi, mi, mi};
            for (int d = 0; d != 3; ++d) {
                float &v = minVec[d];
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp parallel for reduction(min : v)
#endif
#endif
                for (Ti i = 0; i < verts.size(); ++i) {
                    const auto &p = verts[i];
                    if (p[d] < v)
                        v = p[d];
                }
            }
            for (int d = 0; d != 3; ++d) {
                float &v = maxVec[d];
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp parallel for reduction(max : v)
#endif
#endif
                for (Ti i = 0; i < verts.size(); ++i) {
                    const auto &p = verts[i];
                    if (p[d] > v)
                        v = p[d];
                }
            }
            wholeBox.first = minVec;
            wholeBox.second = maxVec;

            /// morton codes
            std::vector<std::pair<Tu, Ti>> records(numLeaves); // <mc, id>
            {
                const auto lengths = wholeBox.second - wholeBox.first;
                auto getUniformCoord = [&wholeBox, &lengths](const TV &p) {
                    auto offsets = p - wholeBox.first;
                    for (int d = 0; d != 3; ++d)
                        offsets[d] = std::clamp(offsets[d], (float)0, lengths[d]) / lengths[d];
                    return offsets;
                };
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti i = 0; i < numLeaves; ++i) {
                    auto tri = tris[i];
                    auto uc = getUniformCoord((verts[tri[0]] + verts[tri[1]] + verts[tri[2]]) / 3);
                    records[i] = std::make_pair(getMortonCode(uc), i);
                }
            }
            std::sort(std::begin(records), std::end(records));

            /// precomputations
            std::vector<Tu> splits(numLeaves);
            constexpr auto numTotalBits = sizeof(Tu) * 8;
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

            std::vector<std::atomic<Ti>> trunkBuildFlags(numLeaves - 1); // already zero-initialized
            {
                std::vector<Ti> trunkL(numLeaves - 1);
                std::vector<Ti> trunkRc(numLeaves - 1);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti idx = 0; idx < numLeaves; ++idx) {
                    leafBvs[idx] = getbv(records[idx].second);

                    leafLca[idx] = -1, leafDepths[idx] = 1;
                    Ti l = idx - 1, r = idx; ///< (l, r]
                    bool mark{false};

                    if (l >= 0)
                        mark = splits[l] < splits[r]; ///< true when right child, false otherwise

                    int cur = mark ? l : r;
                    if (mark)
                        trunkRc[cur] = numTrunk + idx, trunkR[cur] = idx;
                    else
                        trunkLc[cur] = numTrunk + idx, trunkL[cur] = idx;

                    while (trunkBuildFlags[cur].fetch_add(1) == 1) {
                        { // refit
                            int lc = trunkLc[cur], rc = trunkRc[cur];
                            const auto &leftBox = lc >= numTrunk ? leafBvs[lc - numTrunk] : trunkBvs[lc];
                            const auto &rightBox = rc >= numTrunk ? leafBvs[rc - numTrunk] : trunkBvs[rc];
                            Box bv{};
                            for (int d = 0; d != 3; ++d) {
                                bv.first[d] =
                                    leftBox.first[d] < rightBox.first[d] ? leftBox.first[d] : rightBox.first[d];
                                bv.second[d] =
                                    leftBox.second[d] > rightBox.second[d] ? leftBox.second[d] : rightBox.second[d];
                            }
                            trunkBvs[cur] = bv;
                        }

                        l = trunkL[cur] - 1, r = trunkR[cur];
                        leafLca[l + 1] = cur, leafDepths[l + 1]++;
                        atomic_thread_fence(std::memory_order_acquire);

                        if (l >= 0)
                            mark = splits[l] < splits[r]; ///< true when right child, false otherwise
                        else
                            mark = false;

                        if (l + 1 == 0 && r == numLeaves - 1) {
                            // trunkPar(cur) = -1;
                            break;
                        }

                        int par = mark ? l : r;
                        // trunkPar(cur) = par;
                        if (mark) {
                            trunkRc[par] = cur, trunkR[par] = r;
                        } else {
                            trunkLc[par] = cur, trunkL[par] = l + 1;
                        }
                        cur = par;
                    }
                }
            }

            std::vector<Ti> leafOffsets(numLeaves + 1);
            leafOffsets[0] = 0;
            for (Ti i = 0; i != numLeaves; ++i)
                leafOffsets[i + 1] = leafOffsets[i] + leafDepths[i];
            std::vector<Ti> trunkDst(numLeaves - 1);
            /// compute trunk order
            // [levels], [parents], [trunkDst]
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti i = 0; i < numLeaves; ++i) {
                auto offset = leafOffsets[i];
                parents[offset] = -1;
                for (Ti node = leafLca[i], level = leafDepths[i]; --level; node = trunkLc[node]) {
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
                const auto &bv = trunkBvs[i];
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
                } else
                    auxIndices[dst] = -1;
            }

            /// reorder leaf
            // [sortedBvs], [auxIndices], [levels], [parents], [leafIndices]
            // auxIndices here is primitiveIndex (for leaf nodes)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti i = 0; i < numLeaves; ++i) {
                const auto &bv = leafBvs[i];
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

#endif // WXL
    }

    template <class Cond> float intersect(Cond cond, vec3f const &ro, vec3f const &rd) const {
        float ret = std::numeric_limits<float>::infinity();
#if WXL
        const auto &tris = prim->tris;
        if (tris.size() >= threshold) {
            const auto &verts = prim->verts;
            const Ti numLeaves = tris.size();
            const Ti numTrunk = numLeaves - 1;
            const Ti numNodes = numLeaves + numTrunk;
            Ti node = 0;
            while (node != -1 && node != numNodes) {
                Ti level = levels[node];
                // level and node are always in sync
                for (; level; --level, ++node)
                    if (!ray_box_intersect(ro, rd, sortedBvs[node]))
                        break;
                // leaf node check
                if (level == 0) {
                    const auto eid = auxIndices[node];
                    auto ind = tris[eid];
                    auto a = verts[ind[0]];
                    auto b = verts[ind[1]];
                    auto c = verts[ind[2]];
                    float d = tri_intersect(cond, ro, rd, a, b, c);
                    if (std::abs(d) < std::abs(ret))
                        ret = d;
                    if (d < ret) {
                        // id = eid;
                        ret = d;
                    }
                    node++;
                } else // separate at internal nodes
                    node = auxIndices[node];
            }
        } else {
            for (size_t i = 0; i < prim->tris.size(); i++) {
                auto ind = prim->tris[i];
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto c = prim->verts[ind[2]];
                float d = tri_intersect(cond, ro, rd, a, b, c);
                if (std::abs(d) < std::abs(ret))
                    ret = d;
            }
        }
#else
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            float d = tri_intersect(cond, ro, rd, a, b, c);
            if (std::abs(d) < std::abs(ret))
                ret = d;
        }
#endif
        return ret;
    }
};

struct PrimProject : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto targetPrim = get_input<PrimitiveObject>("targetPrim");
        auto offset = get_input2<float>("offset");
        auto limit = get_input2<float>("limit");
        auto nrmAttr = get_input2<std::string>("nrmAttr");
        auto allowDir = get_input2<std::string>("allowDir");

        BVH bvh;
        bvh.build(targetPrim.get());

        if (limit <= 0)
            limit = std::numeric_limits<float>::infinity();

        struct allow_front {
            bool operator()(float x) const {
                return x >= 0;
            }
        };

        struct allow_back {
            bool operator()(float x) const {
                return x <= 0;
            }
        };

        struct allow_both {
            bool operator()(float x) const {
                return true;
            }
        };

        auto const &nrm = prim->verts.attr<vec3f>(nrmAttr);
        auto cond = enum_variant<std::variant<allow_front, allow_back, allow_both>>(
            array_index({"front", "back", "both"}, allowDir));

        std::visit(
            [&](auto cond) {
                parallel_for((size_t)0, prim->verts.size(), [&](size_t i) {
                    auto ro = prim->verts[i];
                    auto rd = normalizeSafe(nrm[i]);
                    float t = bvh.intersect(cond, ro, rd);
                    if (std::abs(t) >= limit)
                        t = 0;
                    t -= offset;
                    prim->verts[i] = ro + t * rd;
                });
            },
            cond);

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimProject, {
                            {
                                {"PrimitiveObject", "prim"},
                                {"PrimitiveObject", "targetPrim"},
                                {"string", "nrmAttr", "nrm"},
                                {"float", "offset", "0"},
                                {"float", "limit", "0"},
                                {"enum front back both", "allowDir", "both"},
                            },
                            {
                                {"PrimitiveObject", "prim"},
                            },
                            {},
                            {"primitive"},
                        });

struct TestRayBox : INode {
    void apply() override {
        auto origin = get_input2<vec3f>("ray_origin");
        auto dir = get_input2<vec3f>("ray_dir");
        auto bmin = get_input2<vec3f>("box_min");
        auto bmax = get_input2<vec3f>("box_max");
        set_output("predicate",
                   std::make_shared<NumericObject>((int)ray_box_intersect(origin, dir, std::make_pair(bmin, bmax))));
    }
};

ZENDEFNODE(TestRayBox, {
                           {
                               {"ray_origin"},
                               {"ray_dir"},
                               {"box_min"},
                               {"box_max"},
                           },
                           {
                               {"predicate"},
                           },
                           {},
                           {"primitive"},
                       });

} // namespace
} // namespace zeno
