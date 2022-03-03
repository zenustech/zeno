#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"
#include <cmath>
#include <atomic>
#include <algorithm>
// #if defined(_OPENMP)
#include <omp.h>
// #endif

namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    int which = 0;
};

struct LBvh : zeno::IObject {
    enum element_e {point = 0, line, tri, tet};
    using TV = zeno::vec3f;
    using Box = std::pair<TV, TV>;
    using Ti = int;
    using Tu = std::make_unsigned_t<Ti>;

    std::weak_ptr<const zeno::PrimitiveObject> primPtr;
    float thickness;
    std::vector<Box> sortedBvs;
    std::vector<Ti> auxIndices, levels, parents, leafIndices;

    decltype(auto) refPrimPositions() const {
        std::shared_ptr<const zeno::PrimitiveObject> pprim = primPtr.lock();
        if (!pprim)
            throw std::runtime_error("the primitive object referenced by lbvh not available anymore");
        return pprim->attr<zeno::vec3f>("pos");
    }

    LBvh(const std::shared_ptr<zeno::PrimitiveObject> &prim, float thickness)
        : primPtr(prim), thickness{thickness} {

        const auto &refpos = refPrimPositions();

        const Ti numLeaves = refpos.size();
        const Ti numNodes = numLeaves + numLeaves - 1;
        sortedBvs.resize(numNodes);
        auxIndices.resize(numNodes);
        levels.resize(numNodes);
        parents.resize(numNodes);
        leafIndices.resize(numLeaves);

        constexpr int dim = 3;
        constexpr auto ma = std::numeric_limits<float>().max();
        constexpr auto mi = std::numeric_limits<float>().lowest();
        Box wholeBox{TV{ma, ma, ma}, TV{mi, mi, mi}};

        /// whole box
        // should use reduce here
        for (Ti i = 0; i != numLeaves; ++i) {
            const auto &p = refpos[i];
            for (int d = 0; d != dim; ++d) {
                if (p[d] < wholeBox.first[d])
                    wholeBox.first[d] = p[d];
                if (p[d] > wholeBox.second[d])
                    wholeBox.second[d] = p[d];
            }
        }
        // printf("lbvh bounding box: %f, %f, %f - %f, %f, %f\n", wholeBox.first[0], wholeBox.first[1], wholeBox.first[2], wholeBox.second[0], wholeBox.second[1], wholeBox.second[2]);

        std::vector<std::pair<Tu, Ti>> records(numLeaves);  // <mc, id>
        /// morton codes 
        auto getMortonCode = [](const TV &p) -> Tu {
            auto expand_bits = [](Tu v) -> Tu {  // expands lower 10-bits to 30 bits
                v = (v * 0x00010001u) & 0xFF0000FFu;
                v = (v * 0x00000101u) & 0x0F00F00Fu;
                v = (v * 0x00000011u) & 0xC30C30C3u;
                v = (v * 0x00000005u) & 0x49249249u;
                return v;
            };
            return (expand_bits((Tu)(p[0] * 1024.f)) << (Tu)2) | (expand_bits((Tu)(p[1] * 1024.f)) << (Tu)1) | expand_bits((Tu)(p[2] * 1024.f));
        };
        {
            const auto lengths = wholeBox.second - wholeBox.first;
            auto getUniformCoord = [&wholeBox, &lengths](const TV &p) {
                // https://newbedev.com/constexpr-variable-captured-inside-lambda-loses-its-constexpr-ness
                constexpr int dim = 3;
                auto offsets = p - wholeBox.first;
                for (int d = 0; d != dim; ++d)
                    offsets[d] = std::clamp(offsets[d], (float)0, lengths[d]) / lengths[d];
                return offsets;
            };
#pragma omp parallel for
            for (Ti i = 0; i < numLeaves; ++i) {
                auto uc = getUniformCoord(refpos[i]);
                records[i] = std::make_pair(getMortonCode(uc), i);
            }
        }
        std::sort(std::begin(records), std::end(records));

        std::vector<Tu> splits(numLeaves);
        /// 
        constexpr auto numTotalBits = sizeof(Tu) * 8;
        auto clz = [](Tu x) -> Tu {
            static_assert(std::is_same_v<Tu, unsigned int>, "Tu should be unsigned int");
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
            return __lzcnt((unsigned int)x);
#elif defined(__clang__) || defined(__GNUC__)
            return __builtin_clz((unsigned int)x);
#endif
        };
#pragma omp parallel for
        for (Ti i = 0; i < numLeaves; ++i) {
            if (i != numLeaves - 1)
                splits[i] = numTotalBits - clz(records[i].first ^ records[i+1].first);
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
#pragma omp parallel for
        for (Ti i = 0; i < numLeaves - 1; ++i) {
            trunkTopoMarks[i] = 0;
            trunkBuildFlags[i] = 0;
        }

#if 0
        for (Ti i = 0; i != numLeaves; ++i) {
            const auto pid = records[i].second;
            printf("prim [%d]: morton code (%x) (%f, %f, %f), original prim id (%d), splits (%d)\n", i, records[i].first, refpos[pid][0], refpos[pid][1], refpos[pid][2], pid, splits[i]);
        }
#endif

        {
        std::vector<Ti> trunkL(numLeaves - 1);
        std::vector<Ti> trunkRc(numLeaves - 1);
#pragma omp parallel for
        for (Ti idx = 0; idx < numLeaves; ++idx) {
            {
                const auto &pos = refpos[records[idx].second];
                leafBvs[idx] = Box{pos - thickness, pos + thickness};
            }

            leafLca[idx] = -1, leafDepths[idx] = 1;
            Ti l = idx - 1, r = idx;  ///< (l, r]
            bool mark{false};

            if (l >= 0) mark = splits[l] < splits[r];  ///< true when right child, false otherwise

            int cur = mark ? l : r;
            if (mark)
                trunkRc[cur] = idx, trunkR[cur] = idx, trunkTopoMarks[cur].fetch_or((Tu)0x00000002u);
            else
                trunkLc[cur] = idx, trunkL[cur] = idx, trunkTopoMarks[cur].fetch_or((Tu)0x00000001u);

            while (trunkBuildFlags[cur].fetch_add(1) == 1) {
                {  // refit
                    int lc = trunkLc[cur], rc = trunkRc[cur];
                    const auto childMask = trunkTopoMarks[cur] & (Tu)3;
                    const auto& leftBox = (childMask & 1) ? leafBvs[lc] : trunkBvs[lc];
                    const auto& rightBox = (childMask & 2) ? leafBvs[rc] : trunkBvs[rc];
                    Box bv{};
                    for (int d = 0; d != dim; ++d) {
                        bv.first[d] = leftBox.first[d] < rightBox.first[d] ? leftBox.first[d] : rightBox.first[d];
                        bv.second[d] = leftBox.second[d] > rightBox.second[d] ? leftBox.second[d] : rightBox.second[d];
                    }
                    trunkBvs[cur] = bv;
                }
                trunkTopoMarks[cur] &= 0x00000007;

                l = trunkL[cur] - 1, r = trunkR[cur];
                leafLca[l + 1] = cur, leafDepths[l + 1]++;
                atomic_thread_fence(std::memory_order_acquire);

                if (l >= 0)
                    mark = splits[l] < splits[r];  ///< true when right child, false otherwise
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
#pragma omp parallel for
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
        // levels store the number of node within the left-child-branch from bottom up starting from 0
#if 0
        for (Ti i = 0; i != numLeaves; ++i) {
            const auto pid = records[i].second;
            fmt::print("leaf {} (pid {}, offset {}, levels {}) morton code: {:x}, lca: {}, depth: {}\n", i, pid, leafOffsets[i], levels[leafOffsets[i + 1] - 1], records[i].first, leafLca[i], leafDepths[i]);
        }

        for (Ti i = 0; i != numLeaves - 1; ++i) {
            fmt::print("trunk {}\t(-> #{})\t[{}, {}];\t chs<{}, {}>;\tmarks[{:x}]\n", i, trunkDst[i], trunkL[i], trunkR[i], trunkLc[i], trunkRc[i], trunkTopoMarks[i]);
        }

        for (Ti i = 0; i != numNodes; ++i) {
            fmt::print("bvh[{}]\t parents: {}, levels: {}\n", i, parents[i], levels[i]);
        }
#endif

        /// reorder trunk
        // [sortedBvs], [auxIndices], [parents]
        // auxIndices here is escapeIndex (for trunk nodes)
#pragma omp parallel for
        for (Ti i = 0; i < numLeaves - 1; ++i) {
            const auto dst = trunkDst[i];
            const auto &bv = trunkBvs[i];
            // auto l = trunkL[i];
            auto r = trunkR[i];
            sortedBvs[dst] = bv;
            const auto rb = r + 1;
            if (rb < numLeaves) {
                auto lca = leafLca[rb];  // rb must be in left-branch
                auto brother = (lca != -1 ? trunkDst[lca] : leafOffsets[rb]);
                auxIndices[dst] = brother;
                if (parents[dst] == dst - 1)
                    parents[brother] = dst - 1;            // setup right-branch brother's parent
            } else
                auxIndices[dst] = -1;
        }

#if 0
        for (Ti i = 0; i != numNodes; ++i) 
            fmt::print("bvh[{}]\t escape indices: {}, parent: {}\n", i, auxIndices[i], parents[i]);
#endif

        /// reorder leaf
        // [sortedBvs], [auxIndices], [levels], [parents], [leafIndices]
        // auxIndices here is primitiveIndex (for leaf nodes)
#pragma omp parallel for
        for (Ti i = 0; i < numLeaves; ++i) {
            const auto &bv = leafBvs[i];
            // const auto leafDepth = leafDepths[i];

            auto dst = leafOffsets[i + 1] - 1;
            leafIndices[i] = dst;
            sortedBvs[dst] = bv;
            auxIndices[dst] = records[i].second;
            levels[dst] = 0;
            if (parents[dst] == dst - 1) 
                parents[dst + 1] = dst - 1;  // setup right-branch brother's parent
            // if (leafDepth > 1) parents[dst + 1] = dst - 1;  // setup right-branch brother's parent
        }
#if 0
        for (Ti i = 0; i != numNodes; ++i) 
            fmt::print("bvh[{}]\t escape indices: {}, parent: {}\n", i, auxIndices[i], parents[i]);
#endif
    }

    static bool intersect(const Box& box, const TV &p) noexcept {
        constexpr int dim = 3;
        for (Ti d = 0; d != dim; ++d)
            if (p[d] < box.first[d] || p[d] > box.second[d])
                return false;
        return true;
    }
    static constexpr float distance(const Box& bv, const TV &x) {
        const auto &[mi, ma] = bv;
        TV center = (mi + ma) / 2;
        TV point = abs(x - center) - (ma - mi) / 2;
        float max = std::numeric_limits<float>::lowest();
        for (int d = 0; d != 3; ++d) {
            if (point[d] > max) max = point[d];
            if (point[d] < 0) point[d] = 0;
        }
        return (max < 0.f ? max : 0.f) + length(point);
    }
    static constexpr float distance(const TV &x, const Box& bv) {
        return distance(bv, x);
    }

    /// closest bounding box
    template <class F>
    void find_nearest(TV const &pos, F const &f, Ti& id, float &dist) const {
        const auto &refpos = refPrimPositions();
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
                const auto pid = auxIndices[node];
                auto d = length(refpos[pid] - pos);
                if (d < dist) {
                    id = pid;
                    dist = d;
                }
                node++;
            } else  // separate at internal nodes
                node = auxIndices[node];
        }
    }

    template <class F>
    void iter_neighbors(TV const &pos, F const &f) const {
        const Ti numNodes = sortedBvs.size();
        Ti node = 0;
        while (node != -1 && node != numNodes) {
            Ti level = levels[node];
            // level and node are always in sync
            for (; level; --level, ++node)
                if (!intersect(sortedBvs[node], pos))
                    break;
            // leaf node check
            if (level == 0) {
                // const auto dist = refpos[pid] - pos;
                // auto dis2 = zeno::dot(dist, dist);
                // if (dis2 <= radius_sqr && dis2 > radius_sqr_min) {
                if (intersect(sortedBvs[node], pos))
                    f(auxIndices[node]);
                // }
                node++;
            } else  // separate at internal nodes
                node = auxIndices[node];
        }
    }
};

struct HashGrid : zeno::IObject {
    float inv_dx;
    float radius;
    float radius_sqr;
    float radius_sqr_min;
    std::vector<zeno::vec3f> const &refpos;

    std::vector<std::vector<int>> table;

//#define DILEI
#define XUBEN

    int hash(int x, int y, int z) {
#ifdef XUBEN
        return (x%gridRes[0]+gridRes[0])%gridRes[0] + ((y%gridRes[1]+gridRes[1])%gridRes[1]) * gridRes[0] + ((z%gridRes[2]+gridRes[2])%gridRes[2]) * gridRes[0] * gridRes[1];
#else
        return ((73856093 * x) ^ (19349663 * y) ^ (83492791 * z)) % table.size();
#endif
    }

#ifdef XUBEN
    zeno::vec3f pMin, pMax;
    zeno::vec3i gridRes;
#endif

    HashGrid(std::vector<zeno::vec3f> const &refpos_,
            float radius_, float radius_min)
        : refpos(refpos_) {

        radius = radius_;
        radius_sqr = radius * radius;
        radius_sqr_min = radius_min < 0.f ? -1.f : radius_min * radius_min;
#ifdef DILEI
        inv_dx = 0.5f / radius;
#else
        inv_dx = 1.0f / radius;
#endif

#ifdef XUBEN
        pMin = refpos[0];
        pMax = refpos[0];
        for (int i = 1; i < refpos.size(); i++) {
            auto coor = refpos[i];
            pMin = zeno::min(pMin, coor);
            pMax = zeno::max(pMax, coor);
        }
        pMin -= radius;
        pMax += radius;
        gridRes = zeno::toint(zeno::floor((pMax - pMin) * inv_dx)) + 1;

        dbg_printf("grid res: %dx%dx%d\n", gridRes[0], gridRes[1], gridRes[2]);
        table.clear();
        table.resize(gridRes[0] * gridRes[1] * gridRes[2]);
#else
        int table_size = refpos.size() / 8;
        dbg_printf("table size: %d\n", table_size);
        table.clear();
        table.resize(table_size);
#endif

        for (int i = 0; i < refpos.size(); i++) {
#ifdef XUBEN
            auto coor = zeno::toint(zeno::floor((refpos[i] - pMin) * inv_dx));
#else
            auto coor = zeno::toint(zeno::floor(refpos[i] * inv_dx));
#endif
            auto key = hash(coor[0], coor[1], coor[2]);
            table[key].push_back(i);
        }
    }

    template <class F>
    void iter_neighbors(zeno::vec3f const &pos, F const &f) {
#ifdef XUBEN
#ifdef DILEI
        auto coor = zeno::toint(zeno::floor((pos - pMin) * inv_dx - 0.5f));
#else
        auto coor = zeno::toint(zeno::floor((pos - pMin) * inv_dx));
#endif
#else
#ifdef DILEI
        auto coor = zeno::toint(zeno::floor(pos * inv_dx - 0.5f));
#else
        auto coor = zeno::toint(zeno::floor(pos * inv_dx));
#endif
#endif
#ifdef DILEI
        for (int dz = 0; dz < 2; dz++) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
#else
        for (int dz = -1; dz < 2; dz++) {
            for (int dy = -1; dy < 2; dy++) {
                for (int dx = -1; dx < 2; dx++) {
#endif
                    int key = hash(coor[0] + dx, coor[1] + dy, coor[2] + dz);
                    for (int pid: table[key]) {
                        //auto dist = refpos[pid] - pos;
                        //auto dis2 = zeno::dot(dist, dist);
                        //if (dis2 <= radius_sqr && dis2 > radius_sqr_min) {
                            f(pid);
                        //}
                    }
                }
            }
        }
    }
};

static void vectors_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<Buffer> const &chs
    , std::vector<Buffer> const &chs2
    , std::vector<zeno::vec3f> const &pos
    , HashGrid *hashgrid
    ) {
    if (chs.size() == 0)
        return;

    #pragma omp parallel for
    for (int i = 0; i < pos.size(); i++) {
        auto ctx = exec->make_context();
        for (int k = 0; k < chs.size(); k++) {
            if (!chs[k].which)
                ctx.channel(k)[0] = chs[k].base[chs[k].stride * i];
        }
        hashgrid->iter_neighbors(pos[i], [&] (int pid) {
            for (int k = 0; k < chs.size(); k++) {
                if (chs[k].which)
                    ctx.channel(k)[0] = chs2[k].base[chs2[k].stride * pid];
            }
            ctx.execute();
        });
        for (int k = 0; k < chs.size(); k++) {
            if (!chs[k].which)
                chs[k].base[chs[k].stride * i] = ctx.channel(k)[0];
        }
    }
}

static void bvh_vectors_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<Buffer> const &chs
    , std::vector<Buffer> const &chs2
    , std::vector<zeno::vec3f> const &pos
    , LBvh *lbvh
    ) {
    if (chs.size() == 0)
        return;

    #pragma omp parallel for
    for (int i = 0; i < pos.size(); i++) {
        auto ctx = exec->make_context();
        for (int k = 0; k < chs.size(); k++) {
            if (!chs[k].which)
                ctx.channel(k)[0] = chs[k].base[chs[k].stride * i];
        }
        lbvh->iter_neighbors(pos[i], [&] (int pid) {
            for (int k = 0; k < chs.size(); k++) {
                if (chs[k].which)
                    ctx.channel(k)[0] = chs2[k].base[chs2[k].stride * pid];
            }
            ctx.execute();
        });
        for (int k = 0; k < chs.size(); k++) {
            if (!chs[k].which)
                chs[k].base[chs[k].stride * i] = ctx.channel(k)[0];
        }
    }
}

struct ParticlesBuildHashGrid : zeno::INode {
    virtual void apply() override {
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        float radius = get_input<zeno::NumericObject>("radius")->get<float>();
        float radiusMin = has_input("radiusMin") ?
            get_input<zeno::NumericObject>("radiusMin")->get<float>() : -1.f;
        auto hashgrid = std::make_shared<HashGrid>(
                primNei->attr<zeno::vec3f>("pos"), radius, radiusMin);
        set_output("hashGrid", std::move(hashgrid));
    }
};

ZENDEFNODE(ParticlesBuildHashGrid, {
    {{"PrimitiveObject", "primNei"}, {"numeric:float", "radius"}, {"numeric:float", "radiusMin"}},
    {{"hashgrid", "hashGrid"}},
    {},
    {"zenofx"},
});

struct ParticlesBuildBvh : zeno::INode {
    virtual void apply() override {
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        float radius = get_input<zeno::NumericObject>("radius")->get<float>();
        float radiusMin = has_input("radiusMin") ?
            get_input<zeno::NumericObject>("radiusMin")->get<float>() : -1.f;
        auto lbvh = std::make_shared<LBvh>(primNei, radius);
        set_output("lbvh", std::move(lbvh));
    }
};

ZENDEFNODE(ParticlesBuildBvh, {
    {{"PrimitiveObject", "primNei"}, {"numeric:float", "radius"}, {"numeric:float", "radiusMin"}},
    {{"LBvh", "lbvh"}},
    {},
    {"zenofx"},
});

struct ParticlesNeighborWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        auto hashgrid = get_input<HashGrid>("hashGrid");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        opts.detect_new_symbols = true;
        prim->foreach_attr([&] (auto const &key, auto const &attr) {
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
            dbg_printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
        });
        primNei->foreach_attr([&] (auto const &key, auto const &attr) {
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
            dbg_printf("define symbol: @@%s dim %d\n", key.c_str(), dim);
            opts.define_symbol("@@" + key, dim);
        });

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, obj]: params->lut) {
            auto key = '$' + key_;
            if (auto o = zeno::silent_any_cast<zeno::NumericValue>(obj); o.has_value()) {
                auto par = o.value();
                auto dim = std::visit([&] (auto const &v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, zeno::vec3f>) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parvals.push_back(v[2]);
                        parnames.emplace_back(key, 0);
                        parnames.emplace_back(key, 1);
                        parnames.emplace_back(key, 2);
                        return 3;
                    } else if constexpr (std::is_same_v<T, float>) {
                        parvals.push_back(v);
                        parnames.emplace_back(key, 0);
                        return 1;
                    } else {
                        printf("invalid parameter type encountered: `%s`\n",
                                typeid(T).name());
                        return 0;
                    }
                }, par);
                dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
                opts.define_param(key, dim);
            }
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        for (auto const &[name, dim]: prog->newsyms) {
            dbg_printf("auto-defined new attribute: %s with dim %d\n",
                    name.c_str(), dim);
            assert(name[0] == '@');
            if (name[1] == '@') {
                dbg_printf("ERROR: cannot define new attribute %s on primNei\n",
                        name.c_str());
            }
            auto key = name.substr(1);
            if (dim == 3) {
                prim->add_attr<zeno::vec3f>(key);
            } else if (dim == 1) {
                prim->add_attr<float>(key);
            } else {
                dbg_printf("ERROR: bad attribute dimension for primitive: %d\n",
                    dim);
                abort();
            }
        }

        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            dbg_printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '$');
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            dbg_printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        std::vector<Buffer> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            zeno::PrimitiveObject *primPtr;
            if (name[1] == '@') {
                name = name.substr(2);
                primPtr = primNei.get();
                iob.which = 1;
            } else {
                name = name.substr(1);
                primPtr = prim.get();
                iob.which = 0;
            }
            prim->attr_visit(name, [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
            chs[i] = iob;
        }
        std::vector<Buffer> chs2(prog->symbols.size());
        for (int i = 0; i < chs2.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            zeno::PrimitiveObject *primPtr;
            if (name[1] == '@') {
                name = name.substr(2);
                primPtr = primNei.get();
                iob.which = 1;
            } else {
                name = name.substr(1);
                primPtr = prim.get();
                iob.which = 0;
            }
            primNei->attr_visit(name, [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
            chs2[i] = iob;
        }

        vectors_wrangle(exec, chs, chs2, prim->attr<zeno::vec3f>("pos"),
                hashgrid.get());

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesNeighborWrangle, {
    {{"PrimitiveObject", "prim"}, {"PrimitiveObject", "primNei"}, {"HashGrid", "hashGrid"},
     {"string", "zfxCode"}, {"DictObject:NumericObject", "params"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"zenofx"},
});


struct ParticlesNeighborBvhWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        auto lbvh = get_input<LBvh>("lbvh");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        opts.detect_new_symbols = true;
        prim->foreach_attr([&] (auto const &key, auto const &attr) {
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
            dbg_printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
        });
        primNei->foreach_attr([&] (auto const &key, auto const &attr) {
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
            dbg_printf("define symbol: @@%s dim %d\n", key.c_str(), dim);
            opts.define_symbol("@@" + key, dim);
        });

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, obj]: params->lut) {
            auto key = '$' + key_;
            if (auto o = zeno::silent_any_cast<zeno::NumericValue>(obj); o.has_value()) {
                auto par = o.value();
                auto dim = std::visit([&] (auto const &v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, zeno::vec3f>) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parvals.push_back(v[2]);
                        parnames.emplace_back(key, 0);
                        parnames.emplace_back(key, 1);
                        parnames.emplace_back(key, 2);
                        return 3;
                    } else if constexpr (std::is_same_v<T, float>) {
                        parvals.push_back(v);
                        parnames.emplace_back(key, 0);
                        return 1;
                    } else {
                        printf("invalid parameter type encountered: `%s`\n",
                                typeid(T).name());
                        return 0;
                    }
                }, par);
                dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
                opts.define_param(key, dim);
            }
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        for (auto const &[name, dim]: prog->newsyms) {
            dbg_printf("auto-defined new attribute: %s with dim %d\n",
                    name.c_str(), dim);
            assert(name[0] == '@');
            if (name[1] == '@') {
                dbg_printf("ERROR: cannot define new attribute %s on primNei\n",
                        name.c_str());
            }
            auto key = name.substr(1);
            if (dim == 3) {
                prim->add_attr<zeno::vec3f>(key);
            } else if (dim == 1) {
                prim->add_attr<float>(key);
            } else {
                dbg_printf("ERROR: bad attribute dimension for primitive: %d\n",
                    dim);
                abort();
            }
        }

        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            dbg_printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '$');
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            dbg_printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        std::vector<Buffer> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            zeno::PrimitiveObject *primPtr;
            if (name[1] == '@') {
                name = name.substr(2);
                primPtr = primNei.get();
                iob.which = 1;
            } else {
                name = name.substr(1);
                primPtr = prim.get();
                iob.which = 0;
            }
            prim->attr_visit(name, [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
            chs[i] = iob;
        }
        std::vector<Buffer> chs2(prog->symbols.size());
        for (int i = 0; i < chs2.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            zeno::PrimitiveObject *primPtr;
            if (name[1] == '@') {
                name = name.substr(2);
                primPtr = primNei.get();
                iob.which = 1;
            } else {
                name = name.substr(1);
                primPtr = prim.get();
                iob.which = 0;
            }
            primNei->attr_visit(name, [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
            chs2[i] = iob;
        }

        bvh_vectors_wrangle(exec, chs, chs2, prim->attr<zeno::vec3f>("pos"),
                lbvh.get());

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesNeighborBvhWrangle, {
    {{"PrimitiveObject", "prim"}, {"PrimitiveObject", "primNei"}, {"LBvh", "lbvh"},
     {"string", "zfxCode"}, {"DictObject:NumericObject", "params"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"zenofx"},
});

}
