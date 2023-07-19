//#include "Structures.hpp"
//#include "Utils.hpp"
#include <cassert>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
//#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/container/Vector.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <random>

namespace zeno {

struct ZSLinkTest : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
#if 0
        using namespace zs;
        zs::initialize_openvdb();
        zs::Vector<int> a{ 100, memsrc_e::host, -1 };
        a[0] = 100;
        fmt::print("first element: {}\n", a[0]);
#endif
        printf("loaded!\n");
        zs::Vector<float> keys{1000};
        zs::Vector<int> vals{1000};
        using Pair = std::pair<float, int>;
        auto pol = zs::omp_exec();
        /// init
        pol(range(keys.size()), [&](int i) {
            std::mt19937 rng;
            rng.seed(i);
            keys[i] = (float)rng();
            vals[i] = i;
        });
        /// ref init
        std::vector<Pair> kvs(1000);
        pol(range(kvs.size()), [&](int i) { kvs[i] = std::make_pair(keys[i], vals[i]); });

        merge_sort_pair(pol, std::begin(keys), std::begin(vals), keys.size());
        merge_sort(pol, std::begin(kvs), std::end(kvs));

        for (auto [no, k, v, kv] : enumerate(keys, vals, kvs)) {
            if (k != kv.first || v != kv.second) {
                fmt::print("divergence at [{}] k: {}, v: {}, kv: <{}, {}>\n", no, k, v, kv.first, kv.second);
            }
        }
        for (int i = 0; i != 10; ++i) {
            fmt::print("[{}] k: {}, v: {}; kv: {}, {}\n", i, keys[i], vals[i], kvs[i].first, kvs[i].second);
            int ii = kvs.size() - 1 - i;
            fmt::print("[{}] k: {}, v: {}; kv: {}, {}\n", ii, keys[ii], vals[ii], kvs[ii].first, kvs[ii].second);
        }
        getchar();
    }
};

ZENDEFNODE(ZSLinkTest, {
                           {},
                           {},
                           {},
                           {"ZPCTest"},
                       });

} // namespace zeno
