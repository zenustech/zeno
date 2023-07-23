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
#include "zensim/execution/Atomics.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <random>
#include <zeno/VDBGrid.h>

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

struct TestAdaptiveGrid : INode {
    template <typename TreeT>
    struct IterateOp {
        using RootT = typename TreeT::RootNodeType;
        using LeafT = typename TreeT::LeafNodeType;

        IterateOp(std::vector<int> &cnts) : cnts(cnts) {
        }
        ~IterateOp() = default;

        // Processes the root node. Required by the DynamicNodeManager
        bool operator()(RootT &rt, size_t) const {
            using namespace zs;
            cnts[RootT::getLevel()]++;
            using namespace zs;
            atomic_add(exec_omp, &cnts[RootT::ChildNodeType::getLevel()], (int)rt.childCount());
            return true;
        }
        void operator()(RootT &rt) const {
            atomic_add(zs::exec_omp, &cnts[rt.getLevel()], 1);
        }

        // Processes the internal nodes. Required by the DynamicNodeManager
        template <typename NodeT>
        bool operator()(NodeT &node, size_t idx) const {
            using namespace zs;
            if (auto tmp = node.getValueMask().countOn(); tmp != 0)
                fmt::print("node [{}, {}, {}] has {} active values.\n", node.origin()[0], node.origin()[1],
                           node.origin()[2], tmp);
            if constexpr (NodeT::ChildNodeType::LEVEL == 0) {
                atomic_add(exec_omp, &cnts[NodeT::ChildNodeType::getLevel()], (int)node.getChildMask().countOn());
                return true;
            } else {
                atomic_add(exec_omp, &cnts[NodeT::ChildNodeType::getLevel()], (int)node.getChildMask().countOn());
                for (auto iter = node.cbeginChildOn(); iter; ++iter) {
                }
                if (node.getValueMask().countOn() > 0)
                    fmt::print("there exists internal node [{}] that have adaptive values!\n", node.origin()[0]);
            }
            return true;
        }
        template <typename NodeT>
        void operator()(NodeT &node) const {
            atomic_add(zs::exec_omp, &cnts[node.getLevel()], 1);
        }
        // Processes the leaf nodes. Required by the DynamicNodeManager
        bool operator()(LeafT &, size_t) const {
            return true;
        }
        void operator()(LeafT &lf) const {
            atomic_add(zs::exec_omp, &cnts[0], 1);
        }

        std::vector<int> &cnts;
    };
    void apply() override {
        using namespace zs;
        openvdb::FloatGrid::Ptr sdf;
        if (has_input("sdf"))
            sdf = get_input("sdf")->as<VDBFloatGrid>()->m_grid;
        else {
            sdf = zs::load_floatgrid_from_vdb_file("/home/mine/Codes/zeno2/zeno/assets/tozeno.vdb")
                      .as<openvdb::FloatGrid::Ptr>();
        }
        using Adapter = openvdb::TreeAdapter<openvdb::FloatGrid>;
        using TreeT = typename Adapter::TreeType;
        auto &tree = Adapter::tree(*sdf);
        // fmt::print("TreeT: {}, tree: {}\n", get_type_str<TreeT>(), get_var_type_str(tree));
        static_assert(is_same_v<TreeT, RM_CVREF_T(tree)>, "???");
        fmt::print("root: {}\n", get_var_type_str(tree.root()));

        CppTimer timer;
        std::vector<int>  cnts(4);
        IterateOp<TreeT> op(cnts);
#if 1
        timer.tick();
        openvdb::tree::DynamicNodeManager<TreeT> nodeManager(sdf->tree());
        nodeManager.foreachTopDown(op);
        timer.tock("dynamic");
#endif
        timer.tick();
        openvdb::tree::NodeManager<TreeT> nm(sdf->tree());
        nm.foreachBottomUp(op);
        timer.tock("static");
        ;
        std::vector<unsigned int> nodeCnts(4);
        tree.root().nodeCount(nodeCnts);
        fmt::print("ref node cnts: {}, {}, {}, {}\n", nodeCnts[0], nodeCnts[1], nodeCnts[2], nodeCnts[3]);
        fmt::print("calced node cnts: {}, {}, {}, {}\n", op.cnts[0], op.cnts[1], op.cnts[2], op.cnts[3]);
    }
};

ZENDEFNODE(TestAdaptiveGrid, {
                                 {"sdf"},
                                 {},
                                 {},
                                 {"ZPCTest"},
                             });

} // namespace zeno
