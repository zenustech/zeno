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
#include "zensim/geometry/AdaptiveGrid.hpp"
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

#define AG_CHECK 1
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
    template <typename TreeT>
    struct VdbConverter {
        using RootT = typename TreeT::RootNodeType;
        using LeafT = typename TreeT::LeafNodeType;
        using ValueT = typename LeafT::ValueType;
        static_assert(RootT::LEVEL == 3, "expects a tree of 3 levels (excluding root level)");
        using ZSGridT = zs::AdaptiveGrid<3, ValueT, RootT::NodeChainType::template Get<0>::LOG2DIM,
                                         RootT::NodeChainType::template Get<1>::LOG2DIM,
                                         RootT::NodeChainType::template Get<2>::LOG2DIM>;
        using ZSCoordT = zs::vec<int, 3>;

        VdbConverter(const std::vector<unsigned int> &nodeCnts)
            : cnts(nodeCnts), ag(std::make_shared<ZSGridT>()), success{std::make_shared<bool>()} {
            // for each level, initialize allocations
            using namespace zs;
            // name_that_type(ag->level(wrapv<0>{}));
            fmt::print("nodes per level: {}, {}, {}\n", cnts[0], cnts[1], cnts[2]);
            zs::get<0>(ag->_levels);
            ag->level(dim_c<0>) = RM_CVREF_T(ag->level(dim_c<0>))({{"sdf", 1}}, nodeCnts[0]);
            ag->level(dim_c<1>) = RM_CVREF_T(ag->level(dim_c<1>))({{"sdf", 1}}, nodeCnts[1]);
            ag->level(dim_c<2>) = RM_CVREF_T(ag->level(dim_c<2>))({{"sdf", 1}}, nodeCnts[2]);
            *success = true;
        }
        ~VdbConverter() = default;

        void operator()(RootT &rt) const {
            ag->_background = rt.background();
#if 0
            openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();
            vec<float, 4, 4> lsv2w;
            for (auto &&[r, c] : ndrange<2>(4))
                lsv2w(r, c) = v2w[r][c];
            ag->resetTransformation(lsv2w);
#endif
        }

        template <typename NodeT>
        void operator()(NodeT &node) const {
            using namespace zs;
            auto &level = ag->level(dim_c<NodeT::LEVEL>);
            using LevelT = RM_CVREF_T(level);
            if (auto tmp = node.getValueMask().countOn(); tmp != 0) {
                fmt::print("node [{}, {}, {}] has {} active values.\n", node.origin()[0], node.origin()[1],
                           node.origin()[2], tmp);
                *success = false;
                return;
            }
            auto &table = level.table;
            auto &grid = level.grid;
            auto &valueMask = level.valueMask;
            auto &childMask = level.childMask;
            auto tabv = proxy<execspace_e::openmp>(table);
            auto gridv = proxy<execspace_e::openmp>(grid);

            auto coord_ = node.origin();
            ZSCoordT coord{coord_[0], coord_[1], coord_[2]};
            auto bno = tabv.insert(coord);
            if (bno < 0 || bno >= cnts[NodeT::LEVEL]) {
                fmt::print("there are redundant threads inserting the same block ({}, {}, {}).\n", coord[0], coord[1],
                           coord[2]);
                *success = false;
                return;
            }

#if AG_CHECK
            if (bno + 1 == cnts[NodeT::LEVEL]) {
                fmt::print(fg(fmt::color::green), "just inserted the last block ({}) at level {}\n", bno, NodeT::LEVEL);
            }
#endif

            auto block = gridv.tile(bno);
            for (auto it = node.cbeginValueOn(); it; ++it) {
                block(0, it.pos()) = it.getValue();
#if 0
                if (bno == 0) {
                    fmt::print("l-{} block[{}, {}, {}] [{}] value: {}\n", NodeT::LEVEL, coord[0], coord[1], coord[2],
                               it.pos(), it.getValue());
                }
#endif
            }

            static_assert(sizeof(typename NodeT::NodeMaskType) == sizeof(typename LevelT::mask_type::value_type),
                          "???");

            std::memcpy(&childMask[bno], &node.getChildMask(), sizeof(childMask[bno]));
            std::memcpy(&valueMask[bno], &node.getValueMask(), sizeof(valueMask[bno]));
        }

        void operator()(LeafT &lf) const {
            using namespace zs;
            auto &level = ag->level(dim_c<0>);
            using LevelT = RM_CVREF_T(level);
            static_assert(LeafT::NUM_VALUES == ZSGridT::block_sizes[0], "????");

            auto &table = level.table;
            auto &grid = level.grid;
            auto &valueMask = level.valueMask;
            auto tabv = proxy<execspace_e::openmp>(table);
            auto gridv = proxy<execspace_e::openmp>(grid);

            auto coord_ = lf.origin();
            ZSCoordT coord{coord_[0], coord_[1], coord_[2]};
            auto bno = tabv.insert(coord);
            if (bno < 0 || bno >= cnts[0]) {
                fmt::print("there are redundant threads inserting the same leaf block ({}, {}, {}).\n", coord[0],
                           coord[1], coord[2]);
                *success = false;
                return;
            }
#if AG_CHECK
            if (bno + 1 == cnts[0]) {
                fmt::print(fg(fmt::color::green), "just inserted the last block ({}) at leaf level\n", bno);
            }
#endif

            auto block = gridv.tile(bno);
            for (auto it = lf.cbeginValueOn(); it; ++it) {
                block(0, it.pos()) = it.getValue();
#if 0
                if (bno == 0) {
                    fmt::print("leaf block[{}, {}, {}] [{}] value: {}\n", coord[0], coord[1], coord[2], it.pos(),
                               it.getValue());
                }
#endif
            }

            static_assert(sizeof(typename LeafT::NodeMaskType) == sizeof(typename LevelT::mask_type::value_type),
                          "???");
            std::memcpy(&valueMask[bno], &lf.getValueMask(), sizeof(valueMask[bno]));
        }

        ZSGridT &&get() {
            return std::move(*ag);
        }

        std::vector<unsigned int> cnts;
        std::shared_ptr<ZSGridT> ag;
        std::shared_ptr<bool> success;
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
        std::vector<int> cnts(4);
        IterateOp<TreeT> op(cnts);
        // for DynamicNodeManager, op is shared
        timer.tick();
        openvdb::tree::DynamicNodeManager<TreeT> nodeManager(sdf->tree());
        nodeManager.foreachTopDown(op);
        timer.tock("dynamic");

        // for NodeManager, op is copied
        timer.tick();
        openvdb::tree::NodeManager<TreeT> nm(sdf->tree());
        nm.foreachBottomUp(op);
        timer.tock("static");

        std::vector<unsigned int> nodeCnts(4);
        tree.root().nodeCount(nodeCnts);

        // target
        timer.tick();
        VdbConverter<TreeT> agBuilder(nodeCnts);
        nm.foreachBottomUp(agBuilder);
        timer.tock("build adaptive grid from vdb levelset");
        fmt::print("examine type: {}\n", get_type_str<typename RM_CVREF_T(agBuilder)::ZSGridT>());

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
