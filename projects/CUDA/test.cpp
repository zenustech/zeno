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
                fmt::print(fg(fmt::color::green), "just inserted the last block ({}) at level {} ({:#0x}, {:#0x})\n",
                           bno, NodeT::LEVEL, LevelT::origin_mask, LevelT::cell_mask);
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
            static_assert(LeafT::NUM_VALUES == ZSGridT::template get_tile_size<0>(), "????");

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
                fmt::print(fg(fmt::color::green), "just inserted the last block ({}) at leaf level ({:#0x}, {:#0x})\n",
                           bno, LevelT::origin_mask, LevelT::cell_mask);
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
        constexpr auto space = execspace_e::openmp;

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

        auto pol = omp_exec();
        /// construct new vdb grid from ag
        /// ref: nanovdb/util/NanoToOpenVDB.h
        auto zsag = agBuilder.get();
        zsag.reorder(pol);

        // test ag view
        using ZSCoordT = zs::vec<int, 3>;
        auto zsagv = view<space>(zsag);
        float v, vref;

        std::vector<ZSCoordT> coords(10000);
        std::mt19937 rng;
        for (auto &c : coords) {
            c[0] = rng() % 30;
            c[1] = rng() % 30;
            c[2] = rng() % 30;
        }
        ZSCoordT c{0, 1, 0};
        timer.tick();
        for (auto &c : coords) {
            (void)(sdf->tree().probeValue(openvdb::Coord{c[0], c[1], c[2]}, vref));
        }
        timer.tock("query (vdb probe)");
        timer.tick();
        for (auto &c : coords) {
            bool found = zsagv.probeValue(0, c, v);
        }
        timer.tock("naive query (zs probe)");
        timer.tick();
        for (auto &c : coords) {
            bool found = zsagv.probeValue(0, c, v, true_c);
        }
        timer.tock("fast query (zs probe)");

        // probe
        for (auto &c : coords) {
            bool found = zsagv.probeValue(0, c, v);
            sdf->tree().probeValue(openvdb::Coord{c[0], c[1], c[2]}, vref);
            if (v != vref && found) {
                //fmt::print("zs ag unnamed view type: {}\n", get_var_type_str(zsagv));
                //fmt::print("zs ag unnamed view type tuple of level views: {}\n", get_var_type_str(zsagv._levels));
                //fmt::print(fg(fmt::color::yellow), "background: {}\n", zsagv._background);
                fmt::print(fg(fmt::color::green), "probed value is {} ({}) at {}, {}, {}. is background: {}\n", v, vref,
                           c[0], c[1], c[2], !found);
            }
        }
        for (auto &c : coords) {
            bool found = zsagv.probeValue(0, c, v, true_c);
            sdf->tree().probeValue(openvdb::Coord{c[0], c[1], c[2]}, vref);
            if (v != vref && found) {
                fmt::print(fg(fmt::color::green), "probed value is {} ({}) at {}, {}, {}. is background: {} ({})\n", v,
                           vref, c[0], c[1], c[2], !found, zsag._background);
            }
        }

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf);
        timer.tick();
        for (auto &c : coords) {
            auto cc = c.cast<f32>() / 3;
            (void)(sampler.isSample(openvdb::Vec3R(cc[0], cc[1], cc[2])));
        }
        timer.tock("query (vdb sample)");

        timer.tick();
        for (auto &c : coords) {
            auto cc = c.cast<f32>() / 3;
            (void)(zsagv.iSample(0, cc));
        }
        timer.tock("query (zs sample)");

        // sample
        for (auto &c : coords) {
            auto cc = c.cast<f32>() / 3;
            auto vv = zsagv.iSample(0, cc);
            openvdb::FloatGrid::ValueType vref = sampler.isSample(openvdb::Vec3R(cc[0], cc[1], cc[2]));
            if (zs::abs(vref - vv) >= limits<float>::epsilon()) {
                fmt::print(fg(fmt::color::green), "sampled value is {} ({}) at {}, {}, {}\n", v, vv, vref, cc[0], cc[1],
                           cc[2]);
            }
        }
        fmt::print("done cross-checking all values.\n");

        //openvdb::Mat4R
        auto trans = sdf->transform().baseMap()->getAffineMap()->getMat4();

        auto ret = std::make_shared<VDBFloatGrid>();
        auto &dstGrid = ret->m_grid;
        dstGrid = openvdb::createGrid<openvdb::FloatGrid>(zsag._background);
        {
            using GridType = openvdb::FloatGrid;
            using TreeType = GridType::TreeType;
            using RootType = TreeType::RootNodeType;  // level 3 RootNode
            using Int2Type = RootType::ChildNodeType; // level 2 InternalNode
            using Int1Type = Int2Type::ChildNodeType; // level 1 InternalNode
            using LeafType = TreeType::LeafNodeType;  // level 0 LeafNode
            using ValueType = LeafType::ValueType;

            fmt::print("leaf node type: {}\n", get_type_str<LeafType>());
            fmt::print("lv1 node type: {}\n", get_type_str<Int1Type>());
            fmt::print("lv2 node type: {}\n", get_type_str<Int2Type>());
            fmt::print("root node type: {}\n", get_type_str<RootType>());

            dstGrid->setName("test_zs_ag");
            dstGrid->setGridClass(openvdb::GRID_LEVEL_SET);
            dstGrid->setTransform(openvdb::math::Transform::createLinearTransform(trans));

            /// @note process tree from bottom up
            // build leaf
            auto &l0 = zsag.level(dim_c<0>);

            timer.tick();
            auto nlvs = l0.numBlocks();
            std::vector<LeafType *> lvs(nlvs);
            pol(enumerate(l0.originRange(), lvs),
                [grid = proxy<space>(l0.grid), vms = proxy<space>(l0.valueMask)] ZS_LAMBDA(size_t i, const auto &origin,
                                                                                           LeafType *&pleaf) {
                    pleaf = new LeafType();
                    LeafType &leaf = const_cast<LeafType &>(*pleaf);
                    leaf.setOrigin(openvdb::Coord{origin[0], origin[1], origin[2]});
                    typename LeafType::NodeMaskType vm;
                    std::memcpy(&vm, &vms[i], sizeof(vm));
                    static_assert(sizeof(vm) == sizeof(vms[i]), "???");
                    leaf.setValueMask(vm);

                    auto block = grid.tile(i);
                    static_assert(LeafType::SIZE == RM_CVREF_T(grid)::lane_width, "???");
                    int src = 0;
                    for (ValueType *dst = leaf.buffer().data(), *end = dst + LeafType::SIZE; dst != end;
                         dst += 4, src += 4) {
                        dst[0] = block(0, src);
                        dst[1] = block(0, src + 1);
                        dst[2] = block(0, src + 2);
                        dst[3] = block(0, src + 3);
                    }
                });
            timer.tock(fmt::format("build vdb leaf level ({} blocks) from adaptive grid", nlvs));

            // build level 1
            auto &l1 = zsag.level(dim_c<1>);

            timer.tick();
            auto nInt1s = l1.numBlocks();
            std::vector<Int1Type *> int1s(nInt1s);
            // @note use the child table, not from this level
            pol(enumerate(l1.originRange(), int1s),
                [grid = proxy<space>(l1.grid), cms = proxy<space>(l1.childMask), vms = proxy<space>(l1.valueMask),
                 tb = proxy<space>(l0.table), &lvs] ZS_LAMBDA(size_t i, const auto &origin, Int1Type *&pnode) mutable {
                    pnode = new Int1Type();
                    Int1Type &node = const_cast<Int1Type &>(*pnode);
                    auto bcoord = openvdb::Coord{origin[0], origin[1], origin[2]};
                    node.setOrigin(bcoord);

                    typename Int1Type::NodeMaskType m;
                    static_assert(sizeof(m) == sizeof(vms[i]) && sizeof(m) == sizeof(cms[i]), "???");
                    std::memcpy(&m, &vms[i], sizeof(m));
                    const_cast<typename Int1Type::NodeMaskType &>(node.getValueMask()) = m;
                    // node.setValueMask(m);
                    std::memcpy(&m, &cms[i], sizeof(m));
                    // node.setChildMask(m);
                    const_cast<typename Int1Type::NodeMaskType &>(node.getChildMask()) = m;

                    auto block = grid.tile(i);
                    auto *dstTable = const_cast<typename Int1Type::UnionType *>(node.getTable());

#if 1
                    for (u32 n = 0; n < Int1Type::NUM_VALUES; ++n) {
                        if (m.isOn(n)) {
                            // childNodes.emplace_back(n, srcData->getChild(n));
                            auto childCoord = node.offsetToGlobalCoord(n);
                            auto chNo = tb.query(ZSCoordT{childCoord[0], childCoord[1], childCoord[2]});
#if AG_CHECK
                            if (chNo >= 0 && chNo < lvs.size()) {
#endif
                                typename Int1Type::ChildNodeType *chPtr =
                                    const_cast<typename Int1Type::ChildNodeType *>(lvs[chNo]);
                                static_assert(is_same_v<typename Int1Type::ChildNodeType, LeafType>, "!!!!");

                                dstTable[n].setChild(chPtr);

#if AG_CHECK
                                //fmt::print("found child block {}, {}, {} at {}\n", childCoord[0], childCoord[1],
                                //           childCoord[2], chNo);
                            } else {
                                fmt::print("child block {}, {}, {} not found!\n", childCoord[0], childCoord[1],
                                           childCoord[2]);
                            }
#endif
                        } else {
                            dstTable[n].setValue(block(0, n));
                        }
                    }
#endif
                    static_assert(Int1Type::NUM_VALUES == RM_CVREF_T(grid)::lane_width, "???");
                });
            timer.tock(fmt::format("build vdb level 1 ({} blocks) from adaptive grid", nInt1s));
            // build level 2
            auto &l2 = zsag.level(dim_c<2>);

            timer.tick();
            auto nInt2s = l2.numBlocks();
            std::vector<Int2Type *> int2s(nInt2s);
            // Int2Type *pInt2s = new Int2Type[nInt2s];
            // @note use the child table, not from this level
            pol(enumerate(l2.originRange(), int2s),
                [grid = proxy<space>(l2.grid), cms = proxy<space>(l2.childMask), vms = proxy<space>(l2.valueMask),
                 tb = proxy<space>(l1.table),
                 &int1s] ZS_LAMBDA(size_t i, const auto &origin, Int2Type *&pnode) mutable {
                    pnode = new Int2Type();
                    Int2Type &node = const_cast<Int2Type &>(*pnode);
                    auto bcoord = openvdb::Coord{origin[0], origin[1], origin[2]};
                    node.setOrigin(bcoord);

                    typename Int2Type::NodeMaskType m;
                    static_assert(sizeof(m) == sizeof(vms[i]) && sizeof(m) == sizeof(cms[i]), "???");
                    std::memcpy(&m, &vms[i], sizeof(m));
                    const_cast<typename Int2Type::NodeMaskType &>(node.getValueMask()) = m;
                    // node.setValueMask(m);
                    std::memcpy(&m, &cms[i], sizeof(m));
                    // node.setChildMask(m);
                    const_cast<typename Int2Type::NodeMaskType &>(node.getChildMask()) = m;

                    auto block = grid.tile(i);
                    auto *dstTable = const_cast<typename Int2Type::UnionType *>(node.getTable());

#if 1
                    for (u32 n = 0; n < Int2Type::NUM_VALUES; ++n) {
                        if (m.isOn(n)) {
                            // childNodes.emplace_back(n, srcData->getChild(n));
                            auto childCoord = node.offsetToGlobalCoord(n);
                            auto chNo = tb.query(ZSCoordT{childCoord[0], childCoord[1], childCoord[2]});
#if AG_CHECK
                            if (chNo >= 0 && chNo < int1s.size()) {
#endif
                                typename Int2Type::ChildNodeType *chPtr =
                                    const_cast<typename Int2Type::ChildNodeType *>(int1s[chNo]);
                                static_assert(is_same_v<typename Int2Type::ChildNodeType, Int1Type>, "!!!!");

                                dstTable[n].setChild(chPtr);

#if AG_CHECK
                                //fmt::print("found child block {}, {}, {} at {}\n", childCoord[0], childCoord[1],
                                //           childCoord[2], chNo);
                            } else {
                                fmt::print("child block {}, {}, {} not found!\n", childCoord[0], childCoord[1],
                                           childCoord[2]);
                            }
#endif
                        } else {
                            dstTable[n].setValue(block(0, n));
                        }
                    }
#endif
                    static_assert(Int2Type::NUM_VALUES == RM_CVREF_T(grid)::lane_width, "???");
                });
            timer.tock(fmt::format("build vdb level 2 ({} blocks) from adaptive grid", nInt2s));
            // build root
            auto &root = dstGrid->tree().root();
            for (u32 i = 0; i != nInt2s; ++i) {
                root.addChild(int2s[i]);
            }
        }
        fmt::print("vec3fgrid value_type: {}\n", get_type_str<typename openvdb::Vec3fGrid::ValueType>());
        set_output("vdb", ret);
    }
};

ZENDEFNODE(TestAdaptiveGrid, {
                                 {"sdf"},
                                 {"vdb"},
                                 {},
                                 {"ZPCTest"},
                             });

} // namespace zeno
