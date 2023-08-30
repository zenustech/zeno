//#include "Structures.hpp"
//#include "Utils.hpp"
#include <cassert>
#include <chrono>
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

#include "zensim/execution/ConcurrencyPrimitive.hpp"

namespace zeno {

struct spinlock {
    std::atomic<bool> lock_ = {0};

    void lock() noexcept {
        for (;;) {
            // Optimistically assume the lock is free on the first try
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            // Wait for lock to be released without generating cache misses
            while (lock_.load(std::memory_order_relaxed)) {
                // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
                // hyper-threads
                zs::pause_cpu();
            }
        }
    }

    bool try_lock() noexcept {
        // First do a relaxed load to check if lock is free in order to prevent
        // unnecessary cache misses if someone does while(!try_lock())
        return !lock_.load(std::memory_order_relaxed) && !lock_.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        lock_.store(false, std::memory_order_release);
    }
};

struct ZSConcurrencyTest : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;

#if 1
        constexpr int N = 10000000;
        constexpr int M = 1;
#else
        constexpr int N = 16;
        constexpr int M = 10000000;
#endif
        u64 sum = 0;

        // zs::Vector<int> keys{N};
        auto pol = zs::omp_exec();

        for (int i = 0; i < N; ++i)
            sum += i;
        fmt::print("ref sum is: {}\n", sum);
        /// init
        CppTimer timer;
        sum = 0;
        timer.tick();
        seq_exec()(range(N), [&](int i) {
            for (int d = 0; d != M; ++d)
                atomic_add(exec_seq, &sum, (u64)i);
        });
        timer.tock("ref (serial)");
        fmt::print("serial sum is: {}\n", sum);

        sum = 0;
        timer.tick();
        pol(range(N), [&](int i) {
            for (int d = 0; d != M; ++d)
                atomic_add(exec_omp, &sum, (u64)i);
        });
        timer.tock("ref (atomic)");
        fmt::print("atomic sum is: {}\n", sum);

        sum = 0;
        timer.tick();
        std::mutex mtx;
        pol(range(N), [&](int i) {
            std::lock_guard<std::mutex> lk(mtx);
            for (int d = 0; d != M; ++d)
                sum += i;
        });
        timer.tock("ref (std mutex)");
        fmt::print("std mutex sum is: {}\n", sum);

#if 0
        sum = 0;
        timer.tick();
        {
            spinlock mtx;
            pol(range(N), [&](int i) {
                mtx.lock();
                for (int d = 0; d != M; ++d)
                    sum += i;
                mtx.unlock();
                });
        }
        timer.tock("ref (spinlock)");
        fmt::print("spinlock sum is: {}\n", sum);
#endif
        sum = 0;
        timer.tick();
        {
            std::mutex mtx;
            pol(range(N), [&](int i) {
                while (!mtx.try_lock())
                    ;
                for (int d = 0; d != M; ++d)
                    sum += i;
                mtx.unlock();
            });
        }
        timer.tock("ref (std mutex trylock)");
        fmt::print("std mutex trylock sum is: {}\n", sum);

        sum = 0;
        timer.tick();
        {
            Mutex mtx;
            pol(range(N), [&](int i) {
                while (!mtx.try_lock())
                    ;
                for (int d = 0; d != M; ++d)
                    sum += i;
                mtx.unlock();
            });
        }
        timer.tock("ref (fast mutex trylock)");
        fmt::print("fast mutex trylock sum is: {}\n", sum);

        sum = 0;
        timer.tick();
        {
            Mutex mtx;
            pol(range(N), [&](int i) {
                mtx.lock();
                for (int d = 0; d != M; ++d)
                    sum += i;
                mtx.unlock();
            });
        }
        timer.tock("ref (fast mutex)");
        fmt::print("fast mutex sum is: {}\n", sum);

        timer.tick();
        {
            std::mutex mtx;
            seq_exec()(range(N), [&](int i) {
                mtx.lock();
                mtx.unlock();
            });
        }
        timer.tock("purely std lock/unlock");
        timer.tick();
        {
            Mutex mtx;
            seq_exec()(range(N), [&](int i) {
                mtx.lock();
                mtx.unlock();
            });
        }
        timer.tock("purely custom lock/unlock");

        timer.tick();
        {
            std::mutex mtx;
            seq_exec()(range(N), [&](int i) { mtx.try_lock(); });
        }
        timer.tock("purely std try lock (mostly fail)");
        timer.tick();
        {
            Mutex mtx;
            seq_exec()(range(N), [&](int i) { mtx.try_lock(); });
        }
        timer.tock("purely custom try lock (mostly fail)");
        timer.tick();
        {
            std::mutex mtx;
            seq_exec()(range(N), [&](int i) {
                if (mtx.try_lock())
                    mtx.unlock();
            });
        }
        timer.tock("purely std try lock");
        timer.tick();
        {
            Mutex mtx;
            seq_exec()(range(N), [&](int i) {
                if (mtx.try_lock())
                    mtx.unlock();
            });
        }
        timer.tock("purely custom try lock");
    }
};

ZENDEFNODE(ZSConcurrencyTest, {
                                  {},
                                  {},
                                  {},
                                  {"ZPCTest"},
                              });

#if defined(ZS_PLATFORM_LINUX)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(ZS_PLATFORM_WINDOWS)
#include <fileapi.h>
#endif

struct ZSFileTest : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto fn = get_input2<std::string>("file");
#if 0
        auto vallocator = get_virtual_memory_source(memsrc_e::host, -1, (size_t)1 << (size_t)23, "STACK");

        vallocator.commit(0, sizeof(double) * 40);
        auto ptr = (double*)vallocator.address(0);
        for (int i = 0; i < 10; ++i) {
            *(ptr + i) = i;
        }
        pol(range(10), [&](int i) { fmt::print("tid[{}]: {}\n", i, *(ptr + i)); });

        CppTimer timer;
        timer.tick();
        timer.tock("...");
#endif

#if defined(ZS_PLATFORM_LINUX)
        // ref: https://man7.org/linux/man-pages/man2/open.2.html
        // ref: https://www.man7.org/linux/man-pages/man2/mmap.2.html
        int fd = open(fn.data(), /* int flag */ O_RDWR, /* mode_t mode */ S_IRUSR | S_IWUSR);
        struct stat st;
        char* addr;
        if (fstat(fd, &st) == -1) {
            addr = nullptr;
            fmt::print("unable to query file size.\n");
        }
        else {
            addr = (char*)mmap(NULL, st.st_size, /* int prot */ PROT_READ | PROT_WRITE, /* int flags */ MAP_SHARED,
                /* int file_handle */ fd, /* offset */ 0);
        }
        /// ...
        fmt::print("====begin====\n");
        for (int i = 0; i < 20 && i < st.st_size; ++i) {
            fmt::print("{}", addr[i]);
            addr[i] = ::toupper(addr[i]);
        }
        fmt::print("\n====done====");

        munmap(addr, st.st_size);
        close(fd);
#elif defined(ZS_PLATFORM_WINDOWS)
        fmt::print("error code 0: {}\n", GetLastError());
        HANDLE fd = CreateFileA(fn.data(), /*dwDesiredAccess*/GENERIC_READ | GENERIC_WRITE, /*dwShareMode*/FILE_SHARE_READ | FILE_SHARE_WRITE, /*lpSecurityAttributes*/NULL, /*dwCreationDisposition*/OPEN_EXISTING, /*dwFlagsAndAttributes*/FILE_ATTRIBUTE_NORMAL, /*hTemplateFile*/NULL);
        fmt::print("error code after createfile: {}\n", GetLastError());
        HANDLE fm;
        size_t sz = 0;
        char* addr = nullptr;
        if (fd != INVALID_HANDLE_VALUE) {
            sz = GetFileSize(fd, NULL);
            fmt::print("file size: {}\n", sz);
            fm = CreateFileMappingA(fd, NULL, /*flProtect*/PAGE_READWRITE, 0, 0, /* lpName */NULL);
            fmt::print("error code after filemapping: {}\n", GetLastError());
            if (fm != INVALID_HANDLE_VALUE) {
                addr = (char*)MapViewOfFile(fm, FILE_MAP_ALL_ACCESS, 0, 0, 0);
                fmt::print("error code after view: {}\n", GetLastError());
                if (addr == nullptr) {
                    fmt::print("wtf??");
                    fmt::print("error code: {}\n", GetLastError());
                }
            }
            else {
                fmt::print("failed to create file mapping\n");
            }
        }
        else {
            fmt::print("failed to open file\n");
        }
        fmt::print("====begin====\n");
        for (int i = 0; addr != nullptr && i < 20 && i < sz; ++i) {
            fmt::print("{}", addr[i]);
            addr[i] = ::toupper(addr[i]);
        }
        fmt::print("\n====done====\n");

        UnmapViewOfFile(addr);
        CloseHandle(fm);
        CloseHandle(fd);
#endif
        // fn;
    }
};

ZENDEFNODE(ZSFileTest, {
                           {{"string", "file", ""}},
                           {},
                           {},
                           {"ZPCTest"},
                       });

struct ZSLoopTest : INode {

    using clock = std::chrono::high_resolution_clock;
    using ns = std::chrono::nanoseconds;
    using us = std::chrono::microseconds;
    using ms = std::chrono::milliseconds;
    template <typename TimeUnit, typename Duration>
    double timeCast(const Duration &interval) {
        return std::chrono::duration_cast<TimeUnit>(interval).count() * 1.;
    }

    void updateUI(double tSlice) {
        ;
    }
    void draw() {
        ;
    }

    void stepSimulation(double tSlice) {
        ;
    }

    std::atomic<zs::u64> workerCounter;

    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        // std::chrono::high_resolution_clock::now();
        // std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - m_Start).count()
        constexpr float t_replay_slice = 1.f / 60; // 60 fps
        bool running = true;
        double tAccum = 0.;
        clock::time_point tNow, tLast = clock::now();
        clock::time_point origin = tLast;
        u64 sum = 0;
        bool show = false;

        workerCounter.store(0);
        std::thread computeWorker([this]() {
            using namespace std::chrono_literals;
            // asynchronously write contents to files
            while (workerCounter.fetch_add(1) < 10)
                std::this_thread::sleep_for(2s);
        });

        /// high-frequency loop
        while (running) {
            tNow = clock::now();
            double dt = timeCast<ns>(tNow - tLast) * (0.001 * 0.001 * 0.001);
            tLast = tNow;
            tAccum += dt;

            /// check msg queue (optional, active when DOP is engaging)
            /// rolling windows (ring buffer) for caching frames, preload enough frames
            /// virtual mem caching
            while (tAccum > t_replay_slice) {
                updateUI(t_replay_slice);
                tAccum -= t_replay_slice;

                sum++;
                show = true;
            }
            draw();

            if (sum % 100 == 0 && show) {
                fmt::print("[{}] slices elapsed {}. worker iter id {}\n", sum,
                           timeCast<ns>(tNow - origin) * (0.001 * 0.001 * 0.001),
                           workerCounter.load(std::memory_order_acquire));
                show = false;
            }
            if (sum > 1000)
                break;
        }

        computeWorker.join();
    }
};

ZENDEFNODE(ZSLoopTest, {
                           {},
                           {},
                           {},
                           {"ZPCTest"},
                       });

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
        using ZSGridT = zs::VdbGrid<3, ValueT,
                                    zs::index_sequence<RootT::NodeChainType::template Get<0>::LOG2DIM,
                                                       RootT::NodeChainType::template Get<1>::LOG2DIM,
                                                       RootT::NodeChainType::template Get<2>::LOG2DIM>>;
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

            static_assert(sizeof(typename NodeT::NodeMaskType) == sizeof(typename LevelT::tile_mask_type::value_type),
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

            static_assert(sizeof(typename LeafT::NodeMaskType) == sizeof(typename LevelT::tile_mask_type::value_type),
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
        CppTimer timer;
#if 0
        using Adapter = openvdb::TreeAdapter<openvdb::FloatGrid>;
        using TreeT = typename Adapter::TreeType;
        auto &tree = Adapter::tree(*sdf);
        // fmt::print("TreeT: {}, tree: {}\n", get_type_str<TreeT>(), get_var_type_str(tree));
        static_assert(is_same_v<TreeT, RM_CVREF_T(tree)>, "???");
        fmt::print("root: {}\n", get_var_type_str(tree.root()));

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
#endif

        auto pol = omp_exec();
        /// construct new vdb grid from ag
        /// ref: nanovdb/util/NanoToOpenVDB.h
        // auto zsag = agBuilder.get();
        auto zsag = convert_floatgrid_to_adaptive_grid(sdf, MemoryHandle{memsrc_e::host, -1}, "sdf");
        zsag.reorder(pol);

        AdaptiveTileTree<3, zs::f32, 3, 2> att;
        // zsag.restructure(pol, att);
        // att.restructure(pol, zsag);

        // test ag view
        using ZSCoordT = zs::vec<int, 3>;
        auto zsagv = view<space>(zsag);
        auto zsacc = zsagv.getAccessor();
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
            (void)(zsagv.probeValue(0, c, v /*, false_c*/));
        }
        timer.tock("naive query (zs probe)");
        timer.tick();
        for (auto &c : coords) {
            (void)(zsagv.probeValue(0, c, v, true_c));
        }
        timer.tock("ordered query (zs probe)");
        timer.tick();
        for (auto &c : coords) {
            (void)(zsacc.probeValue(0, c, v));
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

        openvdb::FloatGrid::ConstAccessor accessor = sdf->getConstAccessor();
        openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler> fastSampler(
            accessor, sdf->transform());
        timer.tick();
        for (auto &c : coords) {
            auto cc = c.cast<f32>() / 3;
            (void)(fastSampler.isSample(openvdb::Vec3R(cc[0], cc[1], cc[2])));
        }
        timer.tock("query (fast vdb sample)");

        timer.tick();
        for (auto &c : coords) {
            auto cc = c.cast<f32>() / 3;
            (void)(zsagv.iSample(0, cc));
        }
        timer.tock("query (zs sample)");

        timer.tick();
        for (auto &c : coords) {
            auto cc = c.cast<f32>() / 3;
            (void)(zsagv.iSample(zsacc, 0, cc));
        }
        timer.tock("further optimized query (zs sample)");

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
