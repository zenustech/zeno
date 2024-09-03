#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/VDBGrid.h>

#include "../scheme.hpp"
#include "../utils.cuh"
#include "zeno/utils/log.h"

namespace zeno {

struct ZSExtendSparseGrid : INode {

    void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto nlayers = get_input2<int>("layers");

        using namespace zs;
        static constexpr auto space = execspace_e::cuda;
        namespace cg = ::cooperative_groups;
        auto pol = cuda_exec();
        auto &spg = zsSPG->getSparseGrid();

        auto nbs = spg.numBlocks();
        auto newNbs = nbs;
        // auto &table = spg._table;
        int nbsOffset = 0;
        /// @note iteratively resize table
        while (nlayers-- > 0) {
            spg.resize(pol, newNbs * 27 + nbsOffset, false);
            pol(range(newNbs * spg._table.bucket_size),
                [spg = proxy<space>(spg), nbsOffset] __device__(std::size_t i) mutable {
                    auto tile = cg::tiled_partition<RM_CVREF_T(spg._table)::bucket_size>(cg::this_thread_block());
                    auto bno = i / spg._table.bucket_size + nbsOffset;
                    auto bcoord = spg.iCoord(bno, 0);
                    constexpr auto failure_token_v = RM_CVREF_T(spg._table)::failure_token_v;
                    for (auto loc : ndrange<3>(3)) {
                        auto dir = make_vec<int>(loc) - 1;
                        if (dir[0] == 0 && dir[1] == 0 && dir[2] == 0)
                            continue;
                        // spg._table.insert(bcoord + dir * spg.side_length);
                        if (spg._table.tile_insert(tile, bcoord + dir * spg.side_length,
                                                   RM_CVREF_T(spg._table)::sentinel_v, true) == failure_token_v)
                            *spg._table._success = false;
                    }
                });

            if (int tag = spg._table._buildSuccess.getVal(); tag == 0)
                zeno::log_error("check build success state: {}\n", tag);

            nbsOffset += newNbs;
            newNbs = spg.numBlocks() - nbsOffset;
        }

        /// @note resize grid
        nbsOffset += newNbs; // current total blocks
        spg.resizeGrid(nbsOffset);
        newNbs = nbsOffset - nbs;

        if (get_input2<bool>("fillByBackground")) {
            pol(zs::Collapse{newNbs, spg.block_size},
                [spgv = proxy<space>(spg), nbs, bg = spg._background,
                 nchns = spg.numChannels()] __device__(int blockno, int cellno) mutable {
                    int m_bno = blockno + nbs;
                    for (int ch = 0; ch < nchns; ++ch) {
                        spgv(ch, m_bno, cellno) = bg;
                    }
                });
        } else {
            zs::memset(mem_device, (void *)spg._grid.tileOffset(nbs), 0, (std::size_t)newNbs * spg._grid.tileBytes());
        }

        bool includeMultigrid = get_input2<bool>("multigrid");
        if (includeMultigrid) {
            /// @brief adjust multigrid accordingly
            // grid
            nbs += newNbs;
            auto &spg1 = zsSPG->spg1;
            spg1.resize(pol, nbs);
            auto &spg2 = zsSPG->spg2;
            spg2.resize(pol, nbs);
            auto &spg3 = zsSPG->spg3;
            spg3.resize(pol, nbs);
            // table
            {
                const auto &table = spg._table;
                auto &table1 = spg1._table;
                auto &table2 = spg2._table;
                auto &table3 = spg3._table;
                table1.reset(true);
                table1._cnt.setVal(nbs);
                table2.reset(true);
                table2._cnt.setVal(nbs);
                table3.reset(true);
                table3._cnt.setVal(nbs);

                table1._buildSuccess.setVal(1);
                table2._buildSuccess.setVal(1);
                table3._buildSuccess.setVal(1);
                pol(range(nbs), [table = proxy<space>(table), tab1 = proxy<space>(table1), tab2 = proxy<space>(table2),
                                 tab3 = proxy<space>(table3)] __device__(std::size_t i) mutable {
                    auto bcoord = table._activeKeys[i];
                    tab1.insert(bcoord / 2, i, true);
                    tab2.insert(bcoord / 4, i, true);
                    tab3.insert(bcoord / 8, i, true);
                });

                int tag1 = table1._buildSuccess.getVal();
                int tag2 = table2._buildSuccess.getVal();
                int tag3 = table3._buildSuccess.getVal();
                if (tag1 == 0 || tag2 == 0 || tag3 == 0)
                    zeno::log_error("check multigrid build success state: {}, {}, {}\n", tag1, tag2, tag3);
            }
        }

        set_output("SparseGrid", zsSPG);
    }
};

ZENDEFNODE(ZSExtendSparseGrid,
           {/* inputs: */
            {"SparseGrid", {"int", "layers", "1"}, {"bool", "multigrid", "0"}, {"bool", "fillByBackground", "0"}},
            /* outputs: */
            {"SparseGrid"},
            /* params: */
            {},
            /* category: */
            {"Eulerian"}});

struct ZSMaintainSparseGrid : INode {
    template <typename PredT>
    void maintain(ZenoSparseGrid *zsgridPtr, zs::SmallString tag, PredT pred, int nlayers) {
        using namespace zs;
        static constexpr auto space = execspace_e::cuda;
        namespace cg = ::cooperative_groups;
        auto pol = cuda_exec();
        auto &spg = zsgridPtr->getSparseGrid();

        if (!spg._grid.hasProperty(tag))
            throw std::runtime_error(fmt::format("property [{}] not exist!", tag));

        auto nbs = spg.numBlocks();
        using Ti = RM_CVREF_T(nbs);

        Vector<Ti> marks{spg.get_allocator(), nbs + 1}, offsets{spg.get_allocator(), nbs + 1};
        marks.reset(0);

        static_assert(RM_CVREF_T(spg)::block_size % 32 == 0, "block size should be a multiple of 32.");

        /// @brief mark active block entries
        pol(range(nbs * 32), [spg = proxy<space>(spg), tagOffset = spg.getPropertyOffset(tag),
                              marks = proxy<space>(marks), pred] __device__(std::size_t i) mutable {
            auto tile = cg::tiled_partition<32>(cg::this_thread_block());
            auto bno = i / 32;
            auto cellno = tile.thread_rank();

            while (cellno < spg.block_size) {
                if (tile.ballot(pred(spg(tagOffset, bno, cellno))))
                    break;
                cellno += 32;
            }
            if (tile.thread_rank() == 0 && cellno < spg.block_size)
                marks[bno] = 1;
        });

        exclusive_scan(pol, std::begin(marks), std::end(marks), std::begin(offsets));
        auto newNbs = offsets.getVal(nbs);
        auto numMarked = newNbs;
        fmt::print("compacting {} blocks to {} active blocks.\n", nbs, newNbs);

        /// @brief compact active block entries
        // table
        auto &table = spg._table;
        table.reset(false);
        table._cnt.setVal(newNbs);
        // backup previous block entries, nbs is the previous count of blocks
        auto prevKeys = table._activeKeys;
        auto &keys = table._activeKeys;

        pol(range(nbs * spg._table.bucket_size),
            [marks = proxy<space>(marks), newKeys = proxy<space>(keys), keys = proxy<space>(prevKeys),
             table = proxy<space>(table), offsets = proxy<space>(offsets),
             bs_c = wrapv<RM_CVREF_T(spg)::block_size>{}] __device__(std::size_t i) mutable {
                constexpr auto block_size = decltype(bs_c)::value;
                constexpr auto bucket_size = RM_CVREF_T(table)::bucket_size;
                static_assert(block_size % bucket_size == 0, "block_size should be a multiple of bucket_size");
                auto tile = cg::tiled_partition<bucket_size>(cg::this_thread_block());
                auto bno = i / bucket_size;
                if (marks[bno] == 0)
                    return;
                auto dstBno = offsets[bno];
                // table
                auto bcoord = keys[bno];
                table.tile_insert(tile, bcoord, dstBno, false); // do not enqueue key, hence set false
                if (tile.thread_rank() == 0)
                    newKeys[dstBno] = bcoord;
            });

        // grid
        /// @note backup the grid ahead
        auto &grid = spg._grid;
        auto prevGrid = grid;

        /// @brief iteratively expand the active domain
        Ti nbsOffset = 0;
        while (nlayers-- > 0 && newNbs > 0) {
            // reserve enough memory for expanded grid and table
            spg.resize(pol, newNbs * 7 + nbsOffset, false);
            // extend one layer
            pol(range(newNbs * spg._table.bucket_size),
                [spg = proxy<space>(spg), nbsOffset] __device__(std::size_t i) mutable {
                    auto tile = cg::tiled_partition<RM_CVREF_T(spg._table)::bucket_size>(cg::this_thread_block());
                    auto bno = i / spg._table.bucket_size + nbsOffset;
                    auto bcoord = spg.iCoord(bno, 0);
                    constexpr auto failure_token_v = RM_CVREF_T(spg._table)::failure_token_v;
                    {
                        for (int d = 0; d != 3; ++d) {
                            auto dir = zs::vec<int, 3>::zeros();
                            dir[d] = -1;
                            if (spg._table.tile_insert(tile, bcoord + dir * spg.side_length,
                                                       RM_CVREF_T(spg._table)::sentinel_v, true) == failure_token_v)
                                *spg._table._success = false;
                            dir[d] = 1;
                            if (spg._table.tile_insert(tile, bcoord + dir * spg.side_length,
                                                       RM_CVREF_T(spg._table)::sentinel_v, true) == failure_token_v)
                                *spg._table._success = false;
                        }
                    }
#if 0
                    for (auto loc : ndrange<3>(3)) {
                        auto dir = make_vec<int>(loc) - 1;
                        // spg._table.insert(bcoord + dir * spg.side_length);
                        spg._table.tile_insert(tile, bcoord + dir * spg.side_length, RM_CVREF_T(spg._table)::sentinel_v,
                                               true);
                    }
#endif
                });
            if (int tag = spg._table._buildSuccess.getVal(); tag == 0)
                zeno::log_error("check build success state: {}\n", tag);

            // slide the window
            nbsOffset += newNbs;
            newNbs = spg.numBlocks() - nbsOffset;
        }

        nbsOffset += newNbs; // current total blocks
        /// @note initialize newly inserted grid blocks
        if (nbsOffset > numMarked) {
            spg.resizeGrid(nbsOffset);
            newNbs = nbsOffset - numMarked;
            zs::memset(mem_device, (void *)spg._grid.tileOffset(numMarked), 0,
                       (std::size_t)newNbs * spg._grid.tileBytes());

            if (tag == "sdf") {
                // special treatment for "sdf" property
                pol(range(newNbs * spg.block_size),
                    [dx = spg.voxelSize()[0], spg = proxy<space>(spg), sdfOffset = spg.getPropertyOffset("sdf"),
                     blockOffset = numMarked * spg.block_size] __device__(std::size_t cellno) mutable {
                        spg(sdfOffset, blockOffset + cellno) = 3 * dx;
                    });
            }
        }

        /// @brief relocate original grid data to the new sparse grid
        pol(range(nbs * spg._table.bucket_size), [grid = proxy<space>(prevGrid), spg = proxy<space>(spg),
                                                  keys = proxy<space>(prevKeys)] __device__(std::size_t i) mutable {
            constexpr auto bucket_size = RM_CVREF_T(spg._table)::bucket_size;
            auto tile = cg::tiled_partition<bucket_size>(cg::this_thread_block());
            auto bno = i / bucket_size;
            auto bcoord = keys[bno];
            auto dstBno = spg._table.tile_query(tile, bcoord);
            // auto dstBno = spg._table.query(bcoord);
            if (dstBno == spg._table.sentinel_v)
                return;
            // table
            for (auto cellno = tile.thread_rank(); cellno < spg.block_size; cellno += bucket_size) {
                for (int chn = 0; chn != grid.numChannels(); ++chn)
                    spg._grid(chn, dstBno, cellno) = grid(chn, bno, cellno);
            }
        });

        if (get_input2<bool>("multigrid")) {
            /// @brief adjust multigrid accordingly
            // grid
            nbs = spg.numBlocks();
            auto &spg1 = zsgridPtr->spg1;
            spg1.resize(pol, nbs);
            auto &spg2 = zsgridPtr->spg2;
            spg2.resize(pol, nbs);
            auto &spg3 = zsgridPtr->spg3;
            spg3.resize(pol, nbs);
            // table
            {
                const auto &table = spg._table;
                auto &table1 = spg1._table;
                auto &table2 = spg2._table;
                auto &table3 = spg3._table;
                table1.reset(true);
                table1._cnt.setVal(nbs);
                table2.reset(true);
                table2._cnt.setVal(nbs);
                table3.reset(true);
                table3._cnt.setVal(nbs);

                table1._buildSuccess.setVal(1);
                table2._buildSuccess.setVal(1);
                table3._buildSuccess.setVal(1);
                pol(range(nbs), [table = proxy<space>(table), tab1 = proxy<space>(table1), tab2 = proxy<space>(table2),
                                 tab3 = proxy<space>(table3)] __device__(std::size_t i) mutable {
                    auto bcoord = table._activeKeys[i];
                    tab1.insert(bcoord / 2, i, true);
                    tab2.insert(bcoord / 4, i, true);
                    tab3.insert(bcoord / 8, i, true);
                });

                int tag1 = table1._buildSuccess.getVal();
                int tag2 = table2._buildSuccess.getVal();
                int tag3 = table3._buildSuccess.getVal();
                if (tag1 == 0 || tag2 == 0 || tag3 == 0)
                    zeno::log_error("check multigrid build success state: {}, {}, {}\n", tag1, tag2, tag3);
            }
        }
    }

    void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto tag = get_input2<std::string>("Attribute");
        auto nlayers = get_input2<int>("layers");
        auto needRefit = get_input2<bool>("refit");

        int opt = 0;
        if (needRefit) {
            if (tag == "rho")
                opt = 1;
            else if (tag == "sdf")
                opt = 2;
        }

        if (opt == 0)
            maintain(
                zsSPG.get(), src_tag(zsSPG, tag), [] __device__(float v) { return true; }, nlayers);
        else if (opt == 1)
            maintain(
                zsSPG.get(), src_tag(zsSPG, tag),
                [] __device__(float v) -> bool { return v > zs::detail::deduce_numeric_epsilon<float>() * 10; }, nlayers);
        else if (opt == 2)
            maintain(
                zsSPG.get(), src_tag(zsSPG, tag),
                [dx = zsSPG->getSparseGrid().voxelSize()[0]] __device__(float v) -> bool { return v < 2 * dx; },
                nlayers);

        set_output("SparseGrid", zsSPG);
    }
};

ZENDEFNODE(ZSMaintainSparseGrid, {/* inputs: */
                                  {"SparseGrid",
                                   {"enum rho sdf", "Attribute", "rho"},
                                   {"bool", "refit", "1"},
                                   {"int", "layers", "2"},
                                   {"bool", "multigrid", "1"}},
                                  /* outputs: */
                                  {"SparseGrid"},
                                  /* params: */
                                  {},
                                  /* category: */
                                  {"Eulerian"}});

} // namespace zeno