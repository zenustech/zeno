#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/VDBGrid.h>

#include "../utils.cuh"
#include "zeno/utils/log.h"

namespace zeno {

struct ZSMakeSparseGrid : INode {
    void apply() override {
        auto attr = get_input2<std::string>("Attribute");
        auto dx = get_input2<float>("Dx");
        auto bg = get_input2<float>("background");
        auto type = get_input2<std::string>("type");
        auto structure = get_input2<std::string>("structure");

        auto zsSPG = std::make_shared<ZenoSparseGrid>();
        auto &spg = zsSPG->spg;

        int nc = 1;
        if (type == "scalar")
            nc = 1;
        else if (type == "vector3")
            nc = 3;

        spg = ZenoSparseGrid::spg_t{{{attr, nc}}, 0, zs::memsrc_e::device};
        spg.scale(dx);
        spg._background = bg;

        if (structure == "vertex-centered") {
            auto trans = zs::vec<float, 3>::uniform(-dx / 2);
            // zs::vec<float, 3> trans{-dx / 2.f, -dx / 2.f, -dx / 2.f};

            spg.translate(trans);
        }

        set_output("Grid", zsSPG);
    }
};

ZENDEFNODE(ZSMakeSparseGrid, {/* inputs: */
                              {{"string", "Attribute", ""},
                               {"float", "Dx", "1.0"},
                               {"float", "background", "0"},
                               {"enum scalar vector3", "type", "scalar"},
                               {"enum cell-centered vertex-centered", "structure", "cell-centered"}},
                              /* outputs: */
                              {"Grid"},
                              /* params: */
                              {},
                              /* category: */
                              {"Eulerian"}});

struct ZSGridTopoCopy : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("Grid");
        auto zs_topo = get_input<ZenoSparseGrid>("TopologyGrid");

        auto &grid = zs_grid->spg;
        auto &topo = zs_topo->spg;

        // topo copy
        grid._table = topo._table;
        grid._transform = topo._transform;
        grid._grid.resize(topo.numBlocks() * topo.block_size);

        if (get_input2<bool>("multigrid")) {
            auto pol = zs::cuda_exec();
            constexpr auto space = zs::execspace_e::cuda;

            size_t curNumBlocks = grid.numBlocks();

            /// @brief adjust multigrid accordingly
            // grid
            auto &spg1 = zs_grid->spg1;
            spg1.resize(pol, curNumBlocks);
            auto &spg2 = zs_grid->spg2;
            spg2.resize(pol, curNumBlocks);
            auto &spg3 = zs_grid->spg3;
            spg3.resize(pol, curNumBlocks);
            // table
            {
                const auto &table = grid._table;
                auto &table1 = spg1._table;
                auto &table2 = spg2._table;
                auto &table3 = spg3._table;
                table1.reset(true);
                table1._cnt.setVal(curNumBlocks);
                table2.reset(true);
                table2._cnt.setVal(curNumBlocks);
                table3.reset(true);
                table3._cnt.setVal(curNumBlocks);

                table1._buildSuccess.setVal(1);
                table2._buildSuccess.setVal(1);
                table3._buildSuccess.setVal(1);
                pol(zs::range(curNumBlocks),
                    [table = zs::proxy<space>(table), tab1 = zs::proxy<space>(table1), tab2 = zs::proxy<space>(table2),
                     tab3 = zs::proxy<space>(table3)] __device__(std::size_t i) mutable {
                        auto bcoord = table._activeKeys[i];
                        tab1.insert(bcoord / 2, i, true);
                        tab2.insert(bcoord / 4, i, true);
                        tab3.insert(bcoord / 8, i, true);
                    });

                int tag1 = table1._buildSuccess.getVal();
                int tag2 = table2._buildSuccess.getVal();
                int tag3 = table3._buildSuccess.getVal();
                if (tag1 == 0 || tag2 == 0 || tag3 == 0)
                    zeno::log_error("check TopoCopy multigrid activate success state: {}, {}, {}\n", tag1, tag2, tag3);
            }
        }

        set_output("Grid", zs_grid);
    }
};

ZENDEFNODE(ZSGridTopoCopy, {/* inputs: */
                            {"Grid", "TopologyGrid", {"bool", "multigrid", "0"}},
                            /* outputs: */
                            {"Grid"},
                            /* params: */
                            {},
                            /* category: */
                            {"Eulerian"}});

struct ZSSparseGridToVDB : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("SparseGrid");
        auto attr = get_input2<std::string>("Attribute");
        auto VDBGridClass = get_input2<std::string>("VDBGridClass");
        auto VDBGridName = get_input2<std::string>("VDBGridName");

        if (attr.empty())
            attr = "sdf";

        auto &spg = zs_grid->spg;

        auto attrTag = src_tag(zs_grid, attr);
        auto num_ch = spg.getPropertySize(attrTag);
        if (VDBGridClass == "STAGGERED" && num_ch != 3) {
            throw std::runtime_error("The size of Attribute is not 3!");
        }

        if (num_ch == 3) {
            auto vdb_ = zs::convert_sparse_grid_to_float3grid(spg, attrTag, VDBGridName);
            auto vdb_grid = std::make_shared<VDBFloat3Grid>();
            vdb_grid->m_grid = vdb_.as<openvdb::Vec3fGrid::Ptr>();

            set_output("VDB", vdb_grid);
        } else {
            zs::u32 gridClass = 0;
            if (VDBGridClass == "UNKNOWN")
                gridClass = 0;
            else if (VDBGridClass == "LEVEL_SET")
                gridClass = 1;
            else if (VDBGridClass == "FOG_VOLUME")
                gridClass = 2;

            auto vdb_ = zs::convert_sparse_grid_to_floatgrid(spg, attrTag, gridClass, VDBGridName);

            auto vdb_grid = std::make_shared<VDBFloatGrid>();
            vdb_grid->m_grid = vdb_.as<openvdb::FloatGrid::Ptr>();

            set_output("VDB", vdb_grid);
        }
    }
};

ZENDEFNODE(ZSSparseGridToVDB, {/* inputs: */
                               {"SparseGrid",
                                {"string", "Attribute", ""},
                                {"enum UNKNOWN LEVEL_SET FOG_VOLUME STAGGERED", "VDBGridClass", "LEVEL_SET"},
                                {"string", "VDBGridName", "SparseGrid"}},
                               /* outputs: */
                               {"VDB"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSVDBToSparseGrid : INode {
    void apply() override {
        auto vdb = get_input<VDBGrid>("VDB");
        auto attr = get_input2<std::string>("Attribute");
        if (attr.empty())
            attr = "sdf";

        if (has_input("SparseGrid")) {
            auto zs_grid = get_input<ZenoSparseGrid>("SparseGrid");
            auto &spg = zs_grid->spg;

            int num_ch;
            if (vdb->getType() == "FloatGrid")
                num_ch = 1;
            else if (vdb->getType() == "Vec3fGrid")
                num_ch = 3;
            else
                throw std::runtime_error("Input VDB must be a FloatGrid or Vec3fGrid!");

            auto attrTag = src_tag(zs_grid, attr);
            if (spg.hasProperty(attrTag)) {
                if (num_ch != spg.getPropertySize(attrTag)) {
                    throw std::runtime_error(fmt::format("The channel number of [{}] doesn't match!", attr));
                }
            } else {
                spg.append_channels(zs::cuda_exec(), {{attrTag, num_ch}});
            }

            if (num_ch == 1) {
                auto vdb_ = std::dynamic_pointer_cast<VDBFloatGrid>(vdb);
                zs::assign_floatgrid_to_sparse_grid(vdb_->m_grid, spg, attrTag);
            } else {
                auto vdb_ = std::dynamic_pointer_cast<VDBFloat3Grid>(vdb);
                zs::assign_float3grid_to_sparse_grid(vdb_->m_grid, spg, attrTag);
            }

            set_output("SparseGrid", zs_grid);
        } else {
            ZenoSparseGrid::spg_t spg;

            auto vdbType = vdb->getType();
            if (vdbType == "FloatGrid") {
                auto vdb_ = std::dynamic_pointer_cast<VDBFloatGrid>(vdb);
                spg =
                    zs::convert_floatgrid_to_sparse_grid(vdb_->m_grid, zs::MemoryProperty{zs::memsrc_e::device, -1}, attr);
            } else if (vdbType == "Vec3fGrid") {
                auto vdb_ = std::dynamic_pointer_cast<VDBFloat3Grid>(vdb);
                spg = zs::convert_float3grid_to_sparse_grid(vdb_->m_grid, zs::MemoryProperty{zs::memsrc_e::device, -1},
                                                            attr);
            } else {
                throw std::runtime_error("Input VDB must be a FloatGrid or Vec3fGrid!");
            }

            auto zsSPG = std::make_shared<ZenoSparseGrid>();
            zsSPG->spg = std::move(spg);

            set_output("SparseGrid", zsSPG);
        }
    }
};

ZENDEFNODE(ZSVDBToSparseGrid, {/* inputs: */
                               {"VDB", "SparseGrid", {"string", "Attribute", ""}},
                               /* outputs: */
                               {"SparseGrid"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSGridVoxelSize : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("SparseGrid");

        float dx = zs_grid->getSparseGrid().voxelSize()[0];

        set_output("dx", std::make_shared<NumericObject>(dx));
    }
};

ZENDEFNODE(ZSGridVoxelSize, {/* inputs: */
                             {"SparseGrid"},
                             /* outputs: */
                             {"dx"},
                             /* params: */
                             {},
                             /* category: */
                             {"Eulerian"}});

struct ZSGridAppendAttribute : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("SparseGrid");
        auto attrTag = get_input2<std::string>("Attribute");
        auto nchns = get_input2<int>("ChannelNumber");

        auto &spg = zs_grid->spg;
        auto pol = zs::cuda_exec();

        if (!spg.hasProperty(attrTag)) {
            spg.append_channels(pol, {{attrTag, nchns}});
        } else {
            int m_nchns = spg.getPropertySize(attrTag);
            if (m_nchns != nchns)
                throw std::runtime_error(
                    fmt::format("the SparseGrid already has [{}] with [{}] channels!", attrTag, m_nchns));
        }

        set_output("SparseGrid", zs_grid);
    }
};

ZENDEFNODE(ZSGridAppendAttribute, {/* inputs: */
                                   {"SparseGrid", {"string", "Attribute", ""}, {"int", "ChannelNumber", "1"}},
                                   /* outputs: */
                                   {"SparseGrid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

struct ZSMultiGridAppendAttribute : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("SparseGrid");
        auto attrTag = get_input2<std::string>("Attribute");
        auto nchns = get_input2<int>("ChannelNumber");

        auto &spg1 = zs_grid->spg1;
        auto &spg2 = zs_grid->spg2;
        auto &spg3 = zs_grid->spg3;
        auto pol = zs::cuda_exec();

        if (!spg1.hasProperty(attrTag)) {
            spg1.append_channels(pol, {{attrTag, nchns}});
            spg2.append_channels(pol, {{attrTag, nchns}});
            spg3.append_channels(pol, {{attrTag, nchns}});
        } else {
            int m_nchns = spg1.getPropertySize(attrTag);
            if (m_nchns != nchns)
                throw std::runtime_error(
                    fmt::format("the SparseGrid already has [{}] with [{}] channels!", attrTag, m_nchns));
        }

        set_output("SparseGrid", zs_grid);
    }
};

ZENDEFNODE(ZSMultiGridAppendAttribute, {/* inputs: */
                                        {"SparseGrid", {"string", "Attribute", ""}, {"int", "ChannelNumber", "1"}},
                                        /* outputs: */
                                        {"SparseGrid"},
                                        /* params: */
                                        {},
                                        /* category: */
                                        {"Eulerian"}});

struct ZSCombineSparseGrid : INode {
    template <int OpId>
    void CSG(zs::CudaExecutionPolicy &pol, ZenoSparseGrid::spg_t &spgA, ZenoSparseGrid::spg_t &spgB, bool AisBigger,
             zs::SmallString sdfTag) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        pol(zs::Collapse(spgA.numBlocks(), spgA.block_size),
            [sdfv = zs::proxy<space>(spgA), smallerv = zs::proxy<space>(spgB), AisBigger,
             sdfTag] __device__(int blockno, int cellno) mutable {
                auto wcoord = sdfv.wCoord(blockno, cellno);

                float sdf_this = sdfv.value(sdfTag, blockno, cellno);
                float sdf_ref = smallerv.wSample(sdfTag, wcoord);

                float sdf_csg = sdf_this;
                if constexpr (OpId == 0)
                    sdf_csg = zs::min(sdf_this, sdf_ref);
                else if constexpr (OpId == 1)
                    sdf_csg = zs::max(sdf_this, sdf_ref);
                else if constexpr (OpId == 2) {
                    if (AisBigger)
                        sdf_csg = zs::max(sdf_this, -sdf_ref);
                    else
                        sdf_csg = zs::max(sdf_ref, -sdf_this);
                }

                sdfv(sdfTag, blockno, cellno) = sdf_csg;
            });
    }

    void apply() override {
        auto GridA = get_input<ZenoSparseGrid>("GridA");
        auto GridB = get_input<ZenoSparseGrid>("GridB");
        auto tag = get_input2<std::string>("SDFAttribute");
        auto op = get_input2<std::string>("OpType");

        auto &spgA = GridA->getSparseGrid();
        auto &spgB = GridB->getSparseGrid();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if (get_input2<bool>("WriteBack")) {
            auto &sdf = spgA;
            auto &smaller = spgB;

            size_t prevNumBlocks = sdf.numBlocks();
            sdf.resizePartition(pol, spgA.numBlocks() + spgB.numBlocks());

            pol(zs::range(smaller.numBlocks()),
                [sdfv = zs::proxy<space>(sdf), smallerv = zs::proxy<space>(smaller)] __device__(std::size_t i) mutable {
                    auto bcoord = smallerv._table._activeKeys[i];
                    sdfv._table.insert(bcoord);
                });

            size_t curNumBlocks = sdf.numBlocks();
            sdf.resizeGrid(curNumBlocks);

            pol(zs::Collapse(curNumBlocks - prevNumBlocks, sdf.block_size),
                [sdfv = zs::proxy<space>(sdf), sdfOffset = sdf.getPropertyOffset(tag),
                 prevNumBlocks] __device__(int blockno, int cellno) mutable {
                    int bo = blockno + prevNumBlocks;
                    sdfv(sdfOffset, bo, cellno) = sdfv._background;
                });

            if (op == "CSGUnion")
                CSG<0>(pol, sdf, smaller, true, tag);
            else if (op == "CSGIntersection")
                CSG<1>(pol, sdf, smaller, true, tag);
            else if (op == "CSGSubtract")
                CSG<2>(pol, sdf, smaller, true, tag);

            set_output("Grid", GridA);
        } else {
            ZenoSparseGrid::spg_t sdf{{{tag, 1}}, 0, zs::memsrc_e::device};
            // topo copy
            bool AisBigger = spgA.numBlocks() >= spgB.numBlocks();
            auto &bigger = AisBigger ? spgA : spgB;
            auto &smaller = AisBigger ? spgB : spgA;

            sdf._table = bigger._table;
            sdf._transform = bigger._transform;
            sdf._background = bigger._background;
            sdf._grid = bigger._grid;

            size_t prevNumBlocks = sdf.numBlocks();
            sdf.resizePartition(pol, spgA.numBlocks() + spgB.numBlocks());

            pol(zs::range(smaller.numBlocks()),
                [sdfv = zs::proxy<space>(sdf), smallerv = zs::proxy<space>(smaller)] __device__(std::size_t i) mutable {
                    auto bcoord = smallerv._table._activeKeys[i];
                    sdfv._table.insert(bcoord);
                });

            size_t curNumBlocks = sdf.numBlocks();
            sdf.resizeGrid(curNumBlocks);

            pol(zs::Collapse(curNumBlocks - prevNumBlocks, sdf.block_size),
                [sdfv = zs::proxy<space>(sdf), sdfOffset = sdf.getPropertyOffset(tag),
                 prevNumBlocks] __device__(int blockno, int cellno) mutable {
                    int bo = blockno + prevNumBlocks;
                    sdfv(sdfOffset, bo, cellno) = sdfv._background;
                });

            if (op == "CSGUnion")
                CSG<0>(pol, sdf, smaller, AisBigger, tag);
            else if (op == "CSGIntersection")
                CSG<1>(pol, sdf, smaller, AisBigger, tag);
            else if (op == "CSGSubtract")
                CSG<2>(pol, sdf, smaller, AisBigger, tag);

            auto zsSPG = std::make_shared<ZenoSparseGrid>();
            zsSPG->spg = std::move(sdf);

            set_output("Grid", zsSPG);
        }
    }
};

ZENDEFNODE(ZSCombineSparseGrid, {/* inputs: */
                                 {"GridA",
                                  "GridB",
                                  {"string", "SDFAttribute", "sdf"},
                                  {"enum CSGUnion CSGIntersection CSGSubtract", "OpType", "CSGUnion"},
                                  {"bool", "WriteBack", "0"}},
                                 /* outputs: */
                                 {"Grid"},
                                 /* params: */
                                 {},
                                 /* category: */
                                 {"Eulerian"}});

struct ZSGridTopoUnion : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("Grid");
        auto zs_topo = get_input<ZenoSparseGrid>("TopologyGrid");

        auto &spgA = zs_grid->spg;
        auto &spgB = zs_topo->spg;

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        size_t prevNumBlocks = spgA.numBlocks();
        spgA.resizePartition(pol, spgA.numBlocks() + spgB.numBlocks());

        pol(zs::range(spgB.numBlocks()),
            [spgAv = zs::proxy<space>(spgA), spgBv = zs::proxy<space>(spgB)] __device__(std::size_t i) mutable {
                auto bcoord = spgBv._table._activeKeys[i];
                spgAv._table.insert(bcoord);
            });

        size_t curNumBlocks = spgA.numBlocks();
        spgA.resizeGrid(curNumBlocks);

        size_t newNbs = curNumBlocks - prevNumBlocks;
        zs::memset(zs::mem_device, (void *)spgA._grid.tileOffset(prevNumBlocks), 0,
                   (std::size_t)newNbs * spgA._grid.tileBytes());

        if (get_input2<bool>("multigrid")) {
            /// @brief adjust multigrid accordingly
            // grid
            auto &spg1 = zs_grid->spg1;
            spg1.resize(pol, curNumBlocks);
            auto &spg2 = zs_grid->spg2;
            spg2.resize(pol, curNumBlocks);
            auto &spg3 = zs_grid->spg3;
            spg3.resize(pol, curNumBlocks);
            // table
            {
                const auto &table = spgA._table;
                auto &table1 = spg1._table;
                auto &table2 = spg2._table;
                auto &table3 = spg3._table;
                table1.reset(true);
                table1._cnt.setVal(curNumBlocks);
                table2.reset(true);
                table2._cnt.setVal(curNumBlocks);
                table3.reset(true);
                table3._cnt.setVal(curNumBlocks);

                table1._buildSuccess.setVal(1);
                table2._buildSuccess.setVal(1);
                table3._buildSuccess.setVal(1);
                pol(zs::range(curNumBlocks),
                    [table = zs::proxy<space>(table), tab1 = zs::proxy<space>(table1), tab2 = zs::proxy<space>(table2),
                     tab3 = zs::proxy<space>(table3)] __device__(std::size_t i) mutable {
                        auto bcoord = table._activeKeys[i];
                        tab1.insert(bcoord / 2, i, true);
                        tab2.insert(bcoord / 4, i, true);
                        tab3.insert(bcoord / 8, i, true);
                    });

                int tag1 = table1._buildSuccess.getVal();
                int tag2 = table2._buildSuccess.getVal();
                int tag3 = table3._buildSuccess.getVal();
                if (tag1 == 0 || tag2 == 0 || tag3 == 0)
                    zeno::log_error("check TopoUnion multigrid activate success state: {}, {}, {}\n", tag1, tag2, tag3);
            }
        }

        set_output("Grid", zs_grid);
    }
};

ZENDEFNODE(ZSGridTopoUnion, {/* inputs: */
                             {"Grid", "TopologyGrid", {"bool", "multigrid", "0"}},
                             /* outputs: */
                             {"Grid"},
                             /* params: */
                             {},
                             /* category: */
                             {"Eulerian"}});

struct ZSGridReduction : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("SparseGrid");
        auto attrTag = get_input2<std::string>("Attribute");
        auto op = get_input2<std::string>("Operation");

        auto &spg = zs_grid->spg;
        auto block_cnt = spg.numBlocks();

        auto tag = src_tag(zs_grid, attrTag);
        auto nchns = spg.getPropertySize(tag);
        bool isVec3 = nchns == 3 ? true : false;

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        size_t cell_cnt = block_cnt * spg.block_size;
        zs::Vector<float> res{spg.get_allocator(), count_warps(cell_cnt)};
        zs::memset(zs::mem_device, res.data(), 0, sizeof(float) * count_warps(cell_cnt));

        // maximum velocity
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), res = zs::proxy<space>(res), cell_cnt, isVec3, op = zs::SmallString{op},
             tagOffset = spg.getPropertyOffset(tag)] __device__(int blockno, int cellno) mutable {
                float val = 0;
                if (isVec3) {
                    float u = spgv.value(tagOffset + 0, blockno, cellno);
                    float v = spgv.value(tagOffset + 1, blockno, cellno);
                    float w = spgv.value(tagOffset + 2, blockno, cellno);

                    val = zs::sqrt(u * u + v * v + w * w);
                } else {
                    val = spgv.value(tagOffset, blockno, cellno);
                }

                size_t cellno_glb = blockno * spgv.block_size + cellno;

                if (op == "max")
                    reduce_max(cellno_glb, cell_cnt, val, res[cellno_glb / 32]);
                else if (op == "min")
                    reduce_min(cellno_glb, cell_cnt, val, res[cellno_glb / 32]);
                else if (op == "average")
                    reduce_add(cellno_glb, cell_cnt, val, res[cellno_glb / 32]);
            });

        float val_rdc;
        if (op == "max")
            val_rdc = reduce(pol, res, thrust::maximum<float>{});
        else if (op == "min")
            val_rdc = reduce(pol, res, thrust::minimum<float>{});
        else if (op == "average")
            val_rdc = reduce(pol, res, thrust::plus<float>{}) / cell_cnt;

        set_output("Value", std::make_shared<NumericObject>(val_rdc));
    }
};

ZENDEFNODE(ZSGridReduction, {/* inputs: */
                             {"SparseGrid", {"string", "Attribute", ""}, {"enum max min average", "Operation", "max"}},
                             /* outputs: */
                             {"Value"},
                             /* params: */
                             {},
                             /* category: */
                             {"Eulerian"}});

struct ZSGridVoxelPos : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("SparseGrid");
        auto attrTag = get_input2<std::string>("PosAttr");

        auto &spg = zs_grid->getSparseGrid();
        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if (!spg.hasProperty(attrTag)) {
            spg.append_channels(pol, {{attrTag, 3}});
        } else {
            int m_nchns = spg.getPropertySize(attrTag);
            if (m_nchns != 3)
                throw std::runtime_error("the size of the PosAttr is not 3!");
        }

        pol(zs::Collapse{spg.numBlocks(), spg.block_size},
            [spgv = zs::proxy<space>(spg), tag = zs::SmallString{attrTag}] __device__(int blockno, int cellno) mutable {
                auto wcoord = spgv.wCoord(blockno, cellno);
                auto block = spgv.block(blockno);
                block.template tuple<3>(tag, cellno) = wcoord;
            });

        set_output("SparseGrid", zs_grid);
    }
};

ZENDEFNODE(ZSGridVoxelPos, {/* inputs: */
                            {"SparseGrid", {"string", "PosAttr", "wPos"}},
                            /* outputs: */
                            {"SparseGrid"},
                            /* params: */
                            {},
                            /* category: */
                            {"Eulerian"}});

struct ZSMakeDenseSDF : INode {
    void apply() override {
        float dx = get_input2<float>("dx");
        int nx = get_input2<int>("nx");
        int ny = get_input2<int>("ny");
        int nz = get_input2<int>("nz");

        int nbx = float(nx + 7) / 8.f;
        int nby = float(ny + 7) / 8.f;
        int nbz = float(nz + 7) / 8.f;

        size_t numExpectedBlocks = nbx * nby * nbz;

        auto zsSPG = std::make_shared<ZenoSparseGrid>();
        auto &spg = zsSPG->spg;
        spg = ZenoSparseGrid::spg_t{{{"sdf", 1}}, numExpectedBlocks, zs::memsrc_e::device};
        spg.scale(dx);
        spg._background = dx;

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;
        using ivec3 = zs::vec<int, 3>;

        pol(zs::range(numExpectedBlocks),
            [table = zs::proxy<space>(spg._table), nbx, nby, nbz] __device__(int nb) mutable {
                int i = nb / (nby * nbz);
                nb -= i * (nby * nbz);
                int j = nb / nbz;
                int k = nb - j * nbz;
                table.insert(ivec3{int(i - nbx / 2) * 8, int(j - nby / 2) * 8, int(k - nbz / 2) * 8});
            });

        ivec3 sphere_c{0, 0, 0};
        int sphere_r = 10; // 10*dx

        auto bcnt = spg.numBlocks();
        pol(zs::range(bcnt * 512), [spgv = zs::proxy<space>(spg), sphere_c, sphere_r] __device__(int cellno) mutable {
#if 0            
			int bno = cellno / 512;
            int cno = cellno & 511;
            auto bcoord = spgv._table._activeKeys[bno];
            auto cellid = RM_CVREF_T(spgv)::local_offset_to_coord(cno);
            auto ccoord = bcoord + cellid;
#endif
            auto icoord = spgv.iCoord(cellno);
            auto dx = spgv.voxelSize()[0]; // spgv._transform(0, 0);

            float dist2c = zs::sqrt(float(zs::sqr(icoord[0] - sphere_c[0]) + zs::sqr(icoord[1] - sphere_c[1]) +
                                          zs::sqr(icoord[2] - sphere_c[2])));
            float dist2s = dist2c - sphere_r;

            float init_sdf = dist2s;
            if (dist2s > 2. * dx)
                init_sdf = 2. * dx;
            else if (dist2s < -2. * dx)
                init_sdf = -2. * dx;

            //spgv("sdf", bno, cno) = ;
            spgv("sdf", icoord) = init_sdf;
        });

        // spg.resize(numExpectedBlocks);

        spg.append_channels(pol, {{"v", 3}});

        set_output("Grid", zsSPG);
    }
};

ZENDEFNODE(ZSMakeDenseSDF, {/* inputs: */
                            {{"float", "dx", "1.0"}, {"int", "nx", "128"}, {"int", "ny", "128"}, {"int", "nz", "128"}},
                            /* outputs: */
                            {"Grid"},
                            /* params: */
                            {},
                            /* category: */
                            {"deprecated"}});

} // namespace zeno