#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ZSMakeDenseSDF : INode {
    void apply() override {
        float dx = get_param<float>("dx");
        if (has_input("Dx")) {
            dx = get_input2<float>("Dx");
        }
        int nx = get_input2<int>("nx");
        int ny = get_input2<int>("ny");
        int nz = get_input2<int>("nz");

        int nbx = float(nx + 7) / 8.f;
        int nby = float(ny + 7) / 8.f;
        int nbz = float(nz + 7) / 8.f;

        size_t numExpectedBlocks = nbx * nby * nbz;

        auto zsSPG = zeno::IObject::make<ZenoSparseGrid>();
        auto &spg = zsSPG->spg;
        spg = ZenoSparseGrid::spg_t{{{"sdf", 1}}, numExpectedBlocks, zs::memsrc_e::device, 0};
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
                table.insert(ivec3{i * 8, j * 8, k * 8});
            });

        ivec3 sphere_c{nx / 2, ny / 2, nz / 2};
        float sphere_r = 10.f;

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

            //spgv("sdf", bno, cno) = ;
            spgv("sdf", icoord) = dist2c > sphere_r ? dx : -dx;
        });

        // spg.resize(numExpectedBlocks);

        spg.append_channels(pol, {{"v", 3}});

        set_output("Grid", zsSPG);
    }
};

ZENDEFNODE(ZSMakeDenseSDF, {/* inputs: */
                            {"Dx", {"int", "nx", "128"}, {"int", "ny", "128"}, {"int", "nz", "128"}},
                            /* outputs: */
                            {"Grid"},
                            /* params: */
                            {{"float", "dx", "1.0"}},
                            /* category: */
                            {"Eulerian"}});

struct ZSMakeSparseGrid : INode {
    void apply() override {
        auto dx = get_input2<float>("Dx");
        auto bg = get_input2<float>("backgroud");
        auto type = get_param<std::string>("type");
        auto structure = get_param<std::string>("structure");
        auto channel = get_param<std::string>("channelName");

        auto zsSPG = zeno::IObject::make<ZenoSparseGrid>();
        auto &spg = zsSPG->spg;

        int nc = 1;
        if (type == "scalar")
            nc = 1;
        else if (type == "vector3")
            nc = 3;

        spg = ZenoSparseGrid::spg_t{{{channel, nc}}, 0, zs::memsrc_e::device, 0};
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
                              {{"float", "Dx", "1.0"}, {"float", "background", "0"}},
                              /* outputs: */
                              {"Grid"},
                              /* params: */
                              {{"enum scaler vector3", "type", "scalar"},
                               {"enum cell-centered vertex-centered", "structure", "cell-centered "},
                               {"string", "channelName", ""}},
                              /* category: */
                              {"Eulerian"}});

struct ZSGridTopoCopy : INode {
    void apply() override {
        auto zs_grid = get_input<ZenoSparseGrid>("Grid");
        auto zs_topo = get_input<ZenoSparseGrid>("Topology");

        auto &grid = zs_grid->spg;
        auto &topo = zs_topo->spg;

        // topo copy
        grid._table = topo._table;
        grid._transform = topo._transform;
        grid._grid.resize(topo.numBlocks() * topo.block_size);

        set_output("Grid", zs_grid);
    }
};

ZENDEFNODE(ZSGridTopoCopy, {/* inputs: */
                              {"Grid", "Topology"},
                              /* outputs: */
                              {"Grid"},
                              /* params: */
                              {},
                              /* category: */
                              {"Eulerian"}});



} // namespace zeno