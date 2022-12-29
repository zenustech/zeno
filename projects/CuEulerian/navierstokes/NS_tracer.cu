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

namespace zeno {

struct ZSTracerAdvectDiffuse : INode {
    void compute(zs::CudaExecutionPolicy &pol, zs::SmallString tag, float diffuse, float dt, std::string scheme,
                 float speedScale, ZenoSparseGrid *NSGrid) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        if (scheme == "Semi-Lagrangian") {
            // Semi-Lagrangian advection (1st order)
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, vSrcTag = src_tag(NSGrid, "v"),
                 trcSrcTag = src_tag(NSGrid, tag),
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = spgv.iStaggeredPack(vSrcTag, icoord) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord - u_adv * dt);

                    spgv(trcDstTag, blockno, cellno) = trc_sl;
                });

            update_cur(NSGrid, tag);
        } else if (scheme == "BFECC") {
            // Back and Forth Error Compensation and Correction (BFECC)
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, vSrcTag = src_tag(NSGrid, "v"),
                 trcSrcTag = src_tag(NSGrid, tag),
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = spgv.iStaggeredPack(vSrcTag, icoord) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord - u_adv * dt);

                    spgv(trcDstTag, blockno, cellno) = trc_sl;
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, vSrcTag = src_tag(NSGrid, "v"),
                 trcTag = src_tag(NSGrid, tag), trcSrcTag = dst_tag(NSGrid, tag),
                 trcDstTag = zs::SmallString{"tmp"}] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = spgv.iStaggeredPack(vSrcTag, icoord) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord + u_adv * dt);
                    float trc_n = spgv.value(trcTag, blockno, cellno);

                    spgv(trcDstTag, blockno, cellno) = trc_n + (trc_n - trc_sl) / 2.f;
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, vSrcTag = src_tag(NSGrid, "v"),
                 trcTag = src_tag(NSGrid, tag), trcSrcTag = zs::SmallString{"tmp"},
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = spgv.iStaggeredPack(vSrcTag, icoord) * speedScale;
                    auto wcoord_src = wcoord - u_adv * dt;

                    float trc_sl = spgv.wSample(trcSrcTag, wcoord_src);

                    // clamp
                    auto arena = spgv.wArena(wcoord_src);
                    auto sl_mi = arena.minimum(trcTag);
                    auto sl_ma = arena.maximum(trcTag);
                    if (trc_sl > sl_ma || trc_sl < sl_mi) {
                        trc_sl = spgv.wSample(trcTag, wcoord_src);
                    }

                    spgv(trcDstTag, blockno, cellno) = trc_sl;
                });

            update_cur(NSGrid, tag);
        } else if (scheme == "Finite-Volume") {
            // Finite Volume Method (FVM)
            // numrtical flux of tracer
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, speedScale, tag = src_tag(NSGrid, tag),
                 vSrcTag = src_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    const int stcl = 2; // stencil point in each side
                    float trc[3][2 * stcl];

                    // | i - 2 | i - 1 | i | i + 1 |
                    for (int i = -stcl; i < stcl; ++i) {
                        trc[0][i + stcl] = spgv.value(tag, icoord + zs::vec<int, 3>(i, 0, 0));
                        trc[1][i + stcl] = spgv.value(tag, icoord + zs::vec<int, 3>(0, i, 0));
                        trc[2][i + stcl] = spgv.value(tag, icoord + zs::vec<int, 3>(0, 0, i));
                    }

                    float u_adv[3];
                    for (int ch = 0; ch < 3; ++ch)
                        u_adv[ch] = spgv.value(vSrcTag, ch, icoord) * speedScale;

                    // approximate value at i - 1/2
                    float flux[3];
                    for (int ch = 0; ch < 3; ++ch) {
                        // convection flux
                        if (u_adv[ch] < 0)
                            flux[ch] = u_adv[ch] * scheme::TVD_MUSCL3(trc[ch][1], trc[ch][2], trc[ch][3]);
                        else
                            flux[ch] = u_adv[ch] * scheme::TVD_MUSCL3(trc[ch][2], trc[ch][1], trc[ch][0]);
                    }

                    for (int ch = 0; ch < 3; ++ch) {
                        spgv("tmp", ch, blockno, cellno) = flux[ch];
                    }
                });

            // time integration of tracer
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, tag = src_tag(NSGrid, tag)] __device__(int blockno,
                                                                                              int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    float flux[3][2];
                    for (int ch = 0; ch < 3; ++ch) {
                        zs::vec<int, 3> offset{0, 0, 0};
                        offset[ch] = 1;

                        flux[ch][0] = spgv.value("tmp", ch, icoord);
                        flux[ch][1] = spgv.value("tmp", ch, icoord + offset);
                    }

                    float dtrc = 0;
                    for (int ch = 0; ch < 3; ++ch) {
                        dtrc += (flux[ch][0] - flux[ch][1]) / dx;
                    }
                    dtrc *= dt;

                    spgv(tag, blockno, cellno) += dtrc;
                });
        } else {
            throw std::runtime_error(fmt::format("Advection scheme [{}] not found!", scheme));
        }

        if (diffuse > 0) {
            // diffusion
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, diffuse, trcSrcTag = src_tag(NSGrid, tag),
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    const int stcl = 1; // stencil point in each side
                    float trc_x[2 * stcl + 1], trc_y[2 * stcl + 1], trc_z[2 * stcl + 1];

                    for (int i = -stcl; i <= stcl; ++i) {
                        trc_x[i + stcl] = spgv.value(trcSrcTag, icoord + zs::vec<int, 3>(i, 0, 0));
                        trc_y[i + stcl] = spgv.value(trcSrcTag, icoord + zs::vec<int, 3>(0, i, 0));
                        trc_z[i + stcl] = spgv.value(trcSrcTag, icoord + zs::vec<int, 3>(0, 0, i));
                    }

                    float trc_xx = scheme::central_diff_2nd(trc_x[0], trc_x[1], trc_x[2], dx);
                    float trc_yy = scheme::central_diff_2nd(trc_y[0], trc_y[1], trc_y[2], dx);
                    float trc_zz = scheme::central_diff_2nd(trc_z[0], trc_z[1], trc_z[2], dx);

                    float diff_term = diffuse * (trc_xx + trc_yy + trc_zz);

                    spgv(trcDstTag, blockno, cellno) = trc_x[1] + diff_term * dt;
                });
            update_cur(NSGrid, tag);
        }
    }

    void clampDensity(zs::CudaExecutionPolicy &pol, zs::SmallString tag, float clampBelow, ZenoSparseGrid *NSGrid) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), rhoOffset = spg.getPropertyOffset(src_tag(NSGrid, tag)),
             clampBelow] __device__(int blockno, int cellno) mutable {
                float rho = spgv.value(rhoOffset, blockno, cellno);
                rho = rho < clampBelow ? 0.f : rho;

                spgv(rhoOffset, blockno, cellno) = rho;
            });
    }

    void coolingTemp(zs::CudaExecutionPolicy &pol, zs::SmallString tag, float coolingRate, float dt,
                     ZenoSparseGrid *NSGrid) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), TOffset = spg.getPropertyOffset(src_tag(NSGrid, tag)), coolingRate,
             dt] __device__(int blockno, int cellno) mutable {
                float T = spgv.value(TOffset, blockno, cellno);
                T = zs::max(T - coolingRate * dt, 0.f);

                spgv(TOffset, blockno, cellno) = T;
            });
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto diffuse = get_input2<float>("Diffusion");
        auto dt = get_input2<float>("dt");
        auto scheme = get_input2<std::string>("Scheme");

        auto pol = zs::cuda_exec();
        ///
        if (get_input2<bool>("Density")) {
            compute(pol, "rho", diffuse, dt, scheme, 1.f, NSGrid.get());

            auto clampBelow = get_input2<float>("ClampDensityBelow");
            clampDensity(pol, "rho", clampBelow, NSGrid.get());
        }
        if (get_input2<bool>("Temperature")) {
            compute(pol, "T", diffuse, dt, scheme, 1.f, NSGrid.get());

            auto coolingRate = get_input2<float>("CoolingRate");
            coolingTemp(pol, "T", coolingRate, dt, NSGrid.get());
        }
        if (get_input2<bool>("Fuel")) {
            auto speedScale = get_input2<float>("FuelSpeedScale");
            compute(pol, "fuel", diffuse, dt, scheme, speedScale, NSGrid.get());
        }

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSTracerAdvectDiffuse, {/* inputs: */
                                   {"NSGrid",
                                    "dt",
                                    {"bool", "Density", "1"},
                                    {"bool", "Temperature", "1"},
                                    {"bool", "Fuel", "0"},
                                    {"enum Finite-Volume Semi-Lagrangian BFECC", "Scheme", "Finite-Volume"},
                                    {"float", "ClampDensityBelow", "0.01"},
                                    {"float", "CoolingRate", "0.0"},
                                    {"float", "FuelSpeedScale", "0.05"},
                                    {"float", "Diffusion", "0.0"}},
                                   /* outputs: */
                                   {"NSGrid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

struct ZSTracerEmission : INode {
    void compute(zs::CudaExecutionPolicy &pol, zs::SmallString tag, ZenoSparseGrid *NSGrid, ZenoSparseGrid *EmitSDF,
                 bool fromObj) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto &sdf = EmitSDF->spg;

        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), dx, fromObj,
             tag = src_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                auto wcoord = spgv.wCoord(blockno, cellno);
                auto emit_sdf = sdfv.wSample("sdf", wcoord);

                if (emit_sdf <= 1.5f * dx) {
                    // fix me: naive emission
                    spgv(tag, blockno, cellno) = 1.f;
                }

                if (fromObj) {
                    if (spgv.value("mark", blockno, cellno) > 0.5f)
                        spgv(tag, blockno, cellno) = 1.f;
                }
            });
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto EmitSDF = get_input<ZenoSparseGrid>("EmitterSDF");
        auto fromObj = get_input2<bool>("fromObjBoundary");

        if (!NSGrid->getSparseGrid().hasProperty("mark")) {
            fromObj = false;
        }

        auto pol = zs::cuda_exec();

        if (get_input2<bool>("Density"))
            compute(pol, "rho", NSGrid.get(), EmitSDF.get(), fromObj);
        if (get_input2<bool>("Temperature"))
            compute(pol, "T", NSGrid.get(), EmitSDF.get(), fromObj);
        if (get_input2<bool>("Fuel"))
            compute(pol, "fuel", NSGrid.get(), EmitSDF.get(), fromObj);

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSTracerEmission, {/* inputs: */
                              {"NSGrid",
                               "EmitterSDF",
                               {"bool", "Density", "1"},
                               {"bool", "Temperature", "1"},
                               {"bool", "Fuel", "0"},
                               {"bool", "fromObjBoundary", "0"}},
                              /* outputs: */
                              {"NSGrid"},
                              /* params: */
                              {},
                              /* category: */
                              {"Eulerian"}});

struct ZSSmokeBuoyancy : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto dt = get_input2<float>("dt");
        auto gravity = get_input2<zeno::vec3f>("Gravity");
        auto alpha = get_input2<float>("DensityCoef");
        auto beta = get_input2<float>("TemperatureCoef");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // add force (accelaration)
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), dt, alpha, beta, gravity = zs::vec<float, 3>::from_array(gravity),
             vSrcTag = src_tag(NSGrid, "v"), rhoSrcTag = src_tag(NSGrid, "rho"),
             TSrcTag = src_tag(NSGrid, "T")] __device__(int blockno, int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                float rho_this = spgv.value(rhoSrcTag, blockno, cellno);
                float T_this = spgv.value(TSrcTag, blockno, cellno);

                for (int ch = 0; ch < 3; ++ch) {
                    zs::vec<int, 3> offset{0, 0, 0};
                    offset[ch] = -1;

                    auto [bno, cno] = spgv.decomposeCoord(icoord + offset);
                    float rho_face = 0, T_face = 0;
                    if (bno >= 0) {
                        rho_face = spgv.value(rhoSrcTag, bno, cno);
                        T_face = spgv.value(TSrcTag, bno, cno);
                    }

                    rho_face = 0.5f * (rho_this + rho_face);
                    T_face = 0.5f * (T_this + T_face);

                    float G_scale = alpha * rho_face - beta * T_face;

                    spgv(vSrcTag, ch, blockno, cellno) += G_scale * gravity[ch] * dt;
                }
            });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSSmokeBuoyancy, {/* inputs: */
                             {"NSGrid",
                              "dt",
                              {"vec3f", "Gravity", "0, -9.8, 0"},
                              {"float", "DensityCoef", "0.0"},
                              {"float", "TemperatureCoef", "0.0"}},
                             /* outputs: */
                             {"NSGrid"},
                             /* params: */
                             {},
                             /* category: */
                             {"Eulerian"}});

struct ZSVolumeCombustion : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto dt = get_input2<float>("dt");
        auto ignitionT = get_input2<float>("IgnitionTemperature");
        auto burnSpeed = get_input2<float>("BurnSpeed");
        auto rhoEmitAmount = get_input2<float>("DensityEmitAmount");
        auto TEmitAmount = get_input2<float>("TemperatureEmitAmount");
        auto volumeExpansion = get_input2<float>("VolumeExpansion");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // add force (accelaration)
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), dt, ignitionT, burnSpeed, rhoEmitAmount, TEmitAmount, volumeExpansion,
             fuelOffset = spg.getPropertyOffset(src_tag(NSGrid, "fuel")),
             rhoOffset = spg.getPropertyOffset(src_tag(NSGrid, "rho")),
             TOffset = spg.getPropertyOffset(src_tag(NSGrid, "T")),
             divOffset = spg.getPropertyOffset(src_tag(NSGrid, "div"))] __device__(int blockno, int cellno) mutable {
                float T = spgv.value(TOffset, blockno, cellno);
                float div = 0.f;

                if (T >= ignitionT) {
                    float fuel = spgv.value(fuelOffset, blockno, cellno);
                    float rho = spgv.value(rhoOffset, blockno, cellno);

                    float dF = zs::min(burnSpeed * dt, fuel);
                    fuel -= dF;
                    rho += rhoEmitAmount * dF;
                    T += TEmitAmount * dF;

                    div = volumeExpansion * dF / dt;

                    spgv(fuelOffset, blockno, cellno) = fuel;
                    spgv(rhoOffset, blockno, cellno) = rho;
                    spgv(TOffset, blockno, cellno) = T;
                }

                spgv(divOffset, blockno, cellno) = div;
            });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSVolumeCombustion, {/* inputs: */
                                {"NSGrid",
                                 "dt",
                                 {"float", "IgnitionTemperature", "0.8"},
                                 {"float", "BurnSpeed", "0.5"},
                                 {"float", "DensityEmitAmount", "0.5"},
                                 {"float", "TemperatureEmitAmount", "0.5"},
                                 {"float", "VolumeExpansion", "1.2"}},
                                /* outputs: */
                                {"NSGrid"},
                                /* params: */
                                {},
                                /* category: */
                                {"Eulerian"}});

} // namespace zeno