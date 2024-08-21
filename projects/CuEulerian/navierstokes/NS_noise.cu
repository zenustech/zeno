#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/utils/log.h>

#include "../utils.cuh"
#include "Noise.cuh"

namespace zeno {

struct ZSGridPerlinNoise : INode {
    virtual void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto attrTag = get_input2<std::string>("GridAttribute");
        auto opType = get_input2<std::string>("OpType");
        auto frequency = get_input2<vec3f>("Frequency");
        auto offset = get_input2<vec3f>("Offset");
        auto roughness = get_input2<float>("Roughness");
        auto turbulence = get_input2<int>("Turbulence");
        auto amplitude = get_input2<float>("Amplitude");
        auto attenuation = get_input2<float>("Attenuation");
        auto mean = get_input2<vec3f>("MeanNoise");

        bool isAccumulate = opType == "accumulate" ? true : false;

        auto tag = src_tag(zsSPG, attrTag);

        auto &spg = zsSPG->spg;
        auto block_cnt = spg.numBlocks();

        if (!spg.hasProperty(tag))
            throw std::runtime_error(fmt::format("GridAttribute [{}] doesn't exist!", tag));
        const int nchns = spg.getPropertySize(tag);

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), tag, nchns, isAccumulate,
             frequency = zs::vec<float, 3>::from_array(frequency), offset = zs::vec<float, 3>::from_array(offset),
             roughness, turbulence, amplitude, attenuation,
             mean = zs::vec<float, 3>::from_array(mean)] __device__(int blockno, int cellno) mutable {
                auto wcoord = spgv.wCoord(blockno, cellno);
                auto pp = frequency * wcoord - offset;

                float scale = amplitude;

                if (nchns == 3) {
                    // fractal Brownian motion
                    auto fbm = zs::vec<float, 3>::uniform(0);
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        zs::vec<float, 3> pln{ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2]),
                                              ZSPerlinNoise1::perlin(pp[1], pp[2], pp[0]),
                                              ZSPerlinNoise1::perlin(pp[2], pp[0], pp[1])};
                        fbm += scale * pln;
                    }
                    auto noise = zs::vec<float, 3>{zs::pow(fbm[0], attenuation), zs::pow(fbm[1], attenuation),
                                                   zs::pow(fbm[2], attenuation)} +
                                 mean;

                    if (isAccumulate)
                        spgv._grid.tuple(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) =
                            spgv._grid.pack(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) + noise;
                    else
                        spgv._grid.tuple(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) = noise;

                } else if (nchns == 1) {
                    float fbm = 0;
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        float pln = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2]);
                        fbm += scale * pln;
                    }
                    auto noise = zs::pow(fbm, attenuation) + mean[0];

                    if (isAccumulate)
                        spgv(tag, blockno, cellno) += noise;
                    else
                        spgv(tag, blockno, cellno) = noise;
                }
            });

        set_output("SparseGrid", zsSPG);
    }
};

ZENDEFNODE(ZSGridPerlinNoise, {/* inputs: */
                               {"SparseGrid",
                                {gParamType_String, "GridAttribute", "v"},
                                {"enum replace accumulate", "OpType", "accumulate"},
                                {gParamType_Vec3f, "Frequency", "1, 1, 1"},
                                {gParamType_Vec3f, "Offset", "0, 0, 0"},
                                {gParamType_Float, "Roughness", "0.5"},
                                {gParamType_Int, "Turbulence", "4"},
                                {gParamType_Float, "Amplitude", "1.0"},
                                {gParamType_Float, "Attenuation", "1.0"},
                                {gParamType_Vec3f, "MeanNoise", "0, 0, 0"}},
                               /* outputs: */
                               {"SparseGrid"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSGridCurlNoise : INode {
    virtual void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto attrTag = get_input2<std::string>("GridAttribute");
        bool isStaggered = get_input2<bool>("staggered");
        auto opType = get_input2<std::string>("OpType");
        auto frequency = get_input2<vec3f>("Frequency");
        auto offset = get_input2<vec3f>("Offset");
        auto roughness = get_input2<float>("Roughness");
        auto turbulence = get_input2<int>("Turbulence");
        auto amplitude = get_input2<float>("Amplitude");
        auto mean = get_input2<vec3f>("MeanNoise");

        bool isAccumulate = opType == "accumulate" ? true : false;

        auto tag = src_tag(zsSPG, attrTag);

        auto &spg = zsSPG->spg;
        auto block_cnt = spg.numBlocks();

        if (!spg.hasProperty(tag))
            throw std::runtime_error(fmt::format("GridAttribute [{}] doesn't exist!", tag));
        if (spg.getPropertySize(tag) != 3)
            throw std::runtime_error(fmt::format("GridAttribute [{}] must have 3 channels!", tag));

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if (isStaggered) {
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), tag, isAccumulate, frequency = zs::vec<float, 3>::from_array(frequency),
                 offset = zs::vec<float, 3>::from_array(offset), roughness, turbulence, amplitude,
                 mean = zs::vec<float, 3>::from_array(mean)] __device__(int blockno, int cellno) mutable {
                    constexpr float eps = 1e-4f;
                    float pln1, pln2, curl;
                    // u
                    auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, 0);
                    auto pp = frequency * wcoord_face - offset;
                    float scale = amplitude;

                    curl = 0;
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        pln1 = ZSPerlinNoise1::perlin(pp[0], pp[1] + eps, pp[2]);
                        pln2 = ZSPerlinNoise1::perlin(pp[0], pp[1] - eps, pp[2]);
                        curl += scale * (pln1 - pln2) / (2.f * eps);
                        pln1 = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2] + eps);
                        pln2 = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2] - eps);
                        curl -= scale * (pln1 - pln2) / (2.f * eps);
                    }

                    if (isAccumulate)
                        spgv(tag, 0, blockno, cellno) += curl + mean[0];
                    else
                        spgv(tag, 0, blockno, cellno) = curl + mean[0];

                    // v
                    wcoord_face = spgv.wStaggeredCoord(blockno, cellno, 1);
                    pp = frequency * wcoord_face - offset;
                    scale = amplitude;

                    curl = 0;
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        pln1 = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2] + eps);
                        pln2 = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2] - eps);
                        curl += scale * (pln1 - pln2) / (2.f * eps);
                        pln1 = ZSPerlinNoise1::perlin(pp[0] + eps, pp[1], pp[2]);
                        pln2 = ZSPerlinNoise1::perlin(pp[0] - eps, pp[1], pp[2]);
                        curl -= scale * (pln1 - pln2) / (2.f * eps);
                    }

                    if (isAccumulate)
                        spgv(tag, 1, blockno, cellno) += curl + mean[1];
                    else
                        spgv(tag, 1, blockno, cellno) = curl + mean[1];

                    // w
                    wcoord_face = spgv.wStaggeredCoord(blockno, cellno, 2);
                    pp = frequency * wcoord_face - offset;
                    scale = amplitude;

                    curl = 0;
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        pln1 = ZSPerlinNoise1::perlin(pp[0] + eps, pp[1], pp[2]);
                        pln2 = ZSPerlinNoise1::perlin(pp[0] - eps, pp[1], pp[2]);
                        curl += scale * (pln1 - pln2) / (2.f * eps);
                        pln1 = ZSPerlinNoise1::perlin(pp[0], pp[1] + eps, pp[2]);
                        pln2 = ZSPerlinNoise1::perlin(pp[0], pp[1] - eps, pp[2]);
                        curl -= scale * (pln1 - pln2) / (2.f * eps);
                    }

                    if (isAccumulate)
                        spgv(tag, 2, blockno, cellno) += curl + mean[2];
                    else
                        spgv(tag, 2, blockno, cellno) = curl + mean[2];
                });
        } else {
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), tag, isAccumulate, frequency = zs::vec<float, 3>::from_array(frequency),
                 offset = zs::vec<float, 3>::from_array(offset), roughness, turbulence, amplitude,
                 mean = zs::vec<float, 3>::from_array(mean)] __device__(int blockno, int cellno) mutable {
                    constexpr float eps = 1e-4f;
                    float pln1, pln2, dPln[3];
                    auto curl = zs::vec<float, 3>::uniform(0);

                    auto wcoord = spgv.wCoord(blockno, cellno);
                    auto pp = frequency * wcoord - offset;
                    float scale = amplitude;

                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        pln1 = ZSPerlinNoise1::perlin(pp[0] + eps, pp[1], pp[2]);
                        pln2 = ZSPerlinNoise1::perlin(pp[0] - eps, pp[1], pp[2]);
                        dPln[0] = (pln1 - pln2) / (2.f * eps);
                        pln1 = ZSPerlinNoise1::perlin(pp[0], pp[1] + eps, pp[2]);
                        pln2 = ZSPerlinNoise1::perlin(pp[0], pp[1] - eps, pp[2]);
                        dPln[1] = (pln1 - pln2) / (2.f * eps);
                        pln1 = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2] + eps);
                        pln2 = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2] - eps);
                        dPln[2] = (pln1 - pln2) / (2.f * eps);

                        curl += scale * zs::vec<float, 3>{dPln[1] - dPln[2], dPln[2] - dPln[0], dPln[0] - dPln[1]};
                    }

                    if (isAccumulate)
                        spgv._grid.tuple(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) =
                            spgv._grid.pack(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) + curl + mean;
                    else
                        spgv._grid.tuple(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) = curl + mean;
                });
        }

        set_output("SparseGrid", zsSPG);
    }
};

ZENDEFNODE(ZSGridCurlNoise, {/* inputs: */
                             {"SparseGrid",
                              {gParamType_String, "GridAttribute", "v"},
                              {gParamType_Bool, "staggered", "1"},
                              {"enum replace accumulate", "OpType", "accumulate"},
                              {gParamType_Vec3f, "Frequency", "1, 1, 1"},
                              {gParamType_Vec3f, "Offset", "0, 0, 0"},
                              {gParamType_Float, "Roughness", "0.5"},
                              {gParamType_Int, "Turbulence", "4"},
                              {gParamType_Float, "Amplitude", "1.0"},
                              {gParamType_Vec3f, "MeanNoise", "0, 0, 0"}},
                             /* outputs: */
                             {"SparseGrid"},
                             /* params: */
                             {},
                             /* category: */
                             {"Eulerian"}});

} // namespace zeno