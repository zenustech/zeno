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
        auto frequency = get_input2<vec3f>("Frequency");
        auto offset = get_input2<vec3f>("Offset");
        auto roughness = get_input2<float>("Roughness");
        auto turbulence = get_input2<int>("Turbulence");
        auto amplitude = get_input2<float>("Amplitude");
        auto attenuation = get_input2<float>("Attenuation");
        auto mean = get_input2<float>("MeanNoise");

        auto tag = src_tag(zsSPG, attrTag);

        auto &spg = zsSPG->spg;
        auto block_cnt = spg.numBlocks();

        if (!spg.hasProperty(tag))
            throw std::runtime_error(fmt::format("GridAttribute [{}] doesn't exist!", tag.asString()));
        const int nchns = spg.getPropertySize(tag);

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        zs::Vector<int> flag{1, zs::memsrc_e::um};
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), tag, nchns, frequency = zs::vec<float, 3>::from_array(frequency),
             offset = zs::vec<float, 3>::from_array(offset), roughness, turbulence, amplitude, attenuation,
             mean] __device__(int blockno, int cellno) mutable {
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

                    spgv._grid.tuple(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) =
                        spgv._grid.pack(zs::dim_c<3>, tag, blockno * spgv.block_size + cellno) + noise;
                } else if (nchns == 1) {
                    float fbm = 0;
                    for (int i = 0; i < turbulence; ++i, pp *= 2.f, scale *= roughness) {
                        float pln = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2]);
                        fbm += scale * pln;
                    }
                    auto noise = zs::pow(fbm, attenuation) + mean;

                    spgv(tag, blockno, cellno) += noise;
                }
            });

        set_output("SparseGrid", zsSPG);
    }
};

ZENDEFNODE(ZSGridPerlinNoise, {/* inputs: */
                               {"SparseGrid",
                                {"string", "GridAttribute", "v"},
                                {"vec3f", "Frequency", "1, 1, 1"},
                                {"vec3f", "Offset", "0, 0, 0"},
                                {"float", "Roughness", "0.5"},
                                {"int", "Turbulence", "4"},
                                {"float", "Amplitude", "1.0"},
                                {"float", "Attenuation", "1.0"},
                                {"float", "MeanNoise", "0"}},
                               /* outputs: */
                               {"SparseGrid"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

} // namespace zeno