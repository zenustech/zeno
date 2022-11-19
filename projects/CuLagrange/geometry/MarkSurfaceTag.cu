#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/tiled_vector_ops.hpp"

namespace zeno {

using T = float;

struct ZSMarkSurfaceMesh : zeno::INode {
    virtual void apply() override {
        using namespace zs;

        auto zsparticles = get_input<ZenoParticles>("ZSParticles");
        if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag)) 
            throw std::runtime_error("the input zsparticles has no surface points");

        auto& verts = zsparticles->getParticles();
        const auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];
        auto tagName = get_input2<std::string>("tagName");

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        if(!verts.hasProperty(tagName))
            verts.append_channels(cudaPol,{{tagName,1}});

        TILEVEC_OPS::fill(cudaPol,verts,tagName,(T)0.0);

        cudaPol(zs::range(points.size()),
            [verts = proxy<space>({},verts),tagName = zs::SmallString(tagName),points = proxy<space>({},points)] ZS_LAMBDA(int pi) mutable {
                auto vi = reinterpret_bits<int>(points("inds",pi));
                verts(tagName,vi) = (T)1.0;
        });

        set_output("ZSParticles",zsparticles);

    }

};

ZENDEFNODE(ZSMarkSurfaceMesh, {{{"ZSParticles"},{"string","tagName","RENAMEME"}},
                            {{"ZSParticles"}},
                            {},
                            {"ZSGeometry"}});


};