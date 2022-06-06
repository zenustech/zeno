#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/gradient_field.hpp"

namespace zeno {

struct ZSEvalGradientFieldOnTets : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;

    virtual void apply() override {
        using namespace zs;
        auto zstets = get_input<ZenoParticles>("zstets");
        auto& verts = zstets->getParticles();

        auto attr = get_param<std::string>("tag");
        auto attrg = get_param<std::string>("gtag");
        if(!verts.hasProperty(attr)){
            fmt::print("the input zstets does not contain specified channel:{}\n",attr);
            throw std::runtime_error("the input zstets does not contain specified channel");
        }
        if(verts.getChannelSize(attr)){
            fmt::print("only scaler field is currently supported\n");
            throw std::runtime_error("only scaler field is currently supported");
        }

        const auto& eles = zstets->getQuadraturePoints();
        auto cdim = eles.getChannelSize("inds");
        if(cdim != 4)
            throw std::runtime_error("ZSEvalGradientFieldOnTets: invalid simplex size");



        static dtiles_t etemp(eles.get_allocator(),{{"g",3}},eles.size());
        static dtiles_t vtemp{verts.get_allocator(),{
            {"T",1},
        },verts.size()};

        etemp.resize(eles.size());
        vtemp.resize(verts.size());

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        eles.append_channels(cudaPol,{{attrg,3}});

        // copy the scaler field from verts to vtemp
        cudaPol(zs::range(verts.size()),
            [verts = proxy<space>({},verts),vtemp = proxy<space>({},vtemp),attr = zs::SmallString(attr),tag = zs::SmallString("T")]
                ZS_LAMBDA(int vi) mutable {
                    vtemp(tag,vi) = verts(attr,vi);
        });

        compute_gradient(cudaPol,eles,verts,"x",vtemp,"T",etemp,"g");
        // copy the gradient field from etemp to eles
        cudaPol(zs::range(eles.size()),
            [eles = proxy<space>({},verts),etemp = proxy<space>({},etemp),gtag = zs::SmallString(gtag)]
                ZS_LAMBDA(int ei) mutable {
                    eles.tuple<3>(gtag,ei) = etemp.pack<3>("g",ei);
        });

        set_output("zstets",zstets);
    }
};

ZENDEFNODE(ZSEvalGradientFieldOnTets, {
                                    {"zstets"},
                                    {"zstets"},
                                    {
                                        {"string","tag","T"},{"string","gtag","gradT"}
                                    },
                                    {"FEM"}
});

};