#include "../../Structures.hpp"
#include "../../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include <iostream>


namespace zeno {

struct ZSJiggle : INode {
    virtual void apply() override {
        using namespace zs;

        auto zsvolume = get_input<ZenoParticles>("zsvolume");

        auto drivenTag = get_param<std::string>("drivenTag");

        auto cjTag = get_param<std::string>("curJiggleTag");
        auto pjTag = get_param<std::string>("preJiggleTag");
        auto ppjTag = get_param<std::string>("prePreJiggleTag");
        auto jwTag = get_param<std::string>("jiggleWeightTag");

        auto& verts = zsvolume->getParticles();


        auto jiggleStiffness = get_input2<float>("jiggleRate");
        // auto characterLen = get_input2<float>("characterLen");
        auto jiggleScale = get_input2<float>("jiggleScale");
        // jiggleStiffness /= characterLen;

        auto jiggleDamp = get_input2<float>("jiggleDamping");
        // auto jiggleWeight = get_input2<float>("jiggleWeight");

        // auto jiggleWs = jprim->attr<float>("jw");

        auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if(jiggleDamp > 0.8) {
            std::cout << "Input jiggle damp >= 0.8, clamp to 0.8" << std::endl;
            jiggleDamp = 0.8;
        } 


        auto jdt = 1.;

        cudaExec(range(verts.size()),
            [verts = proxy<space>({},verts),
                jiggleStiffness,jiggleScale,jiggleDamp,
                drivenTag = zs::SmallString(drivenTag),
                cjTag   = zs::SmallString(cjTag),
                pjTag   = zs::SmallString(pjTag),
                ppjTag  = zs::SmallString(ppjTag),
                jwTag   = zs::SmallString(jwTag),
                jdt] ZS_LAMBDA(int vi) mutable {
                auto cp = verts.template pack<3>(drivenTag,vi);
                auto cj = verts.template pack<3>(cjTag,vi);
                auto pj = verts.template pack<3>(pjTag,vi);
                auto ppj = verts.template pack<3>(ppjTag,vi);

                auto jw = verts(jwTag,vi);

                verts.template tuple<3>(ppjTag,vi) = verts.template pack<3>(pjTag,vi);
                verts.template tuple<3>(pjTag,vi) = verts.template pack<3>(cjTag,vi);

                auto jvec = ( 1- jiggleDamp) * (pj - ppj) / jdt;

                auto tension = jiggleStiffness * (cp - pj);
                jw = jiggleScale * jw;

                verts.template tuple<3>(cjTag,vi) = cp * (1 - jw) + jw * verts.template pack<3>(cjTag,vi);
        });   

        set_output("zsvolume",zsvolume); 

    }
};

ZENDEFNODE(ZSJiggle, {
    {"zsvolume",
        // {"float","jiggleWeight","1"},
        {"float","jiggleDamping","0.5"},
        {"float","jiggleRate","5"},
        // {"float","characterLen","1"},
        {"float","jiggleScale","1"},
    },
    {"zsvolume"},
    {
        {"string","drivenTag","x"},
        {"string","curJiggleTag","cj"},
        {"string","preJiggleTag","pj"},
        {"string","prePreJiggleTag","ppj"},
        {"string","jiggleWeightTag","jw"}
    },
    {"FEM"},
});

}