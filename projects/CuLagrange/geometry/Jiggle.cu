#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include <iostream>


namespace zeno {

struct Jiggle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto jprim = get_input<zeno::PrimitiveObject>("jprim");

        const auto& cpos_vec =  prim->attr<zeno::vec3f>("pos");
        // const auto& ppos_vec =  prim->attr<zeno::vec3f>("prePos");
        // const auto& pppos_vec = prim->attr<zeno::vec3f>("preprePos");

        auto& j_cpos_vec = jprim->attr<zeno::vec3f>("cj");
        auto& j_ppos_vec = jprim->attr<zeno::vec3f>("pj");
        auto& j_pppos_vec= jprim->attr<zeno::vec3f>("ppj");

        auto jiggleStiffness = get_input2<float>("jiggleRate");
        // auto characterLen = get_input2<float>("characterLen");
        auto jiggleScale = get_input2<float>("jiggleScale");
        // jiggleStiffness /= characterLen;

        auto jiggleDamp = get_input2<float>("jiggleDamping");
        // auto jiggleWeight = get_input2<float>("jiggleWeight");

        auto jiggleWs = jprim->attr<float>("jw");

        if(jiggleDamp > 0.8) {
            std::cout << "Input jiggle damp >= 0.8, clamp to 0.8" << std::endl;
            jiggleDamp = 0.8;
        }

        // auto jiggleRate = get_input2<float>("jiggleRate");
        // auto jiggle
        // auto jiggleStiffness = 1.0/jiggleRate;
        auto jdt = 1;

        auto ompExec = zs::omp_exec();
        constexpr auto space = zs::execspace_e::openmp;

        ompExec(zs::range(prim->size()),
            [&cpos_vec,&j_cpos_vec,&j_ppos_vec,&j_pppos_vec,jdt,jiggleScale,&jiggleWs,jiggleStiffness,jiggleDamp] (int vi) mutable {
                const auto& cpos = cpos_vec[vi];
                // const auto& ppos = ppos_vec[vi];
                // const auto& pppos= pppos_vec[vi];
                
                j_pppos_vec[vi] = j_ppos_vec[vi];
                j_ppos_vec[vi] = j_cpos_vec[vi];

                auto& cj = j_cpos_vec[vi];
                const auto& pj = j_ppos_vec[vi];
                const auto& ppj = j_pppos_vec[vi];

                auto jvec = (1 - jiggleDamp) * (pj - ppj)/jdt;    

                auto tension = jiggleStiffness * (cpos - pj);
                cj += jvec * jdt + 0.5 * tension * jdt * jdt;  

                auto jw = jiggleScale * jiggleWs[vi];

                cj = cpos * (1 - jw) + jw * cj;  

                if(vi == 0) {
                    // auto check_cj = verts.template pack<3>(cjTag,vi);
                    printf("IN_JIGGLE[0] : %f %f %f %f %f %f\n",(float)cj[0],(float)cj[1],(float)cj[2],
                        (float)cpos[0],(float)cpos[1],(float)cpos[2]);
                }

        });

        // #pragma omp parallel for
        // for(size_t i = 0;i < prim->size();++i){
        //     const auto& cpos = cpos_vec[i];
        //     // const auto& ppos = ppos_vec[i];
        //     // const auto& pppos= pppos_vec[i];
            
        //     j_pppos_vec[i] = j_ppos_vec[i];
        //     j_ppos_vec[i] = j_cpos_vec[i];

        //     auto& cj = j_cpos_vec[i];
        //     const auto& pj = j_ppos_vec[i];
        //     const auto& ppj = j_pppos_vec[i];

        //     auto jvec = (1 - jiggleDamp) * (pj - ppj)/jdt;    

        //     auto tension = jiggleStiffness * (cpos - pj);
        //     cj += jvec * jdt + 0.5 * tension * jdt * jdt;  

        //     auto jw = jiggleScale * jiggleWs[i];

        //     cj = cpos * (1 - jw) + jw * cj;
        //     // if(i == 0)
        //     //     std::cout << "cj : " << cj[0] << "\t" << cj[1] << "\t" << cj[2] << std::endl;  
        // }

        set_output("jprim",jprim); 

    }
};

ZENDEFNODE(Jiggle, {
    {"prim","jprim",
        // {gParamType_Float,"jiggleWeight","1"},
        {gParamType_Float,"jiggleDamping","0.5"},
        {gParamType_Float,"jiggleRate","5"},
        // {gParamType_Float,"characterLen","1"},
        {gParamType_Float,"jiggleScale","1"},
    },
    {"jprim"},
    {},
    {"FEM"},
});


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
                    auto jw = verts(jwTag,vi);

                    verts.template tuple<3>(ppjTag,vi) = verts.template pack<3>(pjTag,vi);
                    verts.template tuple<3>(pjTag,vi) = verts.template pack<3>(cjTag,vi);

                    auto cp = verts.template pack<3>(drivenTag,vi);
                    auto cj = verts.template pack<3>(cjTag,vi);
                    auto pj = verts.template pack<3>(pjTag,vi);
                    auto ppj = verts.template pack<3>(ppjTag,vi);

                    auto jvec = ( 1- jiggleDamp) * (pj - ppj) / jdt;

                    auto tension = jiggleStiffness * (cp - pj);
                    jw = jiggleScale * jw;

                    cj += jvec * jdt + 0.5 * tension * jdt * jdt;  

                    verts.template tuple<3>(cjTag,vi) = cp * (1 - jw) + jw * cj;

                    // if(vi == 0) {
                    //     auto check_cj = verts.template pack<3>(cjTag,vi);
                    //     printf("ZS_JIGGLE[0] : %f %f %f %f %f %f %f %f %f\n",(float)check_cj[0],(float)check_cj[1],(float)check_cj[2],
                    //         (float)cp[0],(float)cp[1],(float)cp[2],(float)cj[0],(float)cj[1],(float)cj[2]);
                    // }

        });   

        set_output("zsvolume",zsvolume); 

    }
};

ZENDEFNODE(ZSJiggle, {
    {"zsvolume",
        // {gParamType_Float,"jiggleWeight","1"},
        {gParamType_Float,"jiggleDamping","0.5"},
        {gParamType_Float,"jiggleRate","5"},
        // {gParamType_Float,"characterLen","1"},
        {gParamType_Float,"jiggleScale","1"},
    },
    {"zsvolume"},
    {
        {gParamType_String,"drivenTag","x"},
        {gParamType_String,"curJiggleTag","cj"},
        {gParamType_String,"preJiggleTag","pj"},
        {gParamType_String,"prePreJiggleTag","ppj"},
        {gParamType_String,"jiggleWeightTag","jw"}
    },
    {"FEM"},
});

}