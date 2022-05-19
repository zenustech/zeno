#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>

#include "assimp/scene.h"

#include "Definition.h"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

struct ExtractFBXData : zeno::INode {

    virtual void apply() override {

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto fbxData = get_input<FBXData>("data");

        set_output("vertices", std::move(fbxData->iVertices.clone()));
        set_output("indices", std::move(fbxData->iIndices.clone()));
        set_output("material", std::move(fbxData->iMaterial.clone()));
        set_output("boneOffset", std::move(fbxData->iBoneOffset.clone()));
    }
};
ZENDEFNODE(ExtractFBXData,
           {       /* inputs: */
               {
                   {"FBXData", "data"},
               },  /* outputs: */
               {
                   {"IVertices", "vertices"},
                   {"IIndices", "indices"},
                   {"IMaterial", "material"},
                   {"IBoneOffset", "boneOffset"}
               },  /* params: */
               {

               },  /* category: */
               {
                   "primitive",
               }
           });

struct ExtractMatTexList : zeno::INode {

    virtual void apply() override {
        auto mat = get_input<IMaterial>("material");
        auto key = get_input<zeno::StringObject>("key")->get();

        zeno::log_info(">>>>> Get Key {}", key);
        zeno::log_info(">>>>> Get Mat Name {}", mat->value.at(key).matName);
        auto lo = std::make_shared<zeno::ListObject>();
        lo->arr = mat->value.at(key).getTexList();

        for(auto&l: lo->arr){
            //zeno::log_info("Tex: {}", std::any_cast<std::string>(l));
        }
        zeno::log_info(">>>>> Get TexLen {}", lo->arr.size());

        set_output("texLists", std::move(lo));
    }
};
ZENDEFNODE(ExtractMatTexList,
           {       /* inputs: */
               {
                   {"IMaterial", "material"}, {"string", "key"}
               },  /* outputs: */
               {
                    "texLists"
               },  /* params: */
               {

               },  /* category: */
               {
                   "primitive",
               }
           });