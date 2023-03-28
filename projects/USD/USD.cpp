#include "usdDefinition.h"
#include "usdImporter.h"

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>

struct ZUSDContext : zeno::IObjectClone<ZUSDContext> {
    EUsdImporter usdImporter;
};

struct USDOpenStage : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();

        EUsdImporter usdImporter;
        EUsdOpenStageOptions options;
        options.Identifier = path;
        usdImporter.OpenStage(options);

        auto zusdcontext = std::make_shared<ZUSDContext>();
        zusdcontext->usdImporter = usdImporter;

        set_output("zuc", std::move(zusdcontext));
    }
};
ZENDEFNODE(USDOpenStage,
       {       /* inputs: */
        {
            {"readpath", "path"},
        },  /* outputs: */
        {
            "zuc"
        },  /* params: */
        {
        },  /* category: */
        {
            "USD",
        }
       });


struct USDSimpleTraverse : zeno::INode {
    /*
     *  uv
     *      - point     verts.uv
     *      - vertex    loops.uvs
     *  nrm
     *      - point     verts.nrm
     *      - vertex    loops.nrm   ?
     *  clr
     *      - point     verts.clr
     *      - vertex    loops.clr   ?
     *      .
     *      +
     */
    virtual void apply() override {
        auto zusdcontext = get_input<ZUSDContext>("zuc");
        auto prims = std::make_shared<zeno::ListObject>();

        auto& translationContext = zusdcontext->usdImporter.mTranslationContext;

        auto _xformableTranslator = translationContext.GetTranslatorByName("UsdGeomXformable");
        auto _geomTranslator = translationContext.GetTranslatorByName("UsdGeomMesh");

            std::shared_ptr<EUsdGeomXformableTranslator> xformableTranslator = std::dynamic_pointer_cast<EUsdGeomXformableTranslator>(_xformableTranslator);
            std::shared_ptr<EUsdGeomMeshTranslator> geomTranslator = std::dynamic_pointer_cast<EUsdGeomMeshTranslator>(_geomTranslator);
            auto& XformableImportData = xformableTranslator->ImportContext.PathToMeshImportData;
            auto& GeomMeshImportData = geomTranslator->ImportContext.PathToMeshImportData;
            XformableImportData.insert(GeomMeshImportData.begin(), GeomMeshImportData.end());

            for(auto const& [key, value]: XformableImportData){
                std::cout << "Iter Key " << key << "\n";
                auto prim = std::make_shared<zeno::PrimitiveObject>();

                // Point
                for(auto const& p : value.Vertices){
                    prim->verts.emplace_back(p[0], p[1], p[2]);
                }
                // Polys
                size_t pointer = 0;
                for(auto const& p : value.FaceCounts){
                    prim->polys.emplace_back(pointer, p);
                    pointer += p;
                }
                for(auto const& p : value.FaceIndices){
                    prim->loops.emplace_back(p);
                }
                // Vertex-Color
                prim->loops.add_attr<zeno::vec3f>("clr");
                for(int i=0; i<value.Colors.size(); ++i){
                    auto& e = value.Colors[i];
                    prim->loops.attr<zeno::vec3f>("clr")[i] = {e[0], e[1], e[2]};
                }
                // Vertex-Normal
                prim->loops.add_attr<zeno::vec3f>("nrm");
                for(int i=0; i<value.Normals.size(); ++i){
                    auto& e = value.Normals[i];
                    prim->loops.attr<zeno::vec3f>("nrm")[i] = {e[0], e[1], e[2]};
                }
                // Vertex-UV
                // TODO UV-Sets
                for(auto const&[uv_index, uv_value]: value.UVs){
                    prim->uvs.resize(uv_value.size());
                    for(int i=0; i<uv_value.size(); ++i) {
                        auto& e = uv_value[i];
                        prim->uvs[i] = {e[0], e[1]};
                    }
                    break;
                }
                if(prim->uvs.size()) {
                    prim->loops.add_attr<int>("uvs");
                    for (auto i = 0; i < prim->loops.size(); i++) {
                        prim->loops.attr<int>("uvs")[i] = /*prim->loops[i]*/ i; // Already Indices
                    }
                }

                prims->arr.emplace_back(prim);
            }

        set_output("prims", std::move(prims));
    }
};
ZENDEFNODE(USDSimpleTraverse,
           {       /* inputs: */
            {
                "zuc",
            },  /* outputs: */
            {
                "prims"
            },  /* params: */
            {
            },  /* category: */
            {
                "USD",
            }
           });