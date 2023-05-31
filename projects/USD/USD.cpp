#include "usdDefinition.h"
#include "usdImporter.h"
#include "usdHelper.h"

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

    void imported_mesh_data_to_prim(std::shared_ptr<zeno::PrimitiveObject> prim, EGeomMeshData& value){
        // Point
        for(auto const& p : value.Points){
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
    }

    void imported_skel_data_to_prim(std::shared_ptr<zeno::PrimitiveObject> prim,
                                    pxr::GfMatrix4f& root_matrix,
                                    pxr::VtArray<pxr::GfVec3f>& skin_points,
                                    EGeomMeshData& value){

        for (auto& pos: skin_points) {
            pos = root_matrix.Transform(pos);
            prim->verts.emplace_back(pos[0], pos[1], pos[2]);
        }
        size_t pointer = 0;
        for(auto const& p : value.FaceCounts){
            prim->polys.emplace_back(pointer, p);
            pointer += p;
        }
        for(auto const& p : value.FaceIndices){
            prim->loops.emplace_back(p);
        }
    }

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
        int frameid = -1;
        if (has_input("frameid")) {
            frameid = get_input<zeno::NumericObject>("frameid")->get<int>();
        } else {
            frameid = getGlobalState()->frameid;
        }

        auto zusdcontext = get_input<ZUSDContext>("zuc");

        auto prims = std::make_shared<zeno::ListObject>();
        auto stage = zusdcontext->usdImporter.mStage;
        auto stageinfo = EUsdStageInfo(stage);
        auto &translationContext = zusdcontext->usdImporter.mTranslationContext;
        auto &ImportedData = translationContext.Imported;
        auto _xformableTranslator = translationContext.GetTranslatorByName("UsdGeomXformable");
        auto _geomTranslator = translationContext.GetTranslatorByName("UsdGeomMesh");
        auto xformableTranslator = std::dynamic_pointer_cast<EUsdGeomXformableTranslator>(_xformableTranslator);
        auto geomTranslator = std::dynamic_pointer_cast<EUsdGeomMeshTranslator>(_geomTranslator);

        std::cout << "Imported Sizes Mesh: " << ImportedData.PathToMeshImportData.size() << " FrameMesh: " << ImportedData.PathToFrameToMeshImportData.size() << "\n";
        std::cout << "Imported Sizes Skel: " << ImportedData.PathToSkelImportData.size() << "\n";

        try{

            // Geom Mesh
            for(auto & [key, value]: ImportedData.PathToFrameToMeshImportData){
                auto prim = std::make_shared<zeno::PrimitiveObject>();
                int key_mesh = Helper::GetImportedAnimMeshDataKey(frameid, value);

                if(ImportedData.PathToFrameToTransform.find(key) != ImportedData.PathToFrameToTransform.end()){
                    auto FrameToTransform = ImportedData.PathToFrameToTransform[key];
                    int key_trans = Helper::GetImportedAnimMeshDataKey(frameid, FrameToTransform);
                    value[key_mesh].ApplyTransform(FrameToTransform[key_trans], stageinfo);
                }

                imported_mesh_data_to_prim(prim, value[key_mesh]);
                prims->arr.emplace_back(prim);
            }

            // Geom Mesh Cache
            // We may apply transform for each frame, so we copy the value from import data.
            for(auto [key, value]: ImportedData.PathToMeshImportData){
                auto prim = std::make_shared<zeno::PrimitiveObject>();

                if(ImportedData.PathToFrameToTransform.find(key) != ImportedData.PathToFrameToTransform.end()){
                    auto FrameToTransform = ImportedData.PathToFrameToTransform[key];
                    int kf = Helper::GetImportedAnimMeshDataKey(frameid, FrameToTransform);
                    // apply mesh anim-trans
                    value.ApplyTransform(FrameToTransform[kf], stageinfo);
                }

                imported_mesh_data_to_prim(prim, value);
                prims->arr.emplace_back(prim);
            }

            // Skeletal Mesh
            for(auto  [path, skel_data] : ImportedData.PathToSkelImportData)
            {
                std::cout << "Iter (PathToSkel) Key " << path << "\n";

                auto& anim_info = skel_data.SkeletalAnimData.AnimationInfo;
                auto& root_motions = skel_data.SkeletalAnimData.RootMotionTransforms;

                int frameindex = Helper::GetRangeFrameKey((double)frameid, anim_info);

                // Skel Root Trans
                pxr::GfMatrix4f root_matrix(1.0f);
                if(! root_motions.empty()) {
                    root_matrix = Helper::ETransformToUsdMatrix(root_motions[frameindex]);
                }

                for(auto& [_, skel_mesh_data]: skel_data.SkeletalMeshData)
                {
                    auto prim = std::make_shared<zeno::PrimitiveObject>();
                    auto mesh_points = skel_mesh_data.MeshData.Points;

                    pxr::VtArray<pxr::GfVec3f> skin_points{};
                    Helper::EvalSkeletalSkin(skel_data, skel_mesh_data, mesh_points, frameindex, skin_points);

                    // Skeletal BlendShape
                    Helper::EvalSkeletalBlendShape(skel_data, skel_mesh_data, frameindex, skin_points);

                    imported_skel_data_to_prim(prim, root_matrix, skin_points, skel_mesh_data.MeshData);
                    prims->arr.emplace_back(prim);
                }
            }

        } catch (const std::exception& e) {
            // Handle the exception
            std::cerr << "Exception caught: " << e.what() << '\n';
        } catch (...) {
            // Handle any other exception types
            std::cerr << "Unknown exception caught\n";
        }

        set_output("prims", std::move(prims));
    }
};
ZENDEFNODE(USDSimpleTraverse,
           {       /* inputs: */
            {
                "zuc", "frameid",
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