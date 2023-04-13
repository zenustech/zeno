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

    void imported_mesh_data_to_prim(std::shared_ptr<zeno::PrimitiveObject> prim, EGeomMeshImportData& value){
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
    }

    template <typename T>
    int get_imported_anim_mesh_data_key(int frameid, std::map<int, T>& value){
        if(value.find(frameid) != value.end()){
            return frameid;
        }else{
            std::vector<int> keys;
            typename std::map<int, T>::iterator it;
            for (it=value.begin(); it!=value.end(); ++it) {
                keys.push_back(it->first);
            }
            std::sort(keys.begin(), keys.end());
            for(int key : keys){
                if(frameid <= key){
                    return key;
                }
            }
            return keys.back();
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

            // Geom Mesh
            for(auto & [key, value]: ImportedData.PathToFrameToMeshImportData){
                auto prim = std::make_shared<zeno::PrimitiveObject>();
                int kf = get_imported_anim_mesh_data_key(frameid, value);
                imported_mesh_data_to_prim(prim, value[kf]);
                prims->arr.emplace_back(prim);
            }

            // Geom Mesh Cache
            // We may apply transform for each frame, so we copy the value from import data.
            for(auto [key, value]: ImportedData.PathToMeshImportData){
                //std::cout << "Iter (PathToMesh) Key " << key << "\n";
                auto prim = std::make_shared<zeno::PrimitiveObject>();

                if(ImportedData.PathToFrameToTransform.find(key) != ImportedData.PathToFrameToTransform.end()){
                    std::cout << "Apply (Transform) Key " << key << "\n";
                    auto FrameToTransform = ImportedData.PathToFrameToTransform[key];
                    int kf = get_imported_anim_mesh_data_key(frameid, FrameToTransform);

                    value.ApplyTransform(FrameToTransform[kf], stageinfo);
                }

                imported_mesh_data_to_prim(prim, value);
                prims->arr.emplace_back(prim);
            }

            // Skeletal
            for(auto & [key, value] : ImportedData.PathToSkelImportData){
                std::cout << "Iter (PathToSkel) Key " << key << "\n";

                // Compute Bone Global Transform
                //pxr::VtArray<ETransform> world_skel_transform{};
                //auto size_of_bones = value.SkeletonBones.size();
                //world_skel_transform.resize(size_of_bones);
                //for(int i=0; i<size_of_bones; ++i){
                //    auto& skelbone = value.SkeletonBones[i];
                //    world_skel_transform[i] = Helper::RecursiveGetBoneTrans(skelbone, value.SkeletonBones, ETransform());
                //}

                auto prim = std::make_shared<zeno::PrimitiveObject>();

                // Only LOD 0
                auto& skel_mesh_data = value.SkeletalMeshData[0];
                auto& points = skel_mesh_data.Points;
                auto& world_bind_transform = value.SkeletalAnimData.WorldSpaceBindTransforms;
                auto& local_bind_transform = value.SkeletalAnimData.JointLocalRestTransforms;
                auto& anim_tracks = value.SkeletalAnimData.JointAnimTracks;
                auto& root_motions = value.SkeletalAnimData.RootMotionTransforms;

                pxr::VtArray<pxr::GfVec3f> deform_points{};
                pxr::VtArray<float> deform_weight{};
                deform_points.resize(points.size());
                deform_weight.resize(points.size());

                for(int infl_index=0; infl_index<skel_mesh_data.Influences.size(); ++infl_index){
                    auto& influence = skel_mesh_data.Influences[infl_index];

                    auto point_index = influence.VertexIndex;
                    auto bone_index = influence.BoneIndex;
                    auto weight = influence.Weight;

                    auto xform_deform = anim_tracks[bone_index].UsdTransforms[frameid];
                    auto xform_rest = world_bind_transform[bone_index];

                    auto mesh_pos = points[point_index];
                    auto invert_xform_rest = xform_rest.GetInverse();
                    auto pos = xform_deform.Transform(invert_xform_rest.Transform(mesh_pos));

                    deform_points[point_index] += pos * weight;
                    deform_weight[point_index] += weight;
                }

                // BlendShape
                //   Compute InBetween BlendShape
                //    `inbetween_vertex = (1 - weight) * key_shape_1_vertex + weight * key_shape_2_vertex`
                for(auto& [blend_shape_path, usd_blend_shape] :value.BlendShapes){
                    auto name = usd_blend_shape.Name;
                    auto paths = value.SkeletalAnimData.BlendShapePathsToSkinnedPrims;
                    auto skel_mesh_path = skel_mesh_data.PrimPath;
                    if(std::find(paths.begin(), paths.end(), pxr::SdfPath(skel_mesh_path)) != paths.end()){
                        auto index = usd_blend_shape.BlendShapeIndex;
                        auto weight = value.SkeletalAnimData.BlendShapeWeights[index][frameid];

                        std::cout << "Blend Shape: Path " << skel_mesh_path << " " << name << " Weight " << weight << "\n";

                        if(! usd_blend_shape.InBetweens.empty()){
                            for(int inbetween_index=0; inbetween_index<usd_blend_shape.InBetweens.size(); inbetween_index++){
                                auto& inbetween = usd_blend_shape.InBetweens[inbetween_index];
                                auto& inbetween_blend_shape = usd_blend_shape.InBetweenBlendShapes[inbetween.Path];
                                auto& inbetween_weight = inbetween.InbetweenWeight;
                                std::cout << "InBetween: Index " << inbetween_index << " Path " << inbetween.Path << " Weight " << inbetween.InbetweenWeight << "\n";

                                float w1 = weight;
                                float w2 = inbetween_weight;
                                float _w1 = w1/w2;
                                float _w2 = (w1-w2)/w2;

                                if(w1 < w2){
                                    _w2 = 0.0f;
                                }else{
                                    _w1 = 1.0f - (w1-w2)/w2;
                                }

                                for(int vertice_index=0; vertice_index<usd_blend_shape.Vertices.size(); vertice_index++){
                                    auto& vert = usd_blend_shape.Vertices[vertice_index];
                                    auto delta_pos = vert.PositionDelta * _w2;
                                    auto inbetween_pos = inbetween_blend_shape.Vertices[vertice_index].PositionDelta * _w1;

                                    deform_points[vert.SourceIdx] += {delta_pos[0], delta_pos[1], delta_pos[2]};
                                    deform_points[vert.SourceIdx] += {inbetween_pos[0], inbetween_pos[1], inbetween_pos[2]};
                                }
                            }
                        }else{
                            for(int i=0; i<usd_blend_shape.Vertices.size(); i++){
                                auto& vert = usd_blend_shape.Vertices[i];
                                auto delta_pos = vert.PositionDelta * weight;
                                deform_points[vert.SourceIdx] += {delta_pos[0], delta_pos[1], delta_pos[2]};
                            }
                        }
                    }else{
                        std::cout << "ERROR: Not found blend shape " << skel_mesh_path << "\n";
                    }
                }

                // Skel Root Trans
                auto root_matrix = Helper::ETransformToUsdMatrix(root_motions[frameid]);

                // Set prim data
                for(int i=0; i<deform_points.size(); ++i){
                    auto& weight = deform_weight[i];
                    pxr::GfVec3f pos{};
                    if(IsNearlyZero(weight)){
                         pos = points[i];
                    }else{
                         pos = deform_points[i] / weight;
                    }
                    pos = root_matrix.Transform(pos);
                    prim->verts.emplace_back(pos[0], pos[1], pos[2]);
                }
                size_t pointer = 0;
                for(auto const& p : skel_mesh_data.FaceCounts){
                    prim->polys.emplace_back(pointer, p);
                    pointer += p;
                }
                for(auto const& p : skel_mesh_data.FaceIndices){
                    prim->loops.emplace_back(p);
                }
                prims->arr.emplace_back(prim);
            }

        set_output("prims", std::move(prims));
    }
};
ZENDEFNODE(USDSimpleTraverse,
           {       /* inputs: */
            {
                "zuc", "frameid"
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