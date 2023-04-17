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
        auto skel_iter = get_input2<int>("skel_iter");
        auto pose_model = get_input2<int>("pose_model");
        auto blend_shape_iter = get_input2<int>("blend_shape_iter");

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

                // Compute Bone Global Transform
                //pxr::VtArray<ETransform> world_skel_transform{};
                //auto size_of_bones = skel_data.SkeletonBones.size();
                //world_skel_transform.resize(size_of_bones);
                //for(int i=0; i<size_of_bones; ++i){
                //    auto& skelbone = skel_data.SkeletonBones[i];
                //    world_skel_transform[i] = Helper::RecursiveGetBoneTrans(skelbone, skel_data.SkeletonBones, ETransform());
                //}

                auto& anim_info = skel_data.SkeletalAnimData.AnimationInfo;
                auto& world_bind_transform = skel_data.SkeletalAnimData.WorldSpaceBindTransforms;
                auto& local_bind_transform = skel_data.SkeletalAnimData.JointLocalRestTransforms;
                auto& anim_tracks = skel_data.SkeletalAnimData.JointAnimTracks;
                auto& root_motions = skel_data.SkeletalAnimData.RootMotionTransforms;
                auto& paths = skel_data.SkeletalAnimData.BlendShapePathsToSkinnedPrims;
                auto& channel_order = skel_data.SkeletalAnimData.BlendShapeChannelOrder;
                auto& blend_shape_weights = skel_data.SkeletalAnimData.BlendShapeWeights;

                int frameindex = Helper::GetRangeFrameKey((double)frameid, anim_info);

                int skel_iter_count = -1;
                for(auto& [_, skel_mesh_data]: skel_data.SkeletalMeshData)
                {
                    if(skel_iter != -1) {
                        skel_iter_count++;
                        if (skel_iter_count != skel_iter) {
                            continue;
                        }
                    }

                    auto prim = std::make_shared<zeno::PrimitiveObject>();
                    auto mesh_points = skel_mesh_data.Points;

                    //std::cout << "Path: " << skel_mesh_data.PrimPath << " Point Size " << mesh_points.size() << "\n";

                    pxr::VtArray<pxr::GfVec3f> skin_points{};
                    pxr::VtArray<float> skin_weight{};
                    skin_points.resize(mesh_points.size());
                    skin_weight.resize(mesh_points.size());

                    // Skeletal Skin
                    for (int infl_index = 0; infl_index < skel_mesh_data.Influences.size(); ++infl_index) {
                        auto &influence = skel_mesh_data.Influences[infl_index];

                        auto point_index = influence.VertexIndex;
                        auto bone_index = influence.BoneIndex;
                        auto weight = influence.Weight;

                        auto xform_rest = world_bind_transform[bone_index];
                        pxr::GfMatrix4d xform_deform(1.0);
                        if(! anim_tracks.empty()) {
                            xform_deform = anim_tracks[bone_index].UsdTransforms[frameindex];
                        }else{
                            xform_deform = xform_rest;
                        }

                        auto mesh_pos = mesh_points[point_index];
                        auto invert_xform_rest = xform_rest.GetInverse();
                        auto pos = xform_deform.Transform(invert_xform_rest.Transform(mesh_pos));

                        skin_points[point_index] += pos * weight;
                        skin_weight[point_index] += weight;
                    }

                    // Skel Root Trans
                    pxr::GfMatrix4f root_matrix(1.0f);
                    if(! root_motions.empty()) {
                        root_matrix = Helper::ETransformToUsdMatrix(root_motions[frameindex]);
                    }

                    // Set Skel Skin data
                    for (int i = 0; i < skin_points.size(); ++i) {
                        auto &weight = skin_weight[i];
                        pxr::GfVec3f pos{};
                        if (IsNearlyZero(weight)) {
                            pos = mesh_points[i];
                        } else {
                            pos = skin_points[i] / weight;
                        }
                        skin_points[i] = pos;
                    }


                    pxr::VtArray<pxr::GfVec3f> deform_points = skin_points;
                    auto blend_shape_iter_count = -1;
                    // Skeletal BlendShape
                    //std::cout << "Blend Shape: Size " << skel_mesh_data.BlendShapeMap.size() << " " << blend_shape_weights.size() << "\n";
                    for(int blend_shape_index = 0; blend_shape_index < skel_mesh_data.BlendShapeMap.size(); ++blend_shape_index){

                        //if(blend_shape_iter != -1) {
                        //    blend_shape_iter_count++;
                        //    if (blend_shape_iter_count != blend_shape_iter) {
                        //        continue;
                        //    }
                        //}

                        auto blend_shape_map = skel_mesh_data.BlendShapeMap[blend_shape_index];
                        auto usd_blend_shape = skel_data.BlendShapes[blend_shape_map.first];
                        auto name = usd_blend_shape.Name;
                        auto skel_mesh_path = skel_mesh_data.PrimPath;

                        //std::cout << "Blend Shape: Name " << name << "\n";
                        if(std::find(paths.begin(), paths.end(), pxr::SdfPath(skel_mesh_path)) != paths.end()){
                            auto elem_index = Helper::GetElemIndex(channel_order, pxr::TfToken(blend_shape_map.second));
                            if(elem_index == -1){
                                std::cout << "ERROR: The elem index is -1\n";
                            }
                            auto weight = blend_shape_weights[elem_index][frameindex];

                            if(! IsNearlyZero(weight)) {

                                if(blend_shape_iter != -1) {
                                    blend_shape_iter_count++;
                                    if (blend_shape_iter_count != blend_shape_iter) {
                                        continue;
                                    }
                                }

                                if(! usd_blend_shape.InBetweens.empty()){
                                    //std::cout << "Blend Shape: InBetween Size: " << usd_blend_shape.InBetweens.size() << "\n";
                                    for(int inbetween_index=0; inbetween_index<usd_blend_shape.InBetweens.size(); inbetween_index++){
                                        auto& inbetween = usd_blend_shape.InBetweens[inbetween_index];
                                        auto& inbetween_blend_shape = usd_blend_shape.InBetweenBlendShapes[inbetween.Path];
                                        auto& inbetween_weight = inbetween.InbetweenWeight;
                                        //std::cout << "InBetween: Index " << inbetween_index << " Path " << inbetween.Path << " Weight " << inbetween.InbetweenWeight << "\n";

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
                                    //std::cout << "Blend Shape: Index " << blend_shape_index << " Vertices Size: " << usd_blend_shape.Vertices.size() << "\n";
                                    for (int i = 0; i < usd_blend_shape.Vertices.size(); i++) {
                                        auto &vert = usd_blend_shape.Vertices[i];
                                        auto delta_pos = vert.PositionDelta * weight;
                                        deform_points[vert.SourceIdx] += {delta_pos[0], delta_pos[1], delta_pos[2]};
                                    }
                                }
                            }else{
                                //std::cout << " Blend Shape: IsNearlyZero " << name << "\n";
                            }
                        }else{
                            //std::cout << "ERROR: Not found blend shape " << skel_mesh_path << "\n";
                        }
                    }

                    if(pose_model == -1){
                        for (auto& pos: deform_points) {
                            pos = root_matrix.Transform(pos);
                            prim->verts.emplace_back(pos[0], pos[1], pos[2]);
                        }
                    }else if(pose_model == 0){
                        for (auto& pos: mesh_points) {
                            pos = root_matrix.Transform(pos);
                            prim->verts.emplace_back(pos[0], pos[1], pos[2]);
                        }
                    }else if(pose_model == 1){
                        for (auto& pos: skin_points) {
                            pos = root_matrix.Transform(pos);
                            prim->verts.emplace_back(pos[0], pos[1], pos[2]);
                        }
                    }else if(pose_model == 2){
                        for (auto& pos: deform_points) {
                            prim->verts.emplace_back(pos[0], pos[1], pos[2]);
                        }
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
                {"int", "skel_iter", "-1"},
                {"int", "pose_model", "-1"},
                {"int", "blend_shape_iter", "-1"}
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