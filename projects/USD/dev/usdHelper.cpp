#include "usdHelper.h"

bool IsNearlyEqual(double A, double B, double ErrorTolerance){
    return std::fabs(A - B) <= ErrorTolerance;
}

bool IsNearlyZero(float Value, float ErrorTolerance){
    return std::fabs(Value) <= ErrorTolerance;
}

namespace Helper{
    pxr::GfMatrix4f ETransformToUsdMatrix(ETransform& Transform){
        // Create the identity matrix
        pxr::GfMatrix4f transform(1.0);

        transform *= pxr::GfMatrix4f().SetScale(Transform.GetScale3D());
        transform *= pxr::GfMatrix4f().SetRotate(Transform.GetRotation());
        transform *= pxr::GfMatrix4f().SetTranslate(Transform.GetTranslation());

        return transform;
    }

    double GetRangeFrameKey(double Frame, EAnimationInfo& AnimInfo){
        // The data of vector we stored, that index is start from 0, but the frame actually start that is not the
        // e.g. Total 150
        //      Start Frame 101
        //      End Frame 250
        AnimInfo.StageStartTimeCode;
        AnimInfo.NumBakedFrames;
        auto length = AnimInfo.StageSequenceLengthTimeCodes;
        auto end_timecode = AnimInfo.StageStartTimeCode + length;
        auto start_timecode = AnimInfo.StageStartTimeCode;

        auto index = Frame - start_timecode;
        if(index < 0){
            return 0.0;
        }else if(index > length){
            return length;
        }
        return index;
    }

    bool EvalSkeletalSkin(ESkelImportData& SkelImportData,
                          ESkeletalMeshImportData& SkelMeshImportData,
                          pxr::VtArray<pxr::GfVec3f>& MeshPoints,
                          int32 FrameIndex,
                          pxr::VtArray<pxr::GfVec3f>& OutSkinPoints)
    {
        OutSkinPoints.clear();
        OutSkinPoints.resize(MeshPoints.size());

        pxr::VtArray<float> SkinWeights{};
        SkinWeights.resize(MeshPoints.size());

        for (int infl_index = 0; infl_index < SkelMeshImportData.Influences.size(); ++infl_index) {
            auto &influence = SkelMeshImportData.Influences[infl_index];

            auto point_index = influence.VertexIndex;
            auto bone_index = influence.BoneIndex;
            auto weight = influence.Weight;

            auto xform_rest = SkelImportData.SkeletalAnimData.WorldSpaceBindTransforms[bone_index];
            pxr::GfMatrix4d xform_deform(1.0);
            if(! SkelImportData.SkeletalAnimData.JointAnimTracks.empty()) {
                xform_deform = SkelImportData.SkeletalAnimData.JointAnimTracks[bone_index].UsdTransforms[FrameIndex];
            }else{
                xform_deform = xform_rest;
            }

            auto mesh_pos = MeshPoints[point_index];
            auto invert_xform_rest = xform_rest.GetInverse();
            auto pos = xform_deform.Transform(invert_xform_rest.Transform(mesh_pos));

            OutSkinPoints[point_index] += pos * weight;
            SkinWeights[point_index] += weight;
        }

        // Set Skel Skin data
        for (int i = 0; i < OutSkinPoints.size(); ++i) {
           auto &weight = SkinWeights[i];
           pxr::GfVec3f pos{};
           if (IsNearlyZero(weight)) {
               pos = MeshPoints[i];
           } else {
               pos = OutSkinPoints[i] / weight;
           }
           OutSkinPoints[i] = pos;
        }

        return true;
    }

    bool EvalSkeletalBlendShape(ESkelImportData& SkelImportData,
                                ESkeletalMeshImportData& SkelMeshImportData,
                                int32 FrameIndex,
                                pxr::VtArray<pxr::GfVec3f>& OutDeformPoints)
    {
        auto& blend_shape_paths = SkelImportData.SkeletalAnimData.BlendShapePathsToSkinnedPrims;
        auto& blend_shape_channel = SkelImportData.SkeletalAnimData.BlendShapeChannelOrder;
        auto& blend_shape_weights = SkelImportData.SkeletalAnimData.BlendShapeWeights;

        for(int blend_shape_index = 0; blend_shape_index < SkelMeshImportData.BlendShapeMap.size(); ++blend_shape_index){
            auto blend_shape_map = SkelMeshImportData.BlendShapeMap[blend_shape_index];
            auto usd_blend_shape = SkelImportData.BlendShapes[blend_shape_map.first];
            auto skel_mesh_path = SkelMeshImportData.PrimPath;

            if(std::find(blend_shape_paths.begin(), blend_shape_paths.end(), pxr::SdfPath(skel_mesh_path))
                != blend_shape_paths.end())
            {
                auto elem_index = Helper::GetElemIndex(blend_shape_channel, pxr::TfToken(blend_shape_map.second));
                if(elem_index == -1){
                    std::cout << "ERROR: The elem index is -1\n";
                }
                auto weight = blend_shape_weights[elem_index][FrameIndex];

                if(! IsNearlyZero(weight)) {
                    if(! usd_blend_shape.InBetweens.empty()){
                        for(int inbetween_index=0; inbetween_index<usd_blend_shape.InBetweens.size(); inbetween_index++){
                            auto& inbetween = usd_blend_shape.InBetweens[inbetween_index];
                            auto& inbetween_blend_shape = usd_blend_shape.InBetweenBlendShapes[inbetween.Path];
                            auto& inbetween_weight = inbetween.InbetweenWeight;

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

                                OutDeformPoints[vert.SourceIdx] += {delta_pos[0], delta_pos[1], delta_pos[2]};
                                OutDeformPoints[vert.SourceIdx] += {inbetween_pos[0], inbetween_pos[1], inbetween_pos[2]};
                            }
                        }
                    }else{
                        for(auto & vert : usd_blend_shape.Vertices) {
                            auto delta_pos = vert.PositionDelta * weight;
                            OutDeformPoints[vert.SourceIdx] += {delta_pos[0], delta_pos[1], delta_pos[2]};
                        }
                    }
                }
            }
        }

        return true;
    }
}