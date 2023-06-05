#include "usdDefinition.h"
#include "usdUtilities.h"

#include <pxr/base/gf/rotation.h>

bool EUsdLayerTimeInfo::IsAnimated() {
    return !IsNearlyEqual( StartTimeCode, EndTimeCode );
}

bool EUsdBlendShape::IsValid() const {
    return Vertices.size() > 0;
}

EUsdStageInfo::EUsdStageInfo(const pxr::UsdStageRefPtr& Stage){
    pxr::TfToken UsdStageAxis = GetUsdStageUpAxis(Stage);

    if (UsdStageAxis == pxr::UsdGeomTokens->y)
        UpAxis = EUsdUpAxis::YAxis;
    else
        UpAxis = EUsdUpAxis::ZAxis;

    // FIXEM We set this value as the default, in order to avoid any matrix scaling
    //MetersPerUnit = GetUsdStageMetersPerUnit(Stage);

    //ED_COUT << "Meters PerUnit " << MetersPerUnit << "\n";
}

std::ostream& operator<<(std::ostream& out, ETransform const& eTransform){
    out << " - Transformation " << eTransform.Translation << " / " << eTransform.Rotation << " / "
        << eTransform.Scale3D;
    return out;
}

std::ostream& operator<<(std::ostream& out, EBone const& eBone){
    out << " - EBone " << eBone.Name << " / " << eBone.NumChildren
        << " / " << eBone.ParentIndex << " - Trans " << eBone.BonePos.Transform;
    return out;
}

std::ostream &operator<<(std::ostream &out, const EUsdPrimMaterialSlot &primMaterialSlot) {
    out << " - EUsdPrimMaterialSlot " << primMaterialSlot.MaterialSource << " AssignmentType "
        << primMaterialSlot.AssignmentType;
    return out;
}

std::ostream& operator<<(std::ostream& out, ETextureParameterValue const& textureParameterValue){
    out << " - ETextureParameterValue " << textureParameterValue.TexturePath << " UVIndex "
        << textureParameterValue.UVIndex << " OutputIndex " << textureParameterValue.OutputIndex;
    return out;
}

std::ostream& operator<<(std::ostream& out, EUsdPrimMaterialAssignmentInfo const& primMaterialAssignmentInfo){
    out << " - EUsdPrimMaterialAssignmentInfo " << primMaterialAssignmentInfo.Slots << " Indices ";
    //    << primMaterialAssignmentInfo.MaterialIndices;
    return out;
}

std::ostream& operator<<(std::ostream& out, EUVSet const& uvSet){
    out << " - EUVSet UVSetIndex " << uvSet.UVSetIndex << " InterpType " << uvSet.InterpType;
    return out;
}

std::ostream& operator<<(std::ostream& out, ERawBoneInfluence const& rawBoneInfluence){
    out << "(" << rawBoneInfluence.VertexIndex << " " << rawBoneInfluence.BoneIndex
        << " " << rawBoneInfluence.Weight << ")";
    return out;
}

std::ostream& operator<<(std::ostream& out, ESkeletalAnimData const& d){
    out << "Skeletal Anim Data Sizes: " << d.WorldSpaceRestTransforms.size() << " " << d.WorldSpaceBindTransforms.size()
        << " " << d.JointLocalRestTransforms.size() << " " << d.MeshToSkeletonRestPose.size();
    return out;
}

std::ostream& operator<<(std::ostream& out, ESkeletalMeshImportData const& skeletalMeshImportData){
    auto& O = skeletalMeshImportData;
    //out << " - ESkeletalMeshImportData Influences " << O.Influences << " Size " << O.Influences.size()
    //    << "\n"
    // TODO print info
    //out    << " Has " << O.bHasNormals << " " << O.bHasTangents << " " << O.bHasVertexColors
    //    << " Sizes "<<O.Materials.size()<<" "<<O.Points.size()<<" "<<O.Faces.size()<<" "<<O.RefBonesBinary.size()
    //    <<" "<<O.MeshInfos.size()<<" " <<O.PointToRawMap.size()<<" " <<O.Normals.size()<<" " <<O.Colors.size()
    //    <<" "<<O.UVs.size()<<" "<<O.FaceCounts.size()<<" " <<O.FaceIndices.size();
    return out;
}

std::ostream& operator<<(std::ostream& out, EUsdBlendShape const& usdBlendShape){
    out << " - EUsdBlendShape Name " << usdBlendShape.Name << " AuthoredTangents " << usdBlendShape.bHasAuthoredTangents
        << " ArrayOfMorphTargetDelta " << usdBlendShape.Vertices << " Size " << usdBlendShape.Vertices.size()
        << " ArrayOfInbetween" << usdBlendShape.InBetweens << " Size " << usdBlendShape.InBetweens.size();
    return out;
}

std::ostream& operator<<(std::ostream& out, EMorphTargetDelta const& morphTargetDelta){
    out << "(" << morphTargetDelta.PositionDelta << " " << morphTargetDelta.TangentZDelta
        << " " << morphTargetDelta.SourceIdx << ")";
    return out;
}

std::ostream& operator<<(std::ostream& out, EUsdBlendShapeInbetween const& blendShapeInbetween){
    out << "(" << blendShapeInbetween.Name << " " << blendShapeInbetween.InbetweenWeight << ")";
    return out;
}

std::ostream& operator<<(std::ostream& out, ERawAnimSequenceTrack const& animSequenceTrack){
    //out << "PosKeys: " << animSequenceTrack.PosKeys << "\n";
    //out << "RotKeys: " << animSequenceTrack.RotKeys << "\n";
    //out << "ScaleKeys: " << animSequenceTrack.ScaleKeys << "\n";
    //out << "Transform: " << animSequenceTrack.Transforms << "\n";

    //out << "PosKeys: Size " << animSequenceTrack.PosKeys.size() << "\n";
    //out << "RotKeys: Size " << animSequenceTrack.RotKeys.size() << "\n";
    //out << "ScaleKeys: Size " << animSequenceTrack.ScaleKeys.size() << "\n";
    return out;
}

ETransform::ETransform(pxr::GfVec3f InRotation)
{
//    pxr::GfRotation aRotation(InRotation, 0);
//    pxr::GfQuatf InQuatRotation(aRotation.GetQuat());
//    // Rotation = InRotation
//    Rotation = InQuatRotation;
//    // Translation = {0,0,0,0)
//    Translation = {0, 0, 0, 0};
//    // Scale3D = {1,1,1,0);
//    Scale3D = {1, 1, 1, 0};
}

ETransform::ETransform(pxr::GfQuatf InRotation, pxr::GfVec3f InTranslation, pxr::GfVec3f InScale3D)
{
    Rotation = InRotation;
    Translation = {InTranslation[0], InTranslation[1], InTranslation[2], 0.0f};
    Scale3D = {InScale3D[0], InScale3D[1], InScale3D[2], 0.0f};
}

ETransform::ETransform(pxr::GfMatrix4f InMatrix){
    SetFromMatrix(InMatrix);
}

ETransform::ETransform(){
    // Rotation = {0,0,0,1)
//    Rotation = {0, 0, 0, 1};
    Rotation = {1, 0, 0, 0};  // USD
    // Translation = {0,0,0,0)
    Translation = {0, 0, 0, 0};
    // Scale3D = {1,1,1,0);
    Scale3D = {1, 1, 1, 0};
}

void ETransform::ScaleTranslation(float InScale){
    ScaleTranslation(pxr::GfVec3f(InScale));
}

void ETransform::ScaleTranslation(pxr::GfVec3f InScale3D){
    pxr::GfVec4f VectorInScale3D(InScale3D[0], InScale3D[1], InScale3D[2], 0);
    Translation = pxr::GfCompMult(Translation, VectorInScale3D);
}

pxr::GfVec3f ETransform::TransformPosition(const pxr::GfVec3f& V){
    pxr::GfVec4f InputVectorW0 = pxr::GfVec4f(V[0], V[1], V[2], 0.0);

    //Transform using QST is following
    //QST(P) = Q.Rotate(S*P) + T where Q = quaternion, S = scale, T = translation

    //RotatedVec = Q.Rotate(Scale*V.X, Scale*V.Y, Scale*V.Z, 0.f)
    auto ScaledVec = pxr::GfCompMult(Scale3D, InputVectorW0);
    auto _RotatedVec = Rotation.Transform(pxr::GfVec3f(ScaledVec[0], ScaledVec[1], ScaledVec[2]));
    auto RotatedVec = pxr::GfVec4f(_RotatedVec[0], _RotatedVec[1], _RotatedVec[2], 0);

    auto TranslatedVec = RotatedVec + Translation;
    return pxr::GfVec3f(TranslatedVec[0], TranslatedVec[1], TranslatedVec[2]);
}

void ETransform::Multiply(ETransform* OutTransform, const ETransform* A, const ETransform* B){
    //	When Q = quaternion, S = single scalar scale, and T = translation
    //	QST(A) = Q(A), S(A), T(A), and QST(B) = Q(B), S(B), T(B)

    //	QST (AxB)

    // QST(A) = Q(A)*S(A)*P*-Q(A) + T(A)
    // QST(AxB) = Q(B)*S(B)*QST(A)*-Q(B) + T(B)
    // QST(AxB) = Q(B)*S(B)*[Q(A)*S(A)*P*-Q(A) + T(A)]*-Q(B) + T(B)
    // QST(AxB) = Q(B)*S(B)*Q(A)*S(A)*P*-Q(A)*-Q(B) + Q(B)*S(B)*T(A)*-Q(B) + T(B)
    // QST(AxB) = [Q(B)*Q(A)]*[S(B)*S(A)]*P*-[Q(B)*Q(A)] + Q(B)*S(B)*T(A)*-Q(B) + T(B)

    //	Q(AxB) = Q(B)*Q(A)
    //	S(AxB) = S(A)*S(B)
    //	T(AxB) = Q(B)*S(B)*T(A)*-Q(B) + T(B)


    auto QuatA = A->Rotation;
    auto QuatB = B->Rotation;
    auto TranslateA = A->Translation;
    auto TranslateB = B->Translation;
    auto ScaleA = A->Scale3D;
    auto ScaleB = B->Scale3D;

    // RotationResult = B.Rotation * A.Rotation
    OutTransform->Rotation = QuatB * QuatA;

    // TranslateResult = B.Rotate(B.Scale * A.Translation) + B.Translate
    auto ScaledTransA = pxr::GfCompMult(TranslateA, ScaleB);
    auto _RotatedTranslate = QuatB.Transform(pxr::GfVec3f(ScaledTransA[0], ScaledTransA[1], ScaledTransA[2]));
    auto RotatedTranslate = pxr::GfVec4f(_RotatedTranslate[0], _RotatedTranslate[1], _RotatedTranslate[2], 0);
    OutTransform->Translation = RotatedTranslate + TranslateB;

    // ScaleResult = Scale.B * Scale.A
    OutTransform->Scale3D = GfCompMult(ScaleA, ScaleB);
}

ETransform ETransform::operator*(const ETransform& Other){
    ETransform Output;
    Multiply(&Output, this, &Other);
    return Output;
}

void ETransform::SetFromMatrix(pxr::GfMatrix4f InMatrix){
    pxr::GfMatrix4f M = InMatrix;

    pxr::GfVec3f InScale = ExtractScaling<float>(M);
    Scale3D = {InScale[0], InScale[1], InScale[2], 0};

    // If there is negative scaling going on, we handle that here
    if(InMatrix.GetDeterminant() < 0.f)
    {
        // Assume it is along X and modify transform accordingly.
        // It doesn't actually matter which axis we choose, the 'appearance' will be the same
        Scale3D = pxr::GfCompMult(Scale3D, pxr::GfVec4f(-1, 1, 1, 1));
        SetAxis(M, 0, -GetScaledAxis<float>(M, EAxis::X));
    }

    pxr::GfQuatf InRotation = M.ExtractRotationQuat();
    Rotation = InRotation;
    pxr::GfVec3f InTranslation = GetOrigin(InMatrix);
    Translation = {InTranslation[0], InTranslation[1], InTranslation[2], 0};

    // Normalize rotation
    Rotation = InRotation.GetNormalized();
}

pxr::GfVec3f ETransform::TransformVector(const pxr::GfVec3f &V) {
    pxr::GfVec4f InputVectorW0 = pxr::GfVec4f(V[0], V[1], V[2], 0.0);

    //RotatedVec = Q.Rotate(Scale*V.X, Scale*V.Y, Scale*V.Z, 0.f)
    auto ScaledVec = pxr::GfCompMult(Scale3D, InputVectorW0);
    auto _RotatedVec = Rotation.Transform(pxr::GfVec3f(ScaledVec[0], ScaledVec[1], ScaledVec[2]));
    auto RotatedVec = pxr::GfVec4f(_RotatedVec[0], _RotatedVec[1], _RotatedVec[2], 0);

    return pxr::GfVec3f(RotatedVec[0], RotatedVec[1], RotatedVec[2]);
}

pxr::GfVec3f ETransform::GetTranslation() {
    return pxr::GfVec3f(Translation[0], Translation[1], Translation[2]);
}

pxr::GfQuatf ETransform::GetRotation() {
    return Rotation;
}

pxr::GfVec3f ETransform::GetScale3D() {
    return pxr::GfVec3f(Scale3D[0], Scale3D[1], Scale3D[2]);
}


std::ostream& operator<<(std::ostream& out, const ETextureWrapMode value){
    static std::map<ETextureWrapMode, std::string> strings;
    if (strings.size() == 0){
#define INSERT_ELEMENT(p) strings[p] = #p
        INSERT_ELEMENT(ETextureWrapMode::TW_Clamp);
        INSERT_ELEMENT(ETextureWrapMode::TW_Mirror);
        INSERT_ELEMENT(ETextureWrapMode::TW_Black);
        INSERT_ELEMENT(ETextureWrapMode::TW_Repeat);
#undef INSERT_ELEMENT
    }

    return out << strings[value];
}

std::ostream& operator<<(std::ostream& out, const EUsdUpAxis value){
    static std::map<EUsdUpAxis, std::string> strings;
    if (strings.size() == 0){
#define INSERT_ELEMENT(p) strings[p] = #p
        INSERT_ELEMENT(EUsdUpAxis::YAxis);
        INSERT_ELEMENT(EUsdUpAxis::ZAxis);
#undef INSERT_ELEMENT
    }

    return out << strings[value];
}

std::ostream& operator<<(std::ostream& out, const EPrimAssignmentType value){
    static std::map<EPrimAssignmentType, std::string> strings;
    if (strings.size() == 0){
#define INSERT_ELEMENT(p) strings[p] = #p
        INSERT_ELEMENT(EPrimAssignmentType::None);
        INSERT_ELEMENT(EPrimAssignmentType::DisplayColor);
        INSERT_ELEMENT(EPrimAssignmentType::MaterialPrim);
#undef INSERT_ELEMENT
    }

    return out << strings[value];
}

void EGeomMeshData::ReserveNewVertices(const int32 NumVertices){
    Points.resize(Points.size() + NumVertices);
}

//void EGeomMeshData::ReserveNewPolygons(const int32 NumPolygons){
//    Faces.resize(Faces.size() + NumPolygons);
//}

EPolygonID EGeomMeshData::CreatePolygon(pxr::VtArray<EVertexInstanceID> VertexInstanceIDs, pxr::VtArray<EEdgeID>* OutEdgeIDs){
    auto PolygonID = Faces.size();

    const int32 NumVertices = VertexInstanceIDs.size();
    //ED_COUT << "==== " << NumVertices << "\n";
    for (int32 Index = 0; Index < NumVertices; ++Index)
    {
        const EVertexInstanceID ThisVertexInstanceID = VertexInstanceIDs[Index];
        const EVertexInstanceID NextVertexInstanceID = VertexInstanceIDs[(Index + 1 == NumVertices) ? 0 : Index + 1];
        const EVertexID ThisVertexID = VertexInstances[ThisVertexInstanceID.IDValue];
        const EVertexID NextVertexID = VertexInstances[NextVertexInstanceID.IDValue];
        //ED_COUT << "Vertex ID " << ThisVertexID.IDValue << " " << NextVertexID.IDValue << "\n";

        // Handle Connected Edge
        bool Paired = false;
        for(auto const& PairEdge: Edges){
            if((PairEdge.first == ThisVertexID && PairEdge.second == NextVertexID)
                || (PairEdge.first == NextVertexID && PairEdge.second == ThisVertexID))
            {
                Paired = true;
            }
        }
        if(! Paired) {
            auto EdgeID = CreateEdge(ThisVertexID, NextVertexID);
            if (OutEdgeIDs)
                OutEdgeIDs->emplace_back(EdgeID);
        }
    }

    Faces.emplace_back();
    Faces[PolygonID] = PolygonID;

    return PolygonID;
}

EVertexInstanceID EGeomMeshData::CreateVertexInstance(const EVertexID VertexID) {
    auto VertexInstanceID = VertexInstances.size();   // VertexInstanceID - e.g. 1,2,3,4,...
    VertexInstances.emplace_back();                         //                         | | | |
    VertexInstances[VertexInstanceID] = VertexID;           // VertexID         - e.g. 0,1,3,2,...
    return VertexInstanceID;
}

EEdgeID EGeomMeshData::CreateEdge(const EVertexID VertexID0, const EVertexID VertexID1){
    auto EdgeID = Edges.size();
    Edges.emplace_back();
    Edges[EdgeID] = {VertexID0, VertexID1};
    return EdgeID;
}

std::ostream& operator<<(std::ostream& out, EGeomMeshData const& geomMeshImportData){
    out << " - GeomMesh ImportData Sizes: Verts " << geomMeshImportData.Points.size() << " VertIns " << geomMeshImportData.VertexInstances.size()
        << " Norms " << geomMeshImportData.Normals.size() << " Colors " << geomMeshImportData.Colors.size() << " UVs " << geomMeshImportData.UVs.size()
        << " Polys " << geomMeshImportData.Faces.size() << " Edges " << geomMeshImportData.Edges.size()
        << " Counts " << geomMeshImportData.FaceCounts.size() << " Indices " << geomMeshImportData.FaceIndices.size() << "\n";

    for(auto const&[a, b]: geomMeshImportData.UVs){
        out << "   - uv " << a << " size " << b.size() << "\n";
    }

    for(int i=0; i<geomMeshImportData.Edges.size(); ++i){
        auto edge = geomMeshImportData.Edges[i];
        out << "   - edge " << i << " - " << edge.first.IDValue << " " << edge.second.IDValue << "\n";
    }

    return out;
}

bool EGeomMeshData::ApplyTransform(ETransform Transform, EUsdStageInfo StageInfo) {
    if (Normals.size() > 0)
    {
        for (int32 NormalIndex = 0; NormalIndex < Normals.size(); ++NormalIndex) {
            const pxr::GfVec3f &Normal = Normals[NormalIndex];
            // FIXEM
            //Transform.TransformVector(ConvertVector(StageInfo, Normal));
            Transform.TransformVector(Normal);

            Normals[NormalIndex] = Normal;
        }
    }

    if (Points.size() > 0)
    {
        for (int32 PointIndex = 0; PointIndex < Points.size(); ++PointIndex)
        {
            const pxr::GfVec3f Point = Points[PointIndex];
            // FIXEM
            //auto Position = Transform.TransformPosition(ConvertVector(StageInfo, Point));
            auto Position = Transform.TransformPosition(Point);

            Points[PointIndex] = Position;
        }
    }

    return true;
}
