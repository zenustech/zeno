#ifndef YE_USDDEFINITION_H
#define YE_USDDEFINITION_H

#include <string>
#include <iostream>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/tokens.h>

/*
 * FIXME:
 *  1. HumanFemale/HumanFemale.walk.usd Not walking normally, it's influence by ED_METERS_PER_UNIT
 */

#define ED_MAX_TEXCOORDS                4
// FIXEM the unit
#define ED_METERS_PER_UNIT              1.0f
#define ED_RIGHT_COORDINATE             true
#define ED_MAX_INFLUENCES_PER_STREAM    4       // Max number of bone influences that a single skinned vert can have per vertex stream.
#define ED_DEFAULT_SAMPLERATE           30.0f
#define ED_MINIMUM_ANIMATION_LENGTH	    (1/ED_DEFAULT_SAMPLERATE)

#define ED_EMPTY
//#define ED_DEFINE_COUT

#ifdef ED_DEFINE_COUT
    #define ED_COUT             std::cout
    #define ED_CERR             std::cerr
#else
    #define ED_COUT             /ED_EMPTY/
    #define ED_CERR             /ED_EMPTY/
#endif

struct EUsdStageInfo;
struct EUsdBlendShape;
struct ETextureParameterValue;
struct EPrimvarReaderParameterValue;

typedef unsigned char uint8;
typedef signed int int32;
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef unsigned short int uint16;
typedef std::variant<float, pxr::GfVec3f, ETextureParameterValue, EPrimvarReaderParameterValue, bool> FParameterValue;

typedef std::function<bool(const pxr::UsdPrim&)> FunctionPrune;
typedef std::map<std::string, EUsdBlendShape> EBlendShapeMap;

#define ENUM_CLASS_FLAGS(Enum) \
	inline           Enum& operator|=(Enum& Lhs, Enum Rhs) { return Lhs = (Enum)((__underlying_type(Enum))Lhs | (__underlying_type(Enum))Rhs); } \
	inline           Enum& operator&=(Enum& Lhs, Enum Rhs) { return Lhs = (Enum)((__underlying_type(Enum))Lhs & (__underlying_type(Enum))Rhs); } \
	inline           Enum& operator^=(Enum& Lhs, Enum Rhs) { return Lhs = (Enum)((__underlying_type(Enum))Lhs ^ (__underlying_type(Enum))Rhs); } \
	inline constexpr Enum  operator| (Enum  Lhs, Enum Rhs) { return (Enum)((__underlying_type(Enum))Lhs | (__underlying_type(Enum))Rhs); } \
	inline constexpr Enum  operator& (Enum  Lhs, Enum Rhs) { return (Enum)((__underlying_type(Enum))Lhs & (__underlying_type(Enum))Rhs); } \
	inline constexpr Enum  operator^ (Enum  Lhs, Enum Rhs) { return (Enum)((__underlying_type(Enum))Lhs ^ (__underlying_type(Enum))Rhs); } \
	inline constexpr bool  operator! (Enum  E)             { return !(__underlying_type(Enum))E; } \
	inline constexpr Enum  operator~ (Enum  E)             { return (Enum)~(__underlying_type(Enum))E; }

enum {
    INDEX_NONE	= -1
};

enum class ETextureWrapMode{
    TW_Clamp,
    TW_Mirror,
    TW_Black,
    TW_Repeat
};

enum class EUsdInitialLoadSet : uint8
{
    LoadAll,
    LoadNone
};

enum class EUsdUpAxis : uint8
{
    YAxis,
    ZAxis,
};

enum class EPrimAssignmentType : uint8
{
    None,
    DisplayColor,
    MaterialPrim,	// MaterialSource is the USD path to a Material prim on the stage (e.g. '/Root/Materials/Red')
};

enum class EUsdGeomOrientation
{
    RightHanded,
    LeftHanded,
};

enum class ECollapsingType
{
    Assets,
    Components
};

enum class EUsdDefaultKind : int32
{
    None = 0,
    Model = 1,          // 0000 0001
    Component = 2,      // 0000 0010
    Group = 4,          // 0000 0100
    Assembly = 8,       // 0000 1000
    Subcomponent = 16   // 0010 0000
};

ENUM_CLASS_FLAGS(EUsdDefaultKind)

enum class EUsdInterpolationMethod
{
    Vertex,  // Each element in a buffer maps directly to a specific vertex (point)
    FaceVarying,  // Each element in a buffer maps to a specific face/vertex pair
    Uniform,  // Each vertex on a face is the same value
    Constant  // Single value
};

std::ostream& operator<<(std::ostream& out, const ETextureWrapMode value);
std::ostream& operator<<(std::ostream& out, const EUsdUpAxis value);
std::ostream& operator<<(std::ostream& out, const EPrimAssignmentType value);

namespace EIdentifiers
{
    const pxr::TfToken DiffuseColor = pxr::TfToken("diffuseColor");
    const pxr::TfToken EmissiveColor = pxr::TfToken("emissiveColor");
    const pxr::TfToken Metallic = pxr::TfToken("metallic");
    const pxr::TfToken Roughness = pxr::TfToken("roughness");
    const pxr::TfToken Opacity = pxr::TfToken("opacity");
    const pxr::TfToken Normal = pxr::TfToken("normal");
    const pxr::TfToken Displacement = pxr::TfToken("displacement");
    const pxr::TfToken Specular = pxr::TfToken("specular");
    const pxr::TfToken Clearcoat = pxr::TfToken("clearcoat");
    const pxr::TfToken SpecularColor = pxr::TfToken("specularColor");

    const pxr::TfToken Anisotropy = pxr::TfToken("anisotropy");
    const pxr::TfToken Tangent = pxr::TfToken("tangent");
    const pxr::TfToken SubsurfaceColor = pxr::TfToken("subsurfaceColor");
    const pxr::TfToken Occlusion = pxr::TfToken("occlusion");
    const pxr::TfToken Refraction = pxr::TfToken("ior");

    const pxr::TfToken Surface = pxr::TfToken("surface");
    const pxr::TfToken St = pxr::TfToken("st");
    const pxr::TfToken Varname = pxr::TfToken("varname");
    const pxr::TfToken Result = pxr::TfToken("result");
    const pxr::TfToken File = pxr::TfToken("file");
    const pxr::TfToken Scale = pxr::TfToken("scale");
    const pxr::TfToken Bias = pxr::TfToken("bias");
    const pxr::TfToken WrapT = pxr::TfToken( "wrapT" );
    const pxr::TfToken WrapS = pxr::TfToken( "wrapS" );
    const pxr::TfToken Repeat = pxr::TfToken( "repeat" );
    const pxr::TfToken Mirror = pxr::TfToken( "mirror" );
    const pxr::TfToken Clamp = pxr::TfToken( "clamp" );
    const pxr::TfToken Fallback = pxr::TfToken("fallback");
    const pxr::TfToken R = pxr::TfToken("r");
    const pxr::TfToken RGB = pxr::TfToken("rgb");

    const pxr::TfToken UsdPreviewSurface = pxr::TfToken( "UsdPreviewSurface" );
    const pxr::TfToken UsdPrimvarReader_float = pxr::TfToken( "UsdPrimvarReader_float" );
    const pxr::TfToken UsdPrimvarReader_float2 = pxr::TfToken( "UsdPrimvarReader_float2" );
    const pxr::TfToken UsdPrimvarReader_float3 = pxr::TfToken( "UsdPrimvarReader_float3" );
    const pxr::TfToken UsdUVTexture = pxr::TfToken( "UsdUVTexture" );

    const pxr::TfToken LiveLinkAPI = pxr::TfToken("LiveLinkAPI");
    const pxr::TfToken ControlRigAPI = pxr::TfToken("ControlRigAPI");
    const pxr::TfToken GroomAPI = pxr::TfToken( "GroomAPI" );
    const pxr::TfToken LOD("LOD");
}

struct EElementID{
    EElementID(const int32 InitIDValue) : IDValue(InitIDValue){}
    EElementID() : IDValue(-1){}

    bool operator==( const EElementID& Other ) const
    {
        return IDValue == Other.IDValue;
    }

    bool operator==( const int32 Other ) const
    {
        return IDValue == Other;
    }

    bool operator!=( const EElementID& Other ) const
    {
        return IDValue != Other.IDValue;
    }

    bool operator!=( const int32 Other ) const
    {
        return IDValue != Other;
    }

    int32 IDValue;
};

struct EVertexID : public EElementID
{
    EVertexID(){}
    EVertexID(const EElementID InitElementID): EVertexID(InitElementID.IDValue){}
    EVertexID(const int32 InitIDValue): EElementID(InitIDValue){}
};

struct EEdgeID : public EElementID
{
    EEdgeID(){}
    EEdgeID(const EElementID InitElementID): EEdgeID(InitElementID.IDValue){}
    EEdgeID(const int32 InitIDValue): EElementID(InitIDValue){}
};

struct EVertexInstanceID : public EElementID
{
    EVertexInstanceID(){}
    EVertexInstanceID(const EElementID InitElementID): EVertexInstanceID(InitElementID.IDValue){}
    EVertexInstanceID(const int32 InitIDValue): EElementID(InitIDValue){}
};

// TODO rename polygon to face
struct EPolygonID : public EElementID
{
    EPolygonID(){}
    EPolygonID(const EElementID InitElementID): EPolygonID(InitElementID.IDValue){}
    EPolygonID(const int32 InitIDValue): EElementID(InitIDValue){}
};

struct EPolygonGroupID : public EElementID
{
    EPolygonGroupID(){}
    EPolygonGroupID(const EElementID InitElementID): EPolygonGroupID(InitElementID.IDValue){}
    EPolygonGroupID(const int32 InitIDValue): EElementID(InitIDValue){}
};

struct EConvertExtraInfo{
    pxr::VtStringArray ConvertMeshIgnores;
};

struct ETextureParameterValue{
    std::string TexturePath;
    int32 UVIndex;
    int32 OutputIndex = 0;

    friend std::ostream& operator<<(std::ostream& out, ETextureParameterValue const& textureParameterValue);
};

struct EPrimvarReaderParameterValue{
    std::string PrimvarName;
    std::variant<float, pxr::GfVec2f, pxr::GfVec3f> FallbackValue;
};

struct EMorphTargetDelta{
    pxr::GfVec3f PositionDelta;  // change in position
    pxr::GfVec3f TangentZDelta;  // Tangent basis normal
    uint32 SourceIdx;  // index of source vertex to apply deltas to

    friend std::ostream& operator<<(std::ostream& out, EMorphTargetDelta const& morphTargetDelta);
};

struct EUsdBlendShapeInbetween{
    std::string Name;
    std::string Path;
    float InbetweenWeight;

    friend std::ostream& operator<<(std::ostream& out, EUsdBlendShapeInbetween const& blendShapeInbetween);
};

struct ETransform{
    pxr::GfQuatf Rotation = {0, 0, 0, 1};/** Rotation of this transformation, as a quaternion */
    pxr::GfVec4f Translation = {0, 0, 0, 0};/** Translation of this transformation, as a vector */
    pxr::GfVec4f Scale3D = {1, 1, 1, 0}; /** 3D scale (always applied in local space) as a vector */

    void SetFromMatrix(pxr::GfMatrix4f InMatrix);
    void ScaleTranslation(float InScale);
    void ScaleTranslation(pxr::GfVec3f InScale3D);

    void Multiply(ETransform* OutTransform, const ETransform* A, const ETransform* B);
    pxr::GfVec3f TransformPosition(const pxr::GfVec3f& V);
    pxr::GfVec3f TransformVector(const pxr::GfVec3f& V);

    ETransform operator*(const ETransform& Other);
    friend std::ostream& operator<<(std::ostream& out, ETransform const& eTransform);

    pxr::GfVec3f GetTranslation();
    pxr::GfQuatf GetRotation();
    pxr::GfVec3f GetScale3D();

    ETransform(pxr::GfVec3f InRotation);
    ETransform(pxr::GfQuatf InRotation,
               pxr::GfVec3f InTranslation,
               pxr::GfVec3f InScale3D = pxr::GfVec3f(1.f, 1.f, 1.f));
    ETransform(pxr::GfMatrix4f InMatrix);
    ETransform();
};

struct ERawAnimSequenceTrack{
    pxr::VtArray<pxr::GfVec3f> PosKeys;
    pxr::VtArray<pxr::GfQuatf> RotKeys;
    pxr::VtArray<pxr::GfVec3f> ScaleKeys;
    pxr::VtArray<ETransform> ConvertedTransforms;
    pxr::VtArray<pxr::GfMatrix4d> UsdTransforms;

    friend std::ostream& operator<<(std::ostream& out, ERawAnimSequenceTrack const& animSequenceTrack);
};

struct EJointPos{
    ETransform Transform = {};
};

struct EBone{
    std::string Name = {};
    int32 NumChildren = {};
    int32 ParentIndex = {};  // 0/NULL if this is the root bone.
    EJointPos BonePos = {};      // reference position

    friend std::ostream& operator<<(std::ostream& out, EBone const& eBone);
};

struct ERawBoneInfluence {
    float Weight;
    int32 VertexIndex;
    int32 BoneIndex;

    friend std::ostream& operator<<(std::ostream& out, ERawBoneInfluence const& rawBoneInfluence);
};

struct EVertex{
    uint32	VertexIndex;
    pxr::GfVec2f UVs[ED_MAX_TEXCOORDS];
    pxr::GfVec4f Color;
    uint8 MatIndex;
    uint8 Reserved;
};

//struct ETriangle{
//    uint32 WedgeIndex[3]; // Point to three vertices in the vertex list.
//    uint16 MatIndex; // Materials can be anything.
//    uint8 AuxMatIndex; // Second material from exporter (unused)
//    uint32 SmoothingGroups; // 32-bit flag for smoothing groups.
//
//    pxr::GfVec3f TangentX[3];
//    pxr::GfVec3f TangentY[3];
//    pxr::GfVec3f TangentZ[3];
//};

//struct EFace{
//
//};

struct EMaterialParams{
    pxr::GfVec3f baseColor;
    pxr::GfVec3f emissionColor;

    float metallic;
    float roughness;
    float subsurface;
    float specularTint;

    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;

    float specTrans;
    float ior;
    float anisotropic;
};

struct EMaterial{
    std::shared_ptr<EMaterialParams> Material;
    std::string MaterialImportName;
};

struct EMeshInfo
{
    std::string Name;	// The name of the mesh.
    int32 NumVertices;	// The number of imported (dcc) vertices that are part of this mesh. This is a value of 8 for a cube. So NOT the number of render vertices.
    int32 StartImportedVertex;	// The first index of imported (dcc) vertices in the mesh. So this NOT an index into the render vertex buffer. In range of 0..7 for a cube.
};

struct EUsdBlendShape{
    std::string Name{};
    pxr::VtArray<EMorphTargetDelta> Vertices{};
    pxr::VtArray<EUsdBlendShapeInbetween> InBetweens{};
    bool bHasAuthoredTangents = false;

    std::map<std::string, EUsdBlendShape> InBetweenBlendShapes{};

    bool IsValid() const;

    friend std::ostream& operator<<(std::ostream& out, EUsdBlendShape const& usdBlendShape);
};

struct EUsdLayerTimeInfo
{
    std::string Identifier;
    std::string FilePath;

    double StartTimeCode = 0.0;
    double EndTimeCode = 0.0;

    bool IsAnimated();
};

struct EUsdOpenStageOptions{
    std::string Identifier = "";
    double TimeCode = 0.0;
    EUsdInitialLoadSet InitialLoadSet = {};
    bool bUseStageCache = true;
    bool bForceReloadLayersFromDisk = false;
};

struct EUVSet
{
    int32 UVSetIndex;
    pxr::VtIntArray UVIndices; // UVs might be indexed or they might be flat (one per vertex)
    pxr::VtVec2fArray UVs;

    pxr::TfToken InterpType = pxr::UsdGeomTokens->faceVarying;

    friend std::ostream& operator<<(std::ostream& out, EUVSet const& uvSet);
};

struct EUsdTranslatorContext{
    const pxr::UsdPrim& Prim;
    const EConvertExtraInfo& ExtraInfo;
};

struct EGeomMeshData{
    pxr::VtArray<pxr::GfVec3f> Points{};                // Points
    pxr::VtArray<EVertexID> VertexInstances{};          // Vertices
    pxr::VtArray<pxr::GfVec3f> Normals{};               // Vertex Based Normal
    pxr::VtArray<pxr::GfVec4f> Colors{};                // Vertex Based Color
    std::map<int32, pxr::VtArray<pxr::GfVec2f>> UVs;    // Vertex Based UV

    pxr::VtArray<EPolygonID> Faces{};
    pxr::VtArray<std::pair<EEdgeID, EEdgeID>> Edges{};

    pxr::VtArray<int> FaceCounts{};
    pxr::VtArray<int> FaceIndices{};

    void ReserveNewVertices(const int32 NumVertices);
    //void ReserveNewPolygons(const int32 NumPolygons);
    EPolygonID CreatePolygon(pxr::VtArray<EVertexInstanceID> VertexInstanceIDs, pxr::VtArray<EEdgeID>* OutEdgeIDs = nullptr);
    EVertexInstanceID CreateVertexInstance(const EVertexID VertexID);
    EEdgeID CreateEdge(const EVertexID VertexID0, const EVertexID VertexID1);

    bool ApplyTransform(ETransform Transform, EUsdStageInfo StageInfo);

    friend std::ostream& operator<<(std::ostream& out, EGeomMeshData const& geomMeshImportData);
};

struct EUSDConverted{
    std::set<std::string> ConvertedGeomMesh;
};

struct EGeometryCache{
    int32 StartFrame;
    int32 EndFrame;
    pxr::VtArray<EGeomMeshData> GeomMeshDatas;
};

struct EAnimationInfo{
    int32 NumBakedFrames;
    double StageStartTimeCode;
    double StageTimeCodesPerSecond;
    double StageBakeIntervalTimeCodes;
    double StageSequenceLengthTimeCodes;
};

struct ESkeletalAnimData{
    // Index is Bone-Index
    pxr::VtArray<pxr::GfMatrix4d> WorldSpaceRestTransforms{};
    pxr::VtArray<pxr::GfMatrix4d> WorldSpaceBindTransforms{};
    pxr::VtArray<pxr::GfMatrix4d> JointLocalRestTransforms{};
    pxr::VtArray<pxr::GfMatrix4d> MeshToSkeletonRestPose{};
    pxr::VtArray<ERawAnimSequenceTrack> JointAnimTracks{};

    EAnimationInfo AnimationInfo;
    pxr::VtArray<pxr::TfToken> BlendShapeChannelOrder{};
    pxr::VtArray<pxr::SdfPath> BlendShapePathsToSkinnedPrims{};

    // Index is Frame-Index
    pxr::VtArray<ETransform> RootMotionTransforms{};

    // Index is BlendShapeChannel-Index
    pxr::VtArray<pxr::VtArray<float>> BlendShapeWeights{};

    friend std::ostream& operator<<(std::ostream& out, ESkeletalAnimData const& d);
};



struct ESkeletalMeshImportData{
    pxr::VtArray<EMaterial> Materials;
    pxr::VtArray<EBone> RefBonesBinary;
    pxr::VtArray<ERawBoneInfluence> Influences;

    //pxr::VtArray<EMeshInfo> MeshInfos;
    //pxr::VtArray<int32> PointToRawMap;	// Mapping from current point index to the original import point index
    //pxr::VtArray<pxr::GfVec3f> Points;
    //pxr::VtArray<EVertex> Wedges;
    //pxr::VtArray<ETriangle> Faces;
    //pxr::VtArray<EFace> Faces;

    uint32 NumTexCoords; // The number of texture coordinate sets
    uint32 MaxMaterialIndex; // The max material index found on a triangle
    bool bHasVertexColors; // If true there are vertex colors in the imported file
    bool bHasNormals; // If true there are normals in the imported file
    bool bHasTangents; // If true there are tangents in the imported file

    bool bUseT0AsRefPose; // If true, then the pose at time=0 will be used instead of the ref pose
    bool bDiffPose; // If true, one of the bones has a different pose at time=0 vs the ref pose

    //pxr::VtArray<pxr::GfVec3f> Normals{};
    //pxr::VtArray<pxr::GfVec4f> Colors{};
    //std::map<int32, pxr::VtArray<pxr::GfVec2f>> UVs;
    //pxr::VtArray<int> FaceCounts{};
    //pxr::VtArray<int> FaceIndices{};

    EGeomMeshData MeshData{};

    // Morph targets imported(i.e. FBX) data. The name is the morph target name
    //std::vector<ESkeletalMeshImportData> MorphTargets;
    //std::vector<std::set<uint32>> MorphTargetModifiedPoints;
    //std::vector<std::string> MorphTargetNames;

    // Alternate influence imported(i.e. FBX) data. The name is the alternate skinning profile name
    //std::vector<ESkeletalMeshImportData> AlternateInfluences;
    //std::vector<std::string> AlternateInfluenceProfileNames;

    friend std::ostream& operator<<(std::ostream& out, ESkeletalMeshImportData const& skeletalMeshImportData);

    std::string PrimPath{};
    pxr::VtArray<std::pair<std::string, std::string>> BlendShapeMap{};

    ESkeletalMeshImportData()
        : NumTexCoords(0)
        , MaxMaterialIndex(0)
        , bHasVertexColors(false)
        , bHasNormals(false)
        , bHasTangents(false)
        , bUseT0AsRefPose(false)
        , bDiffPose(false)
    {
        ED_COUT << "ESkeletalMeshImportData Constructed.\n";
    }
};

struct ESkelImportData{
    pxr::VtArray<EBone> SkeletonBones{};
    std::map<std::string, EUsdBlendShape> BlendShapes{};
    //pxr::VtArray<ESkeletalMeshImportData> SkeletalMeshData{};
    std::map<std::string, ESkeletalMeshImportData> SkeletalMeshData{};
    ESkeletalAnimData SkeletalAnimData{};
};

struct EUSDImported{
    std::map<std::string, EGeomMeshData> PathToMeshImportData;
    std::map<std::string, std::map<int, EGeomMeshData>> PathToFrameToMeshImportData;
    std::map<std::string, ETransform> PathToMeshTransform;
    std::map<std::string, std::map<int, ETransform>> PathToFrameToTransform;

    std::map<std::string, ESkelImportData> PathToSkelImportData;
};

struct EUsdStageInfo{
    EUsdUpAxis UpAxis = EUsdUpAxis::ZAxis;
    float MetersPerUnit = ED_METERS_PER_UNIT;

    explicit EUsdStageInfo(const pxr::UsdStageRefPtr& Stage);
};

struct EUsdMeshConversionOptions{
    std::map<std::string, std::map<std::string, int32>>* MaterialToPrimvarToUVIndex;
    pxr::UsdTimeCode TimeCode;
    ETransform AdditionalTransform;

    const EConvertExtraInfo* ExtraInfo = nullptr;
    EUSDConverted* Converted = nullptr;
};

struct EUsdPrimMaterialSlot{
    std::string MaterialSource;
    EPrimAssignmentType AssignmentType = EPrimAssignmentType::None;

    friend std::ostream& operator<<(std::ostream& out, EUsdPrimMaterialSlot const& primMaterialSlot);
};

struct EUsdPrimMaterialAssignmentInfo
{
    pxr::VtArray<EUsdPrimMaterialSlot> Slots;
    pxr::VtArray<int32> MaterialIndices;

    friend std::ostream& operator<<(std::ostream& out, EUsdPrimMaterialAssignmentInfo const& primMaterialAssignmentInfo);
};

struct EUSDStage{
    pxr::UsdStageRefPtr USDStage;
};

#endif //YE_USDDEFINITION_H
