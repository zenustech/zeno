#ifndef ZENO_DEFINITION_H
#define ZENO_DEFINITION_H

#include <iostream>
#include <algorithm>

#define COMMON_DEFAULT_basecolor aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_metallic aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_roughness aiColor4D(0.4f, 0.4f, 0.4f, 1.0f)
#define COMMON_DEFAULT_specular aiColor4D(0.5f, 0.5f, 0.5f, 1.0f)
#define COMMON_DEFAULT_subsurface aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_thinkness aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_sssParam aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_sssColor aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_foliage aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_skin aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_curvature aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_specularTint aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_anisotropic aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_sheen aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_sheenTint aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_clearcoat aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_clearcoatGloss aiColor4D(1.0f, 1.0f, 1.0f, 1.0f)
#define COMMON_DEFAULT_normal aiColor4D(0.0f, 0.0f, 1.0f, 1.0f)
#define COMMON_DEFAULT_emission aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_exposure aiColor4D(1.0f, 1.0f, 1.0f, 1.0f)
#define COMMON_DEFAULT_ao aiColor4D(1.0f, 1.0f, 1.0f, 1.0f)
#define COMMON_DEFAULT_toon aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_stroke aiColor4D(1.0f, 1.0f, 1.0f, 1.0f)
#define COMMON_DEFAULT_shape aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_style aiColor4D(1.0f, 1.0f, 1.0f, 1.0f)
#define COMMON_DEFAULT_strokeNoise aiColor4D(1.0f, 1.0f, 1.0f, 1.0f)
#define COMMON_DEFAULT_shad aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_strokeTint aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)
#define COMMON_DEFAULT_opacity aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)

struct SKeyPosition {
    aiVector3D position;
    float timeStamp;
};

struct SKeyRotation {
    aiQuaternion orientation;
    float timeStamp;
};

struct SKeyScale {
    aiVector3D scale;
    float timeStamp;
};

struct SAnimBone {
    std::vector<SKeyPosition> m_Positions;
    std::vector<SKeyRotation> m_Rotations;
    std::vector<SKeyScale> m_Scales;

    int m_NumPositions = 0;
    int m_NumRotations = 0;
    int m_NumScalings = 0;

    aiMatrix4x4 m_LocalTransform;
    std::string m_Name;


    void initBone(std::string name, const aiNodeAnim* channel){
        m_Name = name;
        m_NumPositions = channel->mNumPositionKeys;
        for (int positionIndex = 0; positionIndex < m_NumPositions; ++positionIndex) {
            aiVector3D aiPosition = channel->mPositionKeys[positionIndex].mValue;
            float timeStamp = channel->mPositionKeys[positionIndex].mTime;

            SKeyPosition data;
            data.position = aiPosition;
            data.timeStamp = timeStamp;
            m_Positions.push_back(data);
        }

        m_NumRotations = channel->mNumRotationKeys;
        for (int rotationIndex = 0; rotationIndex < m_NumRotations; ++rotationIndex) {
            aiQuaternion aiOrientation = channel->mRotationKeys[rotationIndex].mValue;
            float timeStamp = channel->mRotationKeys[rotationIndex].mTime;

            SKeyRotation data;
            data.orientation = aiOrientation;
            data.timeStamp = timeStamp;
            m_Rotations.push_back(data);
        }

        m_NumScalings = channel->mNumScalingKeys;
        for (int keyIndex = 0; keyIndex < m_NumScalings; ++keyIndex) {
            aiVector3D scale = channel->mScalingKeys[keyIndex].mValue;
            float timeStamp = channel->mScalingKeys[keyIndex].mTime;

            SKeyScale data;
            data.scale = scale;
            data.timeStamp = timeStamp;
            m_Scales.push_back(data);
        }
    }

    void update(float animationTime) {
        aiMatrix4x4 translation = interpolatePosition(animationTime);
        aiMatrix4x4 rotation = interpolateRotation(animationTime);
        aiMatrix4x4 scale = interpolateScaling(animationTime);

        m_LocalTransform = translation * rotation * scale;
    }

    int getPositionIndex(float animationTime) {
        for (int index = 0; index < m_NumPositions - 1; ++index) {
            if (animationTime < m_Positions[index + 1].timeStamp)
                return index;
        }
    }
    int getRotationIndex(float animationTime) {
        for (int index = 0; index < m_NumRotations - 1; ++index) {
            if (animationTime < m_Rotations[index + 1].timeStamp)
                return index;
        }
    }
    int getScaleIndex(float animationTime) {
        for (int index = 0; index < m_NumScalings - 1; ++index) {
            if (animationTime < m_Scales[index + 1].timeStamp)
                return index;
        }
    }

    aiMatrix4x4 interpolatePosition(float animationTime) {
        aiMatrix4x4 result;

        if (1 == m_NumPositions) {
            aiMatrix4x4::Translation(m_Positions[0].position, result);
            return result;
        }

        int p0Index = getPositionIndex(animationTime);
        int p1Index = p0Index + 1;
        float scaleFactor = getScaleFactor(
                m_Positions[p0Index].timeStamp,
                m_Positions[p1Index].timeStamp,
                animationTime);
        aiVector3D finalPosition = m_Positions[p0Index].position * (1.0f - scaleFactor) + m_Positions[p1Index].position * scaleFactor;
        aiMatrix4x4::Translation(finalPosition, result);

        return result;
    }

    aiMatrix4x4 interpolateRotation(float animationTime) {
        aiMatrix4x4 result;

        if (1 == m_NumRotations) {
            result = result * aiMatrix4x4(m_Rotations[0].orientation.GetMatrix());
            return result;
        }

        int p0Index = getRotationIndex(animationTime);
        int p1Index = p0Index + 1;
        float scaleFactor = getScaleFactor(
                m_Rotations[p0Index].timeStamp,
                m_Rotations[p1Index].timeStamp,
                animationTime);
        aiQuaternion finalRotation;
        aiQuaternion::Interpolate(finalRotation, m_Rotations[p0Index].orientation, m_Rotations[p1Index].orientation, scaleFactor);
        result = result * aiMatrix4x4(finalRotation.GetMatrix());

        return result;
    }

    aiMatrix4x4 interpolateScaling(float animationTime) {
        aiMatrix4x4 result;
        if (1 == m_NumScalings) {
            aiMatrix4x4::Scaling(m_Scales[0].scale, result);
            return result;
        }

        int p0Index = getScaleIndex(animationTime);
        int p1Index = p0Index + 1;
        float scaleFactor = getScaleFactor(
                m_Scales[p0Index].timeStamp,
                m_Scales[p1Index].timeStamp,
                animationTime);
        aiVector3D finalScale = m_Scales[p0Index].scale *  (1.0f - scaleFactor) + m_Scales[p1Index].scale * scaleFactor;
        aiMatrix4x4::Scaling(finalScale, result);

        return result;
    }

    float getScaleFactor(float lastTimeStamp, float nextTimeStamp, float animationTime) {
        float scaleFactor = 0.0f;
        float midWayLength = animationTime - lastTimeStamp;
        float framesDiff = nextTimeStamp - lastTimeStamp;
        scaleFactor = midWayLength / framesDiff;

        return scaleFactor;
    }
};

struct SVertex{
    aiVector3D position;
    aiVector3D texCoord;
    aiVector3D normal;
    aiVector3D tangent;
    aiVector3D bitangent;
    std::unordered_map<std::string, float> boneWeights;
};

struct SMaterialProp{
    int order;
    bool forceDefault;
    std::any value;
    aiTextureType type;
    std::string aiName;
    std::string texPath;
};

struct SDefaultMatProp{
    std::map<std::string, aiColor4D> getUnknownProp(){
        std::map<std::string, aiColor4D> val;

        val.emplace("basecolor", COMMON_DEFAULT_basecolor);
        val.emplace("metallic", COMMON_DEFAULT_metallic);
        val.emplace("roughness", COMMON_DEFAULT_roughness);
        val.emplace("specular", COMMON_DEFAULT_specular);
        val.emplace("subsurface", COMMON_DEFAULT_subsurface);
        val.emplace("thinkness", COMMON_DEFAULT_thinkness);
        val.emplace("sssParam", COMMON_DEFAULT_sssParam);
        val.emplace("sssColor", COMMON_DEFAULT_sssColor);
        val.emplace("foliage", COMMON_DEFAULT_foliage);
        val.emplace("skin", COMMON_DEFAULT_skin);
        val.emplace("curvature", COMMON_DEFAULT_curvature);
        val.emplace("specularTint", COMMON_DEFAULT_specularTint);
        val.emplace("anisotropic", COMMON_DEFAULT_anisotropic);
        val.emplace("sheen", COMMON_DEFAULT_sheen);
        val.emplace("sheenTint", COMMON_DEFAULT_sheenTint);
        val.emplace("clearcoat", COMMON_DEFAULT_clearcoat);
        val.emplace("clearcoatGloss", COMMON_DEFAULT_clearcoatGloss);
        val.emplace("normal", COMMON_DEFAULT_normal);
        val.emplace("emission", COMMON_DEFAULT_emission);
        val.emplace("exposure", COMMON_DEFAULT_exposure);
        val.emplace("ao", COMMON_DEFAULT_ao);
        val.emplace("toon", COMMON_DEFAULT_toon);
        val.emplace("stroke", COMMON_DEFAULT_stroke);
        val.emplace("shape", COMMON_DEFAULT_shape);
        val.emplace("style", COMMON_DEFAULT_style);
        val.emplace("strokeNoise", COMMON_DEFAULT_strokeNoise);
        val.emplace("shad", COMMON_DEFAULT_shad);
        val.emplace("strokeTint", COMMON_DEFAULT_strokeTint);
        val.emplace("opacity", COMMON_DEFAULT_opacity);
        return val;
    }
};

struct SMaterial : zeno::IObjectClone<SMaterial>{
    using pair = std::pair<std::string, SMaterialProp>;

    std::string matName;
    std::map<std::string, SMaterialProp> val;
    std::vector<pair> val_vec;

    void setDefaultValue(std::map<std::string, aiColor4D> dValueMap){
        for(auto&v: val){
            v.second.value = dValueMap.at(v.first);
        }
    }

    SMaterial(){
        // FIXME (aiTextureType_BASE_COLOR 12 basecolor `aiStandardSurface`)
        //      or (aiTextureType_DIFFUSE 1 diffuse `lambert`)
        // aiTextureType_NORMALS or aiTextureType_NORMAL_CAMERA
        val.emplace("basecolor", SMaterialProp{0, false, aiColor4D(), aiTextureType_BASE_COLOR, "$ai.base"});
        val.emplace("metallic", SMaterialProp{1, true, aiColor4D(), aiTextureType_METALNESS, ""});
        val.emplace("roughness", SMaterialProp{2, true, aiColor4D(), aiTextureType_DIFFUSE_ROUGHNESS, ""});
        val.emplace("specular", SMaterialProp{3, true, aiColor4D(), aiTextureType_SPECULAR, "$ai.specular"});
        val.emplace("subsurface", SMaterialProp{4, true, aiColor4D(), aiTextureType_NONE, "$ai.subsurface"});
        val.emplace("thinkness", SMaterialProp{5, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("sssParam", SMaterialProp{6, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("sssColor", SMaterialProp{7, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("foliage", SMaterialProp{8, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("skin", SMaterialProp{9, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("curvature", SMaterialProp{10, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("specularTint", SMaterialProp{11, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("anisotropic", SMaterialProp{12, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("sheen", SMaterialProp{13, true, aiColor4D(), aiTextureType_SHININESS, "$ai.sheen"});
        val.emplace("sheenTint", SMaterialProp{14, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("clearcoat", SMaterialProp{15, true, aiColor4D(), aiTextureType_NONE, "$ai.coat"});
        val.emplace("clearcoatGloss", SMaterialProp{16, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("normal", SMaterialProp{17, true, aiColor4D(), aiTextureType_NORMAL_CAMERA, ""});
        val.emplace("emission", SMaterialProp{18, true, aiColor4D(), aiTextureType_EMISSIVE, "$ai.emission"}); // aiTextureType_EMISSION_COLOR
        val.emplace("exposure", SMaterialProp{19, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("ao", SMaterialProp{20, true, aiColor4D(), aiTextureType_AMBIENT_OCCLUSION, ""});
        val.emplace("toon", SMaterialProp{21, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("stroke", SMaterialProp{22, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("shape", SMaterialProp{23, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("style", SMaterialProp{24, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("strokeNoise", SMaterialProp{25, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("shad", SMaterialProp{26, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("strokeTint", SMaterialProp{27, true, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("opacity", SMaterialProp{28, true, aiColor4D(), aiTextureType_OPACITY, "$ai.transmission"});
    }

    std::vector<zeno::Any> getTexList(){
        std::vector<zeno::Any> tl;

        std::copy(val.begin(),
                  val.end(),
                  std::back_inserter<std::vector<pair>>(val_vec));

        std::sort(val_vec.begin(), val_vec.end(),
                  [](const pair &l, const pair &r)
                  {
                      if (l.second.order != r.second.order) {
                          return l.second.order < r.second.order;
                      }

                      return l.first < r.first;
                  });
        for (auto const &p: val_vec) {
            //zeno::log_info("Pair {} {}", p.first, p.second.order);
            tl.emplace_back(p.second.texPath);
        }

        return tl;
    }

    aiColor4D testColor;
    float testFloat;
};

struct SBoneOffset {
    std::string name;
    aiMatrix4x4 offset;
};

struct SCamera {
    float hFov;
    float focL;
    float aspect;
    float pNear;
    float pFar;
    zeno::vec3f interestPos;
    /*zeno::vec3f lookAt;*/
    zeno::vec3f pos;
    zeno::vec3f up;
    zeno::vec3f view;
    /*aiMatrix4x4 camM;*/
};

struct NodeTree : zeno::IObjectClone<NodeTree>{
    aiMatrix4x4 transformation;
    std::string name;
    int childrenCount;
    std::vector<NodeTree> children;
};

struct BoneTree : zeno::IObjectClone<BoneTree>{
    std::unordered_map<std::string, SAnimBone> AnimBoneMap;
};

struct AnimInfo : zeno::IObjectClone<AnimInfo>{
    float duration;
    float tick;
};

struct IMaterial : zeno::IObjectClone<IMaterial>{
    std::unordered_map<std::string, SMaterial> value;  // key: meshName
};

struct IBoneOffset : zeno::IObjectClone<IBoneOffset>{
    std::unordered_map<std::string, SBoneOffset> value;
};

struct ICamera : zeno::IObjectClone<ICamera>{
    std::unordered_map<std::string, SCamera> value;
};

struct IVertices : zeno::IObjectClone<IVertices>{
    std::vector<SVertex> value;
};

struct IIndices : zeno::IObjectClone<IIndices>{
    std::vector<unsigned int> value;
};

struct FBXData : zeno::IObjectClone<FBXData>{
    IVertices iVertices;
    IIndices iIndices;
    IMaterial iMaterial;
    IBoneOffset iBoneOffset;
    ICamera iCamera;
};

struct Helper{
    static void printAiMatrix(aiMatrix4x4 m, bool transpose = false){
        zeno::log_info("    {: f} {: f} {: f} {: f}", m.a1, m.a2, m.a3, m.a4);
        zeno::log_info("    {: f} {: f} {: f} {: f}", m.b1, m.b2, m.b3, m.b4);
        zeno::log_info("    {: f} {: f} {: f} {: f}", m.c1, m.c2, m.c3, m.c4);
        zeno::log_info("    {: f} {: f} {: f} {: f}", m.d1, m.d2, m.d3, m.d4);

        aiVector3t<float> trans;
        aiQuaterniont<float> rotate;
        aiVector3t<float> scale;
        m.Decompose(scale, rotate, trans);
        zeno::log_info("    T {: f} {: f} {: f}", trans.x, trans.y, trans.z);
        zeno::log_info("    R {: f} {: f} {: f} {: f}", rotate.x, rotate.y, rotate.z, rotate.w);
        zeno::log_info("    S {: f} {: f} {: f}", scale.x, scale.y, scale.z);

        aiMatrix3x3 r = rotate.GetMatrix();
        if (transpose)
            r = rotate.GetMatrix().Transpose();
        zeno::log_info("    {: f} {: f} {: f}", r.a1, r.a2, r.a3);
        zeno::log_info("    {: f} {: f} {: f}", r.b1, r.b2, r.b3);
        zeno::log_info("    {: f} {: f} {: f}", r.c1, r.c2, r.c3);
    }

    static void printNodeTree(NodeTree *root, int space){
        int c = 1;
        if (root == nullptr)
            return;
        space += c;
        for(int i=0;i<root->children.size(); i++){
            printNodeTree(&root->children[i], space);
        }
        for (int i = c; i < space; i++)
            std::cout << "\t";
        std::cout << root->name <<"\n";
    }
};

#endif //ZENO_DEFINITION_H