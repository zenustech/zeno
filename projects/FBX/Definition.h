#ifndef ZENO_DEFINITION_H
#define ZENO_DEFINITION_H

#include <iostream>
#include <algorithm>

#define COMMON_DEFAULT_NORMAL aiColor4D(0.0f, 0.0f, 1.0f, 1.0f)

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
    std::any value;
    aiTextureType type;
    std::string aiName;
    std::string texPath;
};

struct SDefaultMatProp{
    std::map<std::string, aiColor4D> getUnknownProp(){
        std::map<std::string, aiColor4D> val;

        val.emplace("basecolor", aiColor4D());
        val.emplace("metallic", aiColor4D());
        val.emplace("roughness", aiColor4D());
        val.emplace("specular", aiColor4D());
        val.emplace("subsurface", aiColor4D());
        val.emplace("thinkness", aiColor4D());
        val.emplace("sssParam", aiColor4D());
        val.emplace("sssColor", aiColor4D());
        val.emplace("foliage", aiColor4D());
        val.emplace("skin", aiColor4D());
        val.emplace("curvature", aiColor4D());
        val.emplace("specularTint", aiColor4D());
        val.emplace("anisotropic", aiColor4D());
        val.emplace("sheen", aiColor4D());
        val.emplace("sheenTint", aiColor4D());
        val.emplace("clearcoat", aiColor4D());
        val.emplace("clearcoatGloss", aiColor4D());
        val.emplace("normal", COMMON_DEFAULT_NORMAL);
        val.emplace("emission", aiColor4D());
        val.emplace("exposure", aiColor4D());
        val.emplace("ao", aiColor4D());
        val.emplace("toon", aiColor4D());
        val.emplace("stroke", aiColor4D());
        val.emplace("shape", aiColor4D());
        val.emplace("style", aiColor4D());
        val.emplace("strokeNoise", aiColor4D());
        val.emplace("shad", aiColor4D());
        val.emplace("strokeTint", aiColor4D());
        val.emplace("opacity", aiColor4D());
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
        val.emplace("basecolor", SMaterialProp{0, aiColor4D(), aiTextureType_BASE_COLOR, "$ai.base"});
        val.emplace("metallic", SMaterialProp{1, aiColor4D(), aiTextureType_METALNESS, ""});
        val.emplace("roughness", SMaterialProp{2, aiColor4D(), aiTextureType_DIFFUSE_ROUGHNESS, ""});
        val.emplace("specular", SMaterialProp{3, aiColor4D(), aiTextureType_SPECULAR, "$ai.specular"});
        val.emplace("subsurface", SMaterialProp{4, aiColor4D(), aiTextureType_NONE, "$ai.subsurface"});
        val.emplace("thinkness", SMaterialProp{5, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("sssParam", SMaterialProp{6, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("sssColor", SMaterialProp{7, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("foliage", SMaterialProp{8, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("skin", SMaterialProp{9, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("curvature", SMaterialProp{10, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("specularTint", SMaterialProp{11, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("anisotropic", SMaterialProp{12, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("sheen", SMaterialProp{13, aiColor4D(), aiTextureType_SHININESS, "$ai.sheen"});
        val.emplace("sheenTint", SMaterialProp{14, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("clearcoat", SMaterialProp{15, aiColor4D(), aiTextureType_NONE, "$ai.coat"});
        val.emplace("clearcoatGloss", SMaterialProp{16, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("normal", SMaterialProp{17, aiColor4D(), aiTextureType_NORMAL_CAMERA, ""});
        val.emplace("emission", SMaterialProp{18, aiColor4D(), aiTextureType_EMISSIVE, "$ai.emission"}); // aiTextureType_EMISSION_COLOR
        val.emplace("exposure", SMaterialProp{19, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("ao", SMaterialProp{20, aiColor4D(), aiTextureType_AMBIENT_OCCLUSION, ""});
        val.emplace("toon", SMaterialProp{21, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("stroke", SMaterialProp{22, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("shape", SMaterialProp{23, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("style", SMaterialProp{24, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("strokeNoise", SMaterialProp{25, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("shad", SMaterialProp{26, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("strokeTint", SMaterialProp{27, aiColor4D(), aiTextureType_NONE, ""});
        val.emplace("opacity", SMaterialProp{28, aiColor4D(), aiTextureType_OPACITY, "$ai.transmission"});
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
};

#endif //ZENO_DEFINITION_H