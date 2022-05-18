#ifndef ZENO_DEFINITION_H
#define ZENO_DEFINITION_H

#define GET_MAT_COLOR(VAR, KEY, TYPE, INDEX, DEFAULT) \
    if(AI_SUCCESS != aiGetMaterialColor(material, KEY, TYPE, INDEX, &VAR)) \
        VAR = DEFAULT;                                \
    zeno::log_info(">>>>> Material `{}` Result {} {} {} {}", KEY, VAR.r, VAR.g, VAR.b, VAR.a);

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

struct STexture {
    int type;
    std::string path;
};

struct SVertex{
    aiVector3D position;
    aiVector3D texCoord;
    aiVector3D normal;
    aiVector3D tangent;
    aiVector3D bitangent;
    std::unordered_map<std::string, float> boneWeights;
};

struct SMaterial{
    std::string matName;

    std::unordered_map<int, std::vector<STexture>> tex;

    aiColor4D base;
    aiColor4D specular;
    aiColor4D transmission;
    aiColor4D subsurface;
    aiColor4D sheen;
    aiColor4D coat;
    aiColor4D emission;

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
    std::unordered_map<std::string, SMaterial> value;
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