#ifndef ZENO_DEFINITION_H
#define ZENO_DEFINITION_H

struct KeyPosition {
    aiVector3D position;
    float timeStamp;
};

struct KeyRotation {
    aiQuaternion orientation;
    float timeStamp;
};

struct KeyScale {
    aiVector3D scale;
    float timeStamp;
};

struct Bone {
    std::vector<KeyPosition> m_Positions;
    std::vector<KeyRotation> m_Rotations;
    std::vector<KeyScale> m_Scales;

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

            KeyPosition data;
            data.position = aiPosition;
            data.timeStamp = timeStamp;
            m_Positions.push_back(data);
        }

        m_NumRotations = channel->mNumRotationKeys;
        for (int rotationIndex = 0; rotationIndex < m_NumRotations; ++rotationIndex) {
            aiQuaternion aiOrientation = channel->mRotationKeys[rotationIndex].mValue;
            float timeStamp = channel->mRotationKeys[rotationIndex].mTime;

            KeyRotation data;
            data.orientation = aiOrientation;
            data.timeStamp = timeStamp;
            m_Rotations.push_back(data);
        }

        m_NumScalings = channel->mNumScalingKeys;
        for (int keyIndex = 0; keyIndex < m_NumScalings; ++keyIndex) {
            aiVector3D scale = channel->mScalingKeys[keyIndex].mValue;
            float timeStamp = channel->mScalingKeys[keyIndex].mTime;

            KeyScale data;
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

struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};

struct VertexInfo{
    aiVector3D position;
    aiVector3D texCoord;
    aiVector3D normal;
    std::unordered_map<std::string, float> boneWeights;
};

struct BoneInfo {
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
    std::unordered_map<std::string, Bone> BoneMap;
};

struct FBXData : zeno::IObjectClone<FBXData>{
    float duration;
    float tick;
    std::vector<VertexInfo> vertices;
    std::vector<unsigned int> indices;
    std::unordered_map<std::string, BoneInfo> BoneOffsetMap;
};

#endif //ZENO_DEFINITION_H