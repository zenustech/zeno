#include "assimp/scene.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"

#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>

#include <stack>
#include <string>
#include <unordered_map>

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#define MAX_BONE_INFLUENCE 16

struct VertexInfo{
    aiVector3D position;
    aiVector3D texCoord;
    aiVector3D normal;
    std::vector<int> boneIds;
    std::vector<float> boneWeights;
};

struct BoneInfo {
    int id;
    aiMatrix4x4 offset;
};

struct NodeInfo{
    aiMatrix4x4 transformation;
    std::string name;
    int childrenCount;
    std::vector<NodeInfo> children;
};

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
    Bone(std::string& name, int BoneID, const aiNodeAnim* channel){
        m_Name = name;
        m_BoneID = BoneID;
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

    std::vector<KeyPosition> m_Positions;
    std::vector<KeyRotation> m_Rotations;
    std::vector<KeyScale> m_Scales;

    int m_NumPositions;
    int m_NumRotations;
    int m_NumScalings;

    aiMatrix4x4 m_LocalTransform;
    std::string m_Name;
    int m_BoneID;
};

struct Mesh{
    std::vector<VertexInfo> vertices;
    std::vector<unsigned int> indices;
    std::unordered_map<std::string, BoneInfo> m_BoneOffset;
    std::unordered_map<std::string, std::vector<unsigned int>> m_VerticesSlice;
    std::unordered_map<std::string, aiMatrix4x4> m_TransMatrix;
    unsigned int m_IndexIncrease = 0;
    int m_BoneCount = 0;

    void initMesh(const aiScene *scene){
        m_IndexIncrease = 0;

        readTrans(scene->mRootNode, aiMatrix4x4());
        processNode(scene->mRootNode, scene);
    }

    void readTrans(const aiNode * parentNode, aiMatrix4x4 parentTransform){
        unsigned int childrenCount = parentNode->mNumChildren;
        aiMatrix4x4 transformation = parentNode->mTransformation;

        std::string name( parentNode->mName.data ) ;
        if (m_TransMatrix.find(name) == m_TransMatrix.end() && parentNode->mNumMeshes){
            m_TransMatrix[name] = aiMatrix4x4();
        }

        transformation = parentTransform * transformation;

        if(parentNode->mNumMeshes){
            m_TransMatrix[name] = transformation * m_TransMatrix[name];
        }

        for (int i = 0; i < childrenCount; i++) {
            readTrans(parentNode->mChildren[i], transformation);
        }
    }

    void processMesh(aiMesh *mesh) {
        std::string meshName(mesh->mName.data);
        for(unsigned int j = 0; j < mesh->mNumVertices; j++){
            aiVector3D vec(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);

            VertexInfo vertexInfo;
            vertexInfo.position = vec;
            vertexInfo.boneIds.resize(MAX_BONE_INFLUENCE);
            vertexInfo.boneWeights.resize(MAX_BONE_INFLUENCE);

            std::fill(vertexInfo.boneIds.begin(), vertexInfo.boneIds.end(), -1);
            std::fill(vertexInfo.boneWeights.begin(), vertexInfo.boneWeights.end(), 0.0f);

            vertices.push_back(vertexInfo);
        }

        for(unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            for(unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j] + m_IndexIncrease);
        }

        extractBone(mesh);

        m_VerticesSlice[meshName] = std::vector<unsigned int>
                {static_cast<unsigned int>(m_IndexIncrease),
                 m_IndexIncrease + mesh->mNumVertices,
                 mesh->mNumBones};

        m_IndexIncrease += mesh->mNumVertices;
    }

    void processNode(aiNode *node, const aiScene *scene){
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMesh(scene->mMeshes[node->mMeshes[i]]);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNode(node->mChildren[i], scene);
    }

    void extractBone(aiMesh* mesh){
        for (int boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex)
        {
            int boneID = -1;
            std::string boneName(mesh->mBones[boneIndex]->mName.C_Str());

            if (m_BoneOffset.find(boneName) == m_BoneOffset.end()) { // notfound
                BoneInfo newBoneInfo;

                newBoneInfo.id = m_BoneCount;
                newBoneInfo.offset = mesh->mBones[boneIndex]->mOffsetMatrix;

                m_BoneOffset[boneName] = newBoneInfo;
                boneID = m_BoneCount;
                m_BoneCount++;
            }
            else {
                boneID = m_BoneOffset[boneName].id;
            }

            assert(boneID != -1);
            auto weights = mesh->mBones[boneIndex]->mWeights;
            unsigned int numWeights = mesh->mBones[boneIndex]->mNumWeights;
            for (int weightIndex = 0; weightIndex < numWeights; ++weightIndex)
            {
                int vertexId = weights[weightIndex].mVertexId + m_IndexIncrease;
                float weight = weights[weightIndex].mWeight;

                auto& vertex = vertices[vertexId];
                for(int i=0;i<MAX_BONE_INFLUENCE; i++){
                    if(vertex.boneIds[i] < 0){
                        vertex.boneIds[i] = boneID;
                        vertex.boneWeights[i] = weight;
                        break;
                    }
                }
            }
        }
    }

    void processTrans(std::unordered_map<std::string , BoneInfo>& boneInfo) {
        for(auto& iter: m_VerticesSlice) {
            std::string meshName = iter.first;
            std::vector<unsigned int> verSlice = iter.second;
            unsigned int verStart = verSlice[0];
            unsigned int verEnd = verSlice[1];
            unsigned int verBoneNum = verSlice[2];
            bool foundMeshBone = boneInfo.find(meshName) != boneInfo.end();

            int meshBoneId = -1;
            float meshBoneWeight = 0.0f;
            if(foundMeshBone){
                meshBoneId = boneInfo[meshName].id;
                meshBoneWeight = 1.0f;
            }
            zeno::log_info("SetupTrans: {} {} {} : {} {} {}", iter.first, foundMeshBone, verBoneNum, iter.second[0], iter.second[1], iter.second[2]);

            for(unsigned int i=verStart;i<verEnd;i++){
                if(verBoneNum == 0){
                    auto & vertex = vertices[i];

                    if(foundMeshBone){
                        vertex.boneIds[0] = meshBoneId;
                        vertex.boneWeights[0] = meshBoneWeight;
                    }else
                    {
                        vertex.position = m_TransMatrix[meshName] * vertices[i].position;
                    }
                }
            }
        }
    }

    void finalProcess(std::vector<zeno::vec3f> &ver,
                      std::vector<zeno::vec3i> &ind,
                      std::vector<aiMatrix4x4>& transforms
    ){

        for(unsigned int i=0; i<vertices.size(); i++){
            auto& bid = vertices[i].boneIds;
            auto& bwe = vertices[i].boneWeights;
            auto& pos = vertices[i].position;

            glm::vec4 tpos(0.0f, 0.0f, 0.0f, 0.0f);

            bool infd = false;
            for(unsigned int j=0; j<MAX_BONE_INFLUENCE; j++){
                if(bid[j] == -1) {
                    continue;
                }
                infd = true;
                auto& tr = transforms[bid[j]];
                glm::mat4 trans = glm::mat4(tr.a1,tr.b1,tr.c1,tr.d1,
                                            tr.a2,tr.b2,tr.c2,tr.d2,
                                            tr.a3,tr.b3,tr.c3,tr.d3,
                                            tr.a4,tr.b4,tr.c4,tr.d4);
                glm::vec4 lpos = trans * glm::vec4(pos.x, pos.y, pos.z, 1.0f);
                tpos += lpos * bwe[j];
            }
            if(! infd)
                tpos = glm::vec4(pos.x, pos.y, pos.z, 1.0f);

            glm::vec3 fpos = glm::vec3(tpos.x/tpos.w, tpos.y/tpos.w, tpos.z/tpos.w);
            ver.emplace_back(fpos.x, fpos.y, fpos.z);
        }

        for(unsigned int i=0; i<indices.size(); i+=3){
            zeno::vec3i incs(indices[i],indices[i+1],indices[i+2]);
            ind.push_back(incs);
        }
    }
};

struct Anim{
    double m_Duration;
    double m_TicksPerSecond;
    int frame;
    float m_CurrentFrame;
    float m_DeltaTime;
    NodeInfo m_RootNode;
    std::vector<Bone> m_Bones;
    std::vector<aiMatrix4x4> m_Transforms;
    std::unordered_map<std::string, BoneInfo> m_BoneInfoMap;

    void initAnim(aiScene const*scene, Mesh* model){

        readHierarchyData(m_RootNode, scene->mRootNode);

        if(scene->mNumAnimations){
            auto animation = scene->mAnimations[0];
            m_Duration = animation->mDuration;

            m_TicksPerSecond = animation->mTicksPerSecond;
            setupBones(animation, model);
        }

        m_CurrentFrame = 0.0;
        m_Transforms.resize(model->m_BoneCount);
    }

    void readHierarchyData(NodeInfo &dest, const aiNode *src) {
        dest.name = std::string(src->mName.data);
        dest.transformation = src->mTransformation;
        dest.childrenCount = src->mNumChildren;

        for (int i = 0; i < src->mNumChildren; i++) {
            NodeInfo newData;
            readHierarchyData(newData, src->mChildren[i]);
            dest.children.push_back(newData);
        }
    }

    void setupBones(const aiAnimation *animation, Mesh* model) {
        int size = animation->mNumChannels;
        zeno::log_info("SetupBones: Num Channels {}", size);

        auto& boneInfoMap = model->m_BoneOffset;
        int& boneCount = model->m_BoneCount;

        for (int i = 0; i < size; i++) {
            auto channel = animation->mChannels[i];
            std::string boneName(channel->mNodeName.data);

            if (boneInfoMap.find(boneName) == boneInfoMap.end())
            {
                boneInfoMap[boneName].id = boneCount;
                boneCount++;
            }
            m_Bones.push_back(Bone(boneName, boneInfoMap[channel->mNodeName.data].id, channel));
        }

        m_BoneInfoMap = boneInfoMap;
    }

    Bone *findBone(std::string const& name) {
        auto iter = std::find_if(m_Bones.begin(), m_Bones.end(),
                                 [&](const Bone& Bone)
                                 {
                                     return Bone.m_Name == name;
                                 }
        );
        if (iter == m_Bones.end())
            return nullptr;
        else
            return &(*iter);
    }

    void updateAnimation(float dt) {
        m_DeltaTime = dt;

        m_CurrentFrame += m_TicksPerSecond * dt;
        m_CurrentFrame = fmod(m_CurrentFrame, m_Duration);

        calculateBoneTransform(&m_RootNode, aiMatrix4x4());
    }

    void calculateBoneTransform(const NodeInfo *node, aiMatrix4x4 parentTransform) {
        std::string nodeName = node->name;
        aiMatrix4x4 nodeTransform = node->transformation;

        Bone* bone = findBone(nodeName);

        if (bone) {
            bone->update(m_CurrentFrame);
            nodeTransform = bone->m_LocalTransform;
        }

        aiMatrix4x4 globalTransformation = parentTransform * nodeTransform;

        if (m_BoneInfoMap.find(nodeName) != m_BoneInfoMap.end()) {  // found
            int index = m_BoneInfoMap[nodeName].id;
            aiMatrix4x4 offset = m_BoneInfoMap[nodeName].offset;
            m_Transforms[index] = globalTransformation * offset;
        }
        for (int i = 0; i < node->childrenCount; i++)
            calculateBoneTransform(&node->children[i], globalTransformation);
    }
};

void readFBXFile(
        std::vector<zeno::vec3f> &vertices,
        std::vector<zeno::vec3i> &indices,
        const char *fbx_path,
        int frame
)
{
    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_PTV_NORMALIZE, true);
    aiScene const* scene = importer.ReadFile(fbx_path,
                                             aiProcess_Triangulate
                                             | aiProcess_FlipUVs
                                             | aiProcess_CalcTangentSpace
                                             | aiProcess_JoinIdenticalVertices);
    if(! scene)
        zeno::log_error("ReadFBXPrim: Invalid assimp scene");

    Mesh mesh;
    Anim anim;

    mesh.initMesh(scene);
    anim.initAnim(scene, &mesh);
    mesh.processTrans(anim.m_BoneInfoMap);
    anim.updateAnimation(frame/24.0f);

    mesh.finalProcess(vertices, indices, anim.m_Transforms);

    zeno::log_info("ReadFBXPrim: Num Animation {}", scene->mNumAnimations);
    zeno::log_info("ReadFBXPrim: Vertices count {}", mesh.vertices.size());
    zeno::log_info("ReadFBXPrim: Indices count {}", mesh.indices.size());
    zeno::log_info("ReadFBXPrim: Bone count {}", mesh.m_BoneCount);
    zeno::log_info("ReadFBXPrim: readFBXFile done.");
}

struct ReadFBXPrim : zeno::INode {

    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input<zeno::NumericObject>("frameid")->get<int>();
        } else {
            frameid = zeno::state.frameid;
        }

        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::shared_ptr<zeno::DictObject> prims = std::make_shared<zeno::DictObject>();
        auto &pos = prim->verts;
        auto &tris = prim->tris;

        zeno::log_info("ReadFBXPrim: path {}", path);
        zeno::log_info("ReadFBXPrim: frameid {}", frameid);

        readFBXFile(pos, tris, path.c_str(), frameid);

        set_output("prim", std::move(prim));
        set_output("dict", std::move(prims));
    }
};

ZENDEFNODE(ReadFBXPrim,
           {       /* inputs: */
               {
                   {"readpath", "path"},
                   {"frameid"}
               },  /* outputs: */
               {
                   "prim", "dict",
               },  /* params: */
               {

               },  /* category: */
               {
                   "primitive",
               }
           });
