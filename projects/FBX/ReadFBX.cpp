#include "assimp/scene.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
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

#include <stb_image.h>

#include "Definition.h"


struct Mesh{
    FBXData fbxData;
    BoneOffset boneOffset;
    std::vector<Texture> textures_loaded;
    std::unordered_map<std::string, std::vector<unsigned int>> m_VerticesSlice;
    std::unordered_map<std::string, aiMatrix4x4> m_TransMatrix;
    unsigned int m_VerticesIncrease = 0;
    unsigned int m_IndicesIncrease = 0;
    int m_BoneCount = 0;

    void initMesh(const aiScene *scene){
        m_VerticesIncrease = 0;

        readTrans(scene->mRootNode, aiMatrix4x4());

        // DEBUG
//        zeno::log_info("Mesh: Read Trans Done.");
//        for(auto& t:m_TransMatrix){
//            zeno::log_info(">>>>> Trans Name {}", t.first);
//        }
//        zeno::log_info("\n");

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

    void processMesh(aiMesh *mesh, const aiScene *scene) {
        std::string meshName(mesh->mName.data);
        std::vector<Texture> textures;

        // Vertices
        for(unsigned int j = 0; j < mesh->mNumVertices; j++){
            aiVector3D vec(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);

            VertexInfo vertexInfo;
            vertexInfo.position = vec;

            fbxData.vertices.push_back(vertexInfo);
        }

        // Indices
        for(unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            for(unsigned int j = 0; j < face.mNumIndices; j++)
                fbxData.indices.push_back(face.mIndices[j] + m_VerticesIncrease);
        }

        // Material
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
//        zeno::log_info("MatTex: Mesh {} Material {}", meshName, material->GetName().data);
        std::vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

        // Bone
        extractBone(mesh);

        // DEBUG
//        zeno::log_info("Mesh: Bone extract with offset Matrix. Size {}", m_BoneOffset.size());
//        for(auto&b: m_BoneOffset){
//            zeno::log_info(">>>>> Bone Name {}", b.first);
//        }
//        zeno::log_info("\n");

        m_VerticesSlice[meshName] = std::vector<unsigned int>
                {static_cast<unsigned int>(m_VerticesIncrease),  // Vert Start
                 m_VerticesIncrease + mesh->mNumVertices,  // Vert End
                 mesh->mNumBones,
                 m_IndicesIncrease,  // Indices Start
                 static_cast<unsigned int>(m_IndicesIncrease + (fbxData.indices.size() - m_IndicesIncrease))  // Indices End
                 };

        m_IndicesIncrease += (fbxData.indices.size() - m_IndicesIncrease);
        m_VerticesIncrease += mesh->mNumVertices;
    }

    void processNode(aiNode *node, const aiScene *scene){

        //zeno::log_info("Node: Name {}", node->mName.data);
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMesh(scene->mMeshes[node->mMeshes[i]], scene);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNode(node->mChildren[i], scene);
    }

    std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName)
    {
        std::vector<Texture> textures;
        for(unsigned int i = 0; i < mat->GetTextureCount(type); i++)
        {
            aiString str;
            mat->GetTexture(type, i, &str);
//            zeno::log_info("MatTex: Texture Name {}", str.data);

            bool skip = false;
            for(unsigned int j = 0; j < textures_loaded.size(); j++)
            {
                if(std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)  // Compare two strings
                {
                    textures.push_back(textures_loaded[j]);
                    skip = true;
                    break;
                }
            }
            if(! skip)
            {
                Texture texture;
                texture.id;
                texture.type = typeName;
                texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);
            }
        }
        return textures;
    }

    void extractBone(aiMesh* mesh){
        for (int boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex)
        {
            std::string boneName(mesh->mBones[boneIndex]->mName.C_Str());
            //zeno::log_info("Extracting {}", boneName);
            // Not Found, Create one, If Found, will have same offset-matrix
            if (boneOffset.BoneOffsetMap.find(boneName) == boneOffset.BoneOffsetMap.end()) {
                BoneInfo newBoneInfo;

                newBoneInfo.name = boneName;
                newBoneInfo.offset = mesh->mBones[boneIndex]->mOffsetMatrix;

                boneOffset.BoneOffsetMap[boneName] = newBoneInfo;
            }

            auto weights = mesh->mBones[boneIndex]->mWeights;
            unsigned int numWeights = mesh->mBones[boneIndex]->mNumWeights;
            for (int weightIndex = 0; weightIndex < numWeights; ++weightIndex)
            {
                int vertexId = weights[weightIndex].mVertexId + m_VerticesIncrease;
                float weight = weights[weightIndex].mWeight;

                auto& vertex = fbxData.vertices[vertexId];
                vertex.boneWeights[boneName] = weight;
            }
        }
    }

    void processTrans(std::unordered_map<std::string, Bone>& bones,
                      std::shared_ptr<zeno::DictObject>& prims) {
        for(auto& iter: m_VerticesSlice) {
            std::string meshName = iter.first;
            std::vector<unsigned int> verSlice = iter.second;
            unsigned int verStart = verSlice[0];
            unsigned int verEnd = verSlice[1];
            unsigned int verBoneNum = verSlice[2];
            unsigned int indicesStart = verSlice[3];
            unsigned int indicesEnd = verSlice[4];

            // TODO full support blend bone-animation and mesh-animation, See SimTrans.fbx
            bool foundMeshBone = bones.find(meshName) != bones.end();
            /*
            int meshBoneId = -1;
            float meshBoneWeight = 0.0f;
            if(foundMeshBone){
                meshBoneId = boneInfo[meshName].id;
                meshBoneWeight = 1.0f;
            }
             */

            // DEBUG
//            zeno::log_info("SetupTrans: {} {} : {} {} {} {} {}",
//                           iter.first, foundMeshBone,
//                           iter.second[0], iter.second[1], iter.second[2], iter.second[3], iter.second[4]);

            for(unsigned int i=verStart; i<verEnd; i++){
                if(verBoneNum == 0){
                    auto & vertex = fbxData.vertices[i];

                    if(foundMeshBone){
                        //vertex.boneIds[0] = meshBoneId;
                        //vertex.boneWeights[0] = meshBoneWeight;
                    }else
                    {
                        vertex.position = m_TransMatrix[meshName] * fbxData.vertices[i].position;
                    }
                }
            }

            // Sub-prims (apply node transform)
            auto sub_prim = std::make_shared<zeno::PrimitiveObject>();
            for(unsigned int i=indicesStart; i<indicesEnd; i+=3){
                zeno::vec3i incs(fbxData.indices[i]-verStart,
                                 fbxData.indices[i+1]-verStart,
                                 fbxData.indices[i+2]-verStart);
                sub_prim->tris.push_back(incs);
            }
            for(unsigned int i=verStart; i< verEnd; i++){
                sub_prim->verts.emplace_back(fbxData.vertices[i].position.x,
                                             fbxData.vertices[i].position.y,
                                             fbxData.vertices[i].position.z);
            }
            prims->lut[meshName] = sub_prim;
        }
    }

    void processPrim(std::vector<zeno::vec3f> &ver,
                      std::vector<zeno::vec3i> &ind
    ){

        for(unsigned int i=0; i<fbxData.vertices.size(); i++){
            auto& vpos = fbxData.vertices[i].position;
            ver.emplace_back(vpos.x, vpos.y, vpos.z);
        }

        for(unsigned int i=0; i<fbxData.indices.size(); i+=3){
            zeno::vec3i incs(fbxData.indices[i],
                             fbxData.indices[i+1],
                             fbxData.indices[i+2]);
            ind.push_back(incs);
        }
    }
};

struct Anim{
    NodeTree m_RootNode;
    BoneTree m_Bones;

    float duration;
    float tick;
    void initAnim(aiScene const*scene, Mesh* model){

        readHierarchyData(m_RootNode, scene->mRootNode);
        //zeno::log_info("----- Anim: Convert AssimpNode.");

        if(scene->mNumAnimations){
            // TODO handle more animation if have
            auto animation = scene->mAnimations[0];
            duration = animation->mDuration;
            tick = animation->mTicksPerSecond;

            setupBones(animation, model);
        }

        // DEBUG
//        zeno::log_info("Anim: Bone Setup, Size {}", m_Bones.BoneMap.size());
//        for(auto&b: m_Bones.BoneMap){
//            zeno::log_info(">>>>> Bone Name {}", b.first);  // <BoneName>_$AssimpFbx$_Translation
//        }
//        zeno::log_info("\n");
    }

    void readHierarchyData(NodeTree &dest, const aiNode *src) {
        dest.name = std::string(src->mName.data);
        dest.transformation = src->mTransformation;
        dest.childrenCount = src->mNumChildren;

        //zeno::log_info("Tree: Name {}", dest.name);

        for (int i = 0; i < src->mNumChildren; i++) {
            NodeTree newData;
            readHierarchyData(newData, src->mChildren[i]);
            dest.children.push_back(newData);
        }
    }

    void setupBones(const aiAnimation *animation, Mesh* model) {
        int size = animation->mNumChannels;
        zeno::log_info("SetupBones: Num Channels {}", size);

//        auto& boneOffset = model->m_BoneOffset;
//        auto& boneOffset = m_Bones.BoneInfoMap;
//        int& boneCount = model->m_BoneCount;

        for (int i = 0; i < size; i++) {
            auto channel = animation->mChannels[i];
            std::string boneName(channel->mNodeName.data);
//            zeno::log_info("----- Name {}", boneName);
//            if (boneOffset.find(boneName) == boneOffset.end())
//            {
//                // TODO Assimp $AssimpFbx$_Translation
//                boneOffset[boneName].name = boneName;
//            }
            Bone bone;
            bone.initBone(boneName, channel);

//            zeno::log_info("Anim Bone {}", boneName);
            m_Bones.BoneMap[boneName] = bone;
        }

//        for(auto& b:m_Bones.BoneMap){
//            zeno::log_info("----- ----- Bone Name {} : {} {} {}", b.first,
//                           b.second.m_NumPositions,
//                           b.second.m_NumRotations,
//                           b.second.m_NumScalings);
//        }
//        zeno::log_warn("------------------------------------");

        // BoneMap: Anims
        // BoneOffset: Joints
        // BoneInfoMap: Joints and Anims
//        m_Bones.BoneInfoMap = boneOffset;

        zeno::log_info("SetupBones: Num BoneMap: {} Num BoneInfoMap {}",
                       m_Bones.BoneMap.size(),
                       m_Bones.BoneInfoMap.size());
    }

};

void readFBXFile(
        std::vector<zeno::vec3f> &vertices,
        std::vector<zeno::vec3i> &indices,
        std::shared_ptr<zeno::DictObject>& prims,
        std::shared_ptr<NodeTree>& nodeTree,
        std::shared_ptr<FBXData>& fbxData,
        std::shared_ptr<BoneTree>& boneTree,
        std::shared_ptr<BoneOffset>& boneOffset,
        const char *fbx_path)
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
    mesh.processTrans(anim.m_Bones.BoneMap, prims);
    mesh.processPrim(vertices, indices);

    *fbxData = mesh.fbxData;
    *nodeTree = anim.m_RootNode;
    *boneTree = anim.m_Bones;
    *boneOffset = mesh.boneOffset;
    fbxData->duration = anim.duration;
    fbxData->tick = anim.tick;

//    zeno::log_info("ReadFBXPrim: Num Animation {}", scene->mNumAnimations);
//    zeno::log_info("ReadFBXPrim: Vertices count {}", mesh.fbxData.vertices.size());
//    zeno::log_info("ReadFBXPrim: Indices count {}", mesh.fbxData.indices.size());
//    zeno::log_info("ReadFBXPrim: Bone count {}", mesh.m_BoneCount);
//    zeno::log_info("ReadFBXPrim: readFBXFile done.");
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
        auto nodeTree = std::make_shared<NodeTree>();
        auto fbxData = std::make_shared<FBXData>();
        auto boneTree = std::make_shared<BoneTree>();
        auto boneOffset = std::make_shared<BoneOffset>();
        auto &pos = prim->verts;
        auto &tris = prim->tris;

        zeno::log_info("ReadFBXPrim: File Path {}", path);
//        zeno::log_info("ReadFBXPrim: frameid {}", frameid);

        readFBXFile(pos, tris,
                    prims,
                    nodeTree, fbxData, boneTree, boneOffset,
                    path.c_str());

        set_output("prim", std::move(prim));
        set_output("dict", std::move(prims));
        set_output("nodetree", std::move(nodeTree));
        set_output("fbxdata", std::move(fbxData));
        set_output("bonetree", std::move(boneTree));
        set_output("boneoffset", std::move(boneOffset));
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
                   {"FBXData", "fbxdata"},
                   {"NodeTree", "nodetree"},
                   {"BoneTree", "bonetree"},
                   {"BoneOffset", "boneoffset"}
               },  /* params: */
               {

               },  /* category: */
               {
                   "primitive",
               }
           });
