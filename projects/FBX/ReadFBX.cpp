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
    std::unordered_map<std::string, std::vector<unsigned int>> m_VerticesSlice;
    std::unordered_map<std::string, aiMatrix4x4> m_TransMatrix;
    unsigned int m_VerticesIncrease = 0;
    unsigned int m_IndicesIncrease = 0;

    void initMesh(const aiScene *scene){
        m_VerticesIncrease = 0;

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

    void processMesh(aiMesh *mesh, const aiScene *scene) {
        std::string meshName(mesh->mName.data);

        // Vertices
        for(unsigned int j = 0; j < mesh->mNumVertices; j++){
            SVertex vertexInfo;

            aiVector3D vec(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            vertexInfo.position = vec;

            if (mesh->mTextureCoords[0]){
                aiVector3D uvw(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y, 0.0f);
                vertexInfo.texCoord = uvw;
            }
            if (mesh->mNormals) {
                aiVector3D normal(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);
                vertexInfo.normal = normal;
            }
            if(mesh->mTangents){
                aiVector3D tangent(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z);
                vertexInfo.tangent = tangent;
            }
            if(mesh->mBitangents){
                aiVector3D bitangent(mesh->mBitangents[j].x, mesh->mBitangents[j].y, mesh->mBitangents[j].z);
                vertexInfo.bitangent = bitangent;
            }

            fbxData.iVertices.value.push_back(vertexInfo);
        }

        // Indices
        for(unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            for(unsigned int j = 0; j < face.mNumIndices; j++)
                fbxData.iIndices.value.push_back(face.mIndices[j] + m_VerticesIncrease);
        }

        // Material
        if(mesh->mNumVertices)
            readMaterial(mesh, scene);

        // FBXBone
        extractBone(mesh);

        m_VerticesSlice[meshName] = std::vector<unsigned int>
                {static_cast<unsigned int>(m_VerticesIncrease),  // Vert Start
                 m_VerticesIncrease + mesh->mNumVertices,  // Vert End
                 mesh->mNumBones,
                 m_IndicesIncrease,  // Indices Start
                 static_cast<unsigned int>(m_IndicesIncrease + (fbxData.iIndices.value.size() - m_IndicesIncrease))  // Indices End
                 };

        m_IndicesIncrease += (fbxData.iIndices.value.size() - m_IndicesIncrease);
        m_VerticesIncrease += mesh->mNumVertices;
    }

    void processNode(aiNode *node, const aiScene *scene){

        //zeno::log_info("Node: Name {}", node->mName.data);
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMesh(scene->mMeshes[node->mMeshes[i]], scene);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNode(node->mChildren[i], scene);
    }

    void readMaterial(aiMesh* mesh, aiScene const* scene){
        SMaterial mat;
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        GET_MAT_COLOR(mat.base, "$ai.base",0,0, aiColor4D(0.0f, 0.0f, 0.0f, 0.0f))
        GET_MAT_COLOR(mat.specular, "$ai.specular",0,0, aiColor4D(0.0f, 0.0f, 0.0f, 0.0f))
        GET_MAT_COLOR(mat.transmission, "$ai.transmission",0,0, aiColor4D(0.0f, 0.0f, 0.0f, 0.0f))
        GET_MAT_COLOR(mat.subsurface, "$ai.subsurface",0,0, aiColor4D(0.0f, 0.0f, 0.0f, 0.0f))
        GET_MAT_COLOR(mat.sheen, "$ai.sheen",0,0, aiColor4D(0.0f, 0.0f, 0.0f, 0.0f))
        GET_MAT_COLOR(mat.coat, "$ai.coat",0,0, aiColor4D(0.0f, 0.0f, 0.0f, 0.0f))
        GET_MAT_COLOR(mat.emission, "$ai.emission",0,0, aiColor4D(0.0f, 0.0f, 0.0f, 0.0f))

        // TODO read material-float properties
//        if(AI_SUCCESS != aiGetMaterialFloat(material, "$ai.normalCameraFactor", 0, 0, &mat.testFloat))
//            mat.testFloat = 0.5f;
//        zeno::log_info("----- TestFloat {}", mat.testFloat);

        zeno::log_info("MatTex: Mesh {} Material {} NumTex {}",
                       mesh->mName.data, material->GetName().data, scene->mNumTextures);

        for(int texTypeIndex=0; texTypeIndex<=AI_TEXTURE_TYPE_MAX; texTypeIndex++)
        {
            std::vector<STexture> textures;
            aiTextureType texType = (aiTextureType)texTypeIndex;

            for(unsigned int i = 0; i < material->GetTextureCount(texType); i++)
            {
                aiString str;
                material->GetTexture(texType, i, &str);
                zeno::log_info(">>>>> MatTex: TexName {} TexType {}", str.data, texType);

                    STexture texture;
                    texture.type = texTypeIndex;
                    texture.path = str.C_Str();
                    textures.push_back(texture);
            }

            mat.tex[texTypeIndex] = textures;
        }

        mat.matName = material->GetName().data;

        fbxData.iMaterial.value[mesh->mName.data] = mat;
    }

    void extractBone(aiMesh* mesh){
        for (int boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex)
        {
            std::string boneName(mesh->mBones[boneIndex]->mName.C_Str());

            // Not Found, Create one, If Found, will have same offset-matrix
            if (fbxData.iBoneOffset.value.find(boneName) == fbxData.iBoneOffset.value.end()) {
                SBoneOffset newBoneInfo;

                newBoneInfo.name = boneName;
                newBoneInfo.offset = mesh->mBones[boneIndex]->mOffsetMatrix;

                fbxData.iBoneOffset.value[boneName] = newBoneInfo;
            }

            auto weights = mesh->mBones[boneIndex]->mWeights;
            unsigned int numWeights = mesh->mBones[boneIndex]->mNumWeights;
            for (int weightIndex = 0; weightIndex < numWeights; ++weightIndex)
            {
                int vertexId = weights[weightIndex].mVertexId + m_VerticesIncrease;
                float weight = weights[weightIndex].mWeight;

                auto& vertex = fbxData.iVertices.value[vertexId];
                vertex.boneWeights[boneName] = weight;
            }
        }
    }

    void processTrans(std::unordered_map<std::string, SAnimBone>& bones,
                      std::shared_ptr<zeno::DictObject>& prims,
                      std::shared_ptr<zeno::DictObject>& datas) {
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

            for(unsigned int i=verStart; i<verEnd; i++){
                if(verBoneNum == 0){
                    auto & vertex = fbxData.iVertices.value[i];

                    if(foundMeshBone){
                        //vertex.boneIds[0] = meshBoneId;
                        //vertex.boneWeights[0] = meshBoneWeight;
                    }else
                    {
                        vertex.position = m_TransMatrix[meshName] * fbxData.iVertices.value[i].position;
                    }
                }
            }

            // Sub-prims (applied node transform)
            auto sub_prim = std::make_shared<zeno::PrimitiveObject>();
            auto sub_data = std::make_shared<FBXData>();
            std::vector<SVertex> sub_vertices;
            std::vector<unsigned int> sub_indices;

            for(unsigned int i=indicesStart; i<indicesEnd; i+=3){
                auto i1 = fbxData.iIndices.value[i]-verStart;
                auto i2 = fbxData.iIndices.value[i+1]-verStart;
                auto i3 = fbxData.iIndices.value[i+2]-verStart;
                zeno::vec3i incs(i1, i2, i3);
                sub_prim->tris.push_back(incs);
                sub_indices.push_back(i1);
                sub_indices.push_back(i2);
                sub_indices.push_back(i3);
            }
            for(unsigned int i=verStart; i< verEnd; i++){
                sub_prim->verts.emplace_back(fbxData.iVertices.value[i].position.x,
                                             fbxData.iVertices.value[i].position.y,
                                             fbxData.iVertices.value[i].position.z);
                sub_vertices.push_back(fbxData.iVertices.value[i]);
            }
            sub_data->iIndices.value = sub_indices;
            sub_data->iVertices.value = sub_vertices;
            sub_data->iBoneOffset = fbxData.iBoneOffset;

            prims->lut[meshName] = sub_prim;
            datas->lut[meshName] = sub_data;
        }
    }

    void processPrim(std::shared_ptr<zeno::PrimitiveObject>& prim){
        auto &ver = prim->verts;
        auto &ind = prim->tris;
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");

        for(unsigned int i=0; i<fbxData.iVertices.value.size(); i++){
            auto& vpos = fbxData.iVertices.value[i].position;
            auto& vnor = fbxData.iVertices.value[i].normal;
            auto& vuv = fbxData.iVertices.value[i].texCoord;
            ver.emplace_back(vpos.x, vpos.y, vpos.z);
            uv.emplace_back(vuv.x, vuv.y, vuv.z);
            norm.emplace_back(vnor.x, vnor.y, vnor.z);
        }

        for(unsigned int i=0; i<fbxData.iIndices.value.size(); i+=3){
            zeno::vec3i incs(fbxData.iIndices.value[i],
                             fbxData.iIndices.value[i+1],
                             fbxData.iIndices.value[i+2]);
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
        //zeno::log_info("SetupBones: Num Channels {}", size);

        for (int i = 0; i < size; i++) {
            auto channel = animation->mChannels[i];
            std::string boneName(channel->mNodeName.data);
            //zeno::log_info("----- Name {}", boneName);

            SAnimBone bone;
            bone.initBone(boneName, channel);

            m_Bones.AnimBoneMap[boneName] = bone;
        }

        zeno::log_info("SetupBones: Num AnimBoneMap: {}", m_Bones.AnimBoneMap.size());
    }

};

void readFBXFile(
        std::shared_ptr<zeno::PrimitiveObject>& prim,
        std::shared_ptr<zeno::DictObject>& prims,
        std::shared_ptr<zeno::DictObject>& datas,
        std::shared_ptr<NodeTree>& nodeTree,
        std::shared_ptr<FBXData>& fbxData,
        std::shared_ptr<BoneTree>& boneTree,
        std::shared_ptr<AnimInfo>& animInfo,
        const char *fbx_path)
{
    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_PTV_NORMALIZE, true);
    aiScene const* scene = importer.ReadFile(fbx_path,
                                             aiProcess_Triangulate
                                             | aiProcess_FlipUVs
                                             | aiProcess_CalcTangentSpace
                                             | aiProcess_JoinIdenticalVertices
                                             );
    if(! scene)
        zeno::log_error("ReadFBXPrim: Invalid assimp scene");

    Mesh mesh;
    Anim anim;

    mesh.initMesh(scene);
    anim.initAnim(scene, &mesh);
    mesh.processTrans(anim.m_Bones.AnimBoneMap, prims, datas);
    mesh.processPrim(prim);

    *fbxData = mesh.fbxData;
    *nodeTree = anim.m_RootNode;
    *boneTree = anim.m_Bones;

    animInfo->duration = anim.duration;
    animInfo->tick = anim.tick;

    zeno::log_info("ReadFBXPrim: Num Animation {}", scene->mNumAnimations);
    zeno::log_info("ReadFBXPrim: Vertices count {}", mesh.fbxData.iVertices.value.size());
    zeno::log_info("ReadFBXPrim: Indices count {}", mesh.fbxData.iIndices.value.size());
}

struct ReadFBXPrim : zeno::INode {

    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::shared_ptr<zeno::DictObject> prims = std::make_shared<zeno::DictObject>();
        std::shared_ptr<zeno::DictObject> datas = std::make_shared<zeno::DictObject>();
        auto nodeTree = std::make_shared<NodeTree>();
        auto animInfo = std::make_shared<AnimInfo>();
        auto fbxData = std::make_shared<FBXData>();
        auto boneTree = std::make_shared<BoneTree>();

        zeno::log_info("ReadFBXPrim: File Path {}", path);

        readFBXFile(prim,prims, datas,
                    nodeTree, fbxData, boneTree, animInfo,
                    path.c_str());

        set_output("prim", std::move(prim));
        set_output("prims", std::move(prims));
        set_output("data", std::move(fbxData));
        set_output("datas", std::move(datas));
        set_output("animinfo", std::move(animInfo));
        set_output("nodetree", std::move(nodeTree));
        set_output("bonetree", std::move(boneTree));
    }
};

ZENDEFNODE(ReadFBXPrim,
           {       /* inputs: */
               {
                   {"readpath", "path"},
                   {"frameid"}
               },  /* outputs: */
               {
                   "prim", "prims", "data", "datas",
                   {"AnimInfo", "animinfo"},
                   {"NodeTree", "nodetree"},
                   {"BoneTree", "bonetree"},
               },  /* params: */
               {

               },  /* category: */
               {
                   "primitive",
               }
           });
