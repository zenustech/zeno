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
#include <filesystem>

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <stb_image.h>
//#ifndef ZENO2
    //#define STBI_MSC_SECURE_CRT
    //#define STB_IMAGE_WRITE_IMPLEMENTATION
//#endif
//#include <stb_image_write.h>

#include "Definition.h"

void readFBXFile(
    std::shared_ptr<zeno::PrimitiveObject>& prim,
    std::shared_ptr<zeno::DictObject>& prims,
    std::shared_ptr<zeno::DictObject>& datas,
    std::shared_ptr<NodeTree>& nodeTree,
    std::shared_ptr<FBXData>& fbxData,
    std::shared_ptr<BoneTree>& boneTree,
    std::shared_ptr<AnimInfo>& animInfo,
    const char *fbx_path);

struct Mesh{
    FBXData fbxData;
    std::filesystem::path fbxPath;
    std::unordered_map<std::string, std::vector<unsigned int>> m_VerticesSlice;
    std::unordered_map<std::string, std::vector<unsigned int>> m_BlendShapeSlice;
    std::unordered_map<std::string, aiMatrix4x4> m_TransMatrix;
    unsigned int m_VerticesIncrease = 0;
    unsigned int m_IndicesIncrease = 0;

    std::string createTexDir(std::string subPath){
        auto p = fbxPath;
        p += subPath;

        if(! std::filesystem::exists(p)){
            std::filesystem::create_directory(p);
        }

        return p.string();
    }

    void initMesh(const aiScene *scene){
        m_VerticesIncrease = 0;
        fbxData.iMeshName.value = "__root__";

        createTexDir("valueTex");
        readTrans(scene->mRootNode, aiMatrix4x4());
        processNode(scene->mRootNode, scene);
        // TODO read Animation of Camera property and Light property,
        // e.g. FocalLength LightIntensity
        processCamera(scene);
        readLights(scene);
    }

    void readLights(const aiScene *scene){
        zeno::log_info("Num Light {}", scene->mNumLights);

        for(unsigned int i=0; i<scene->mNumLights; i++){
            aiLight* l = scene->mLights[i];
            std::string lightName = l->mName.data;
            // TODO support to import light
            // Except the CD is valid property we read.
            // aiLight -> Model: 1019948832, "Model::aiAreaLight1", "Null"
            // mayaLight -> Model: 911159248, "Model::ambientLight1", "Light"
            // So we can't import aiLight
            // In maya, export `aiAreaLight1` to .fbx -> import the .fbx
            // `aiAreaLight1` -> Changed to a transform-node
            zeno::log_info("Light N {} T {} P {} {} {} S {} {}\nD {} {} {} U {} {} {}"
                           " AC {} AL {} AQ {} CD {} {} {} CS {} {} {} CA {} {} {}"
                           " AI {} AO {}",
                           lightName, l->mType, l->mPosition.x, l->mPosition.y, l->mPosition.z,
                           l->mSize.x, l->mSize.y,
                           l->mDirection.x, l->mDirection.y, l->mDirection.z, l->mUp.x, l->mUp.y, l->mUp.z,
                           l->mAttenuationConstant, l->mAttenuationLinear, l->mAttenuationQuadratic,
                           l->mColorDiffuse.r, l->mColorDiffuse.g, l->mColorDiffuse.b,
                           l->mColorSpecular.r, l->mColorSpecular.g, l->mColorSpecular.b,
                           l->mColorAmbient.r, l->mColorAmbient.g, l->mColorAmbient.b,
                           l->mAngleInnerCone, l->mAngleOuterCone
                           );
            SLight sLig{
                lightName, l->mType, l->mPosition, l->mDirection, l->mUp,
                l->mAttenuationConstant, l->mAttenuationLinear, l->mAttenuationQuadratic,
                l->mColorDiffuse, l->mColorSpecular, l->mColorAmbient,
                l->mAngleInnerCone, l->mAngleOuterCone, l->mSize
            };

            fbxData.iLight.value[lightName] = sLig;
        }
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
        auto numAnimMesh = mesh->mNumAnimMeshes;

        // Vertices & BlendShape
        for(unsigned int j = 0; j < mesh->mNumVertices; j++){
            SVertex vertexInfo;

            aiVector3D vec(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            vertexInfo.position = vec;

            if (mesh->mTextureCoords[0]){
                aiVector3D uvw(fmodf(mesh->mTextureCoords[0][j].x, 1.0f),
                               fmodf(mesh->mTextureCoords[0][j].y, 1.0f), 0.0f);
                // Same vert but diff uv
                // U 0.980281 0.0276042 V -0.5 -1 -0.866026
                // U 0.0325739 0.0276042 V -0.5 -1 -0.866026
                //zeno::log_info(">>>>> U {} {} V {} {} {}", uvw.x, uvw.y, vec.x, vec.y, vec.z);
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

            // TODO Support more color channel
            if(mesh->HasVertexColors(0)){
                aiColor4D cls(mesh->mColors[0][j].r, mesh->mColors[0][j].g,
                              mesh->mColors[0][j].b, mesh->mColors[0][j].a);
                vertexInfo.vectexColor = cls;
            }else{
                vertexInfo.vectexColor = aiColor4D(0, 0, 0, 0);
            }

            fbxData.iVertices.value.push_back(vertexInfo);
        }

        // BlendShape
        if(numAnimMesh){
            // The number of vertices should be the same no matter how many anim-meshes there are
            // So let's take the first one
            unsigned int bsNumVert = mesh->mAnimMeshes[0]->mNumVertices;

            zeno::log_info("BS MeshName {} NumAnim {}", meshName, numAnimMesh);

            m_BlendShapeSlice[meshName] =
                    std::vector<unsigned int>
                            {
                            };

            std::vector<std::vector<SBSVertex>> blendShapeData;
            blendShapeData.resize(numAnimMesh);

            for(unsigned int a=0; a<numAnimMesh; a++){
                auto& animMesh = mesh->mAnimMeshes[a];
                unsigned int aNumV = animMesh->mNumVertices;

                blendShapeData[a].resize(aNumV);

                for(unsigned int i=0; i<aNumV; i++){
                    SBSVertex sbsVertex;

                    sbsVertex.position = animMesh->mVertices[i];
                    sbsVertex.normal = animMesh->mNormals[i];
                    sbsVertex.deltaPosition = sbsVertex.position
                            - fbxData.iVertices.value[i+m_VerticesIncrease].position;
                    sbsVertex.deltaNormal = sbsVertex.normal
                            - fbxData.iVertices.value[i+m_VerticesIncrease].normal;
                    sbsVertex.weight = animMesh->mWeight;

                    blendShapeData[a][i] = sbsVertex;
                }
            }
            fbxData.iBlendSData.value[meshName] = blendShapeData;
        }

        // Indices
        for(unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            //zeno::log_info("-----");
            for(unsigned int j = 0; j < face.mNumIndices; j++) {
                fbxData.iIndices.value.push_back(face.mIndices[j] + m_VerticesIncrease);
                //zeno::log_info(" {}", face.mIndices[j] + m_VerticesIncrease);
            }
        }

        // Material
        if(mesh->mNumVertices)
            readMaterial(mesh, scene);

        // FBXBone
        extractBone(mesh);

        unsigned int offsetIndices = fbxData.iIndices.value.size() - m_IndicesIncrease;
        m_VerticesSlice[meshName] = std::vector<unsigned int>
                {static_cast<unsigned int>(m_VerticesIncrease),  // Vert Start
                 m_VerticesIncrease + mesh->mNumVertices,  // Vert End
                 mesh->mNumBones,
                 m_IndicesIncrease,  // Indices Start
                 static_cast<unsigned int>(m_IndicesIncrease + offsetIndices)  // Indices End
                 };

        m_IndicesIncrease += offsetIndices;
        m_VerticesIncrease += mesh->mNumVertices;
    }

    void processNode(aiNode *node, const aiScene *scene){

        //zeno::log_info("Node: Name {}", node->mName.data);
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMesh(scene->mMeshes[node->mMeshes[i]], scene);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNode(node->mChildren[i], scene);
    }

    void processCamera(const aiScene *scene){
        // If Maya's camera does not have `LookAt`, it will use A `InterestPosition`

        zeno::log_info("Num Camera {}", scene->mNumCameras);
        for(unsigned int i=0;i<scene->mNumCameras; i++){
            auto& cam = scene->mCameras[i];
            std::string camName = cam->mName.data;
            aiMatrix4x4 camMatrix;
            cam->GetCameraMatrix(camMatrix);

#if USE_OFFICIAL_ASSIMP
            SCamera sCam{cam->mHorizontalFOV,
                         35.0f,
                         cam->mAspect,
                         1.417f * 25.4f,  // inch to mm
                         0.945f * 25.4f,
                         cam->mClipPlaneNear,
                         cam->mClipPlaneFar,
                         zeno::vec3f(0, 0, 0),
            };
#else
            SCamera sCam{cam->mHorizontalFOV,
                         cam->mFocalLength,
                         cam->mAspect,
                         cam->mFilmWidth * 25.4f,  // inch to mm
                         cam->mFilmHeight * 25.4f,
                         cam->mClipPlaneNear,
                         cam->mClipPlaneFar,
                         zeno::vec3f(cam->mInterestPosition.x, cam->mInterestPosition.y, cam->mInterestPosition.z),
                // TODO The following data that is all default, we use Cam-Anim TRS instead of them
                /*zeno::vec3f(cam->mLookAt.x, cam->mLookAt.y, cam->mLookAt.z),*/
                /*zeno::vec3f(cam->mPosition.x, cam->mPosition.y, cam->mPosition.z),*/
                /*zeno::vec3f(cam->mUp.x, cam->mUp.y, cam->mUp.z),*/
                /*camMatrix*/
            };
#endif

            zeno::log_info(">>>>> {} {} {} {} {} {} - {} {}\n {} {} {}",
                           camName, sCam.hFov, sCam.focL, sCam.aspect, sCam.pNear, sCam.pFar,
                           sCam.filmW, sCam.filmH,
                           /*sCam.lookAt[0], sCam.lookAt[1], sCam.lookAt[2],  // default is 1,0,0*/
                           /*sCam.pos[0], sCam.pos[1], sCam.pos[2], // default is 0,0,0*/
                           /*sCam.up[0], sCam.up[1], sCam.up[2],  // default is 0,1,0*/
                           sCam.interestPos[0], sCam.interestPos[1], sCam.interestPos[2]
                           );

            fbxData.iCamera.value[camName] = sCam;
        }
    }

    size_t findCaseInsensitive(std::string data, std::string toSearch, size_t pos = 0)
    {
        std::transform(data.begin(), data.end(), data.begin(), ::tolower);
        std::transform(toSearch.begin(), toSearch.end(), toSearch.begin(), ::tolower);
        return data.find(toSearch, pos);
    }

    void readMaterial(aiMesh* mesh, aiScene const* scene){
        /*  assimp - v5.0.1

            aiTextureType_NONE = 0,
            aiTextureType_DIFFUSE = 1,
            aiTextureType_SPECULAR = 2,
            aiTextureType_AMBIENT = 3,
            aiTextureType_EMISSIVE = 4,
            aiTextureType_HEIGHT = 5,
            aiTextureType_NORMALS = 6,
            aiTextureType_SHININESS = 7,
            aiTextureType_OPACITY = 8,
            aiTextureType_DISPLACEMENT = 9,
            aiTextureType_LIGHTMAP = 10,
            aiTextureType_REFLECTION = 11,
            aiTextureType_BASE_COLOR = 12,
            aiTextureType_NORMAL_CAMERA = 13,
            aiTextureType_EMISSION_COLOR = 14,
            aiTextureType_METALNESS = 15,
            aiTextureType_DIFFUSE_ROUGHNESS = 16,
            aiTextureType_AMBIENT_OCCLUSION = 17,
            aiTextureType_UNKNOWN = 18,
         */

        SMaterial mat;
        SDefaultMatProp dMatProp;
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        std::string meshName = mesh->mName.data;
        std::string matName = material->GetName().data;
        std::string vmPath = createTexDir("valueTex/" + matName);

        //zeno::log_info("1M {} M {} NT {}", meshName, matName, scene->mNumTextures);

        if( findCaseInsensitive(matName, "SKIN") != std::string::npos ){
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }else if( findCaseInsensitive(matName, "CLOTH") != std::string::npos ){
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }else if( findCaseInsensitive(matName, "HAIR") != std::string::npos ){
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }else{
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }

        for(auto&com: mat.val){

            aiTextureType texType = com.second.type;
            bool forceD = com.second.forceDefault;

            // TODO Support material multi-tex
            // The first step - to find the texture
            if(material->GetTextureCount(texType)){
                aiString str;
                material->GetTexture(texType, 0, &str);
                auto p = fbxPath;
                auto s = std::string(str.C_Str());
                auto c = (p += s).string();
                std::replace(c.begin(), c.end(), '\\', '/');
                //zeno::log_info("2N {} TN {} TT {} CP {}", com.first, str.data, texType, c);

                mat.val.at(com.first).texPath = c;
            }
            // The second step - to find the material-prop and to generate a value-based texture
            else
            {
                aiColor4D tmp;
                bool found = false;
                auto key = com.second.aiName.c_str();
                if(!forceD && com.second.aiName != ""){
                    if(AI_SUCCESS == aiGetMaterialColor(material,
                                                        key,0,0,
                                                        &tmp)){ // Found or use default value
                        found = true;
                        com.second.value = tmp;
                    }
                }

                // TODO read material-float properties
                //if(AI_SUCCESS != aiGetMaterialFloat(material, "$ai.normalCameraFactor", 0, 0, &mat.testFloat))
                //    mat.testFloat = 0.5f;
                //zeno::log_info("----- TestFloat {}", mat.testFloat);

                auto v = std::any_cast<aiColor4D>(com.second.value);
                int channel_num = 4;
                int width = 2, height = 2;
                uint8_t* pixels = new uint8_t[width * height * channel_num];

                int index = 0;
                for (int j = height - 1; j >= 0; --j)
                {
                    for (int i = 0; i < width; ++i)
                    {
                        int ir = int(255.99 * v.r);
                        int ig = int(255.99 * v.g);
                        int ib = int(255.99 * v.b);
                        int ia = int(255.99 * 1.0f);

                        pixels[index++] = ir;
                        pixels[index++] = ig;
                        pixels[index++] = ib;
                        pixels[index++] = ia;
                    }
                }

                std::string img_path = vmPath+"/"+com.first+".png";

                stbi_write_png(img_path.c_str(), width, height, channel_num, pixels, width*channel_num);

                mat.val.at(com.first).texPath = img_path;
                //zeno::log_info("3N {} P `{}` F {} V `{} {} {} {}`", com.first, key, found, v.r, v.g, v.b, v.a);
            }
        }

        mat.matName = matName;

        fbxData.iMaterial.value[meshName] = mat;
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

    void processTrans(std::unordered_map<std::string, std::vector<SKeyMorph>>& morph,
                      std::unordered_map<std::string, SAnimBone>& bones,
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
            sub_data->iMeshName.value = meshName;
            // TODO Currently it is the sub-data that has all the data
            sub_data->iBoneOffset = fbxData.iBoneOffset;
            sub_data->iBlendSData = fbxData.iBlendSData;
            sub_data->iKeyMorph.value = morph;

            prims->lut[meshName] = sub_prim;
            datas->lut[meshName] = sub_data;
        }
    }

    void processPrim(std::shared_ptr<zeno::PrimitiveObject>& prim){
        auto &ver = prim->verts;
        auto &ind = prim->tris;
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &clr0 = prim->verts.add_attr<zeno::vec3f>("clr0");

        for(unsigned int i=0; i<fbxData.iVertices.value.size(); i++){
            auto& vpos = fbxData.iVertices.value[i].position;
            auto& vnor = fbxData.iVertices.value[i].normal;
            auto& vuv = fbxData.iVertices.value[i].texCoord;
            auto& vc = fbxData.iVertices.value[i].vectexColor;

            ver.emplace_back(vpos.x, vpos.y, vpos.z);
            uv.emplace_back(vuv.x, vuv.y, vuv.z);
            norm.emplace_back(vnor.x, vnor.y, vnor.z);
            clr0.emplace_back(vc.r, vc.g, vc.b);
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

    std::unordered_map<std::string, std::vector<SKeyMorph>> m_Morph;  // Value: NumKeys

    float duration;
    float tick;
    void initAnim(aiScene const*scene, Mesh* model){

        readHierarchyData(m_RootNode, scene->mRootNode);
        //zeno::log_info("----- Anim: Convert AssimpNode.");

        //Helper::printNodeTree(&m_RootNode, 0);

        if(scene->mNumAnimations){
            // TODO handle more animation if have
            for(unsigned int i=0; i<scene->mNumAnimations; i++){
                auto animation = scene->mAnimations[i];
                duration = animation->mDuration;
                tick = animation->mTicksPerSecond;
                zeno::log_info("AniName: {} NC {} NMC {} NMMC {} D {} T {}",
                               animation->mName.data,
                               animation->mNumChannels,
                               animation->mNumMeshChannels,
                               animation->mNumMorphMeshChannels,
                               animation->mDuration,
                               animation->mTicksPerSecond
                               );

                setupBones(animation);
                setupBlendShape(animation);
            }
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

    void setupBones(const aiAnimation *animation) {
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

        //zeno::log_info("SetupBones: Num AnimBoneMap: {}", m_Bones.AnimBoneMap.size());
    }

    void setupBlendShape(const aiAnimation *animation){
        auto NumMorphChannel = animation->mNumMorphMeshChannels;
        if(NumMorphChannel){
            for(int j=0; j<NumMorphChannel; j++){
                aiMeshMorphAnim* channel = animation->mMorphMeshChannels[j];
                std::string channelName(channel->mName.data);  // pPlane1*0 with *0
                channelName = channelName.substr(0, channelName.find_last_of('*'));
                zeno::log_info("BlendShape Channel Name {}", channelName);

                if(channel->mNumKeys) {
                    std::vector<SKeyMorph> keyMorph;

                    zeno::log_info("BlendShape NumKeys {} NumVAW {}", channel->mNumKeys,
                                   channel->mKeys[0].mNumValuesAndWeights);

                    for (int i = 0; i < channel->mNumKeys; ++i) {
                        SKeyMorph morph{};
                        auto &key = channel->mKeys[i];
                        morph.m_NumValuesAndWeights = key.mNumValuesAndWeights; // e.g. pPlane2 pPlane3...
                        morph.m_Time = key.mTime;  // Frame
                        morph.m_Weights = new double[key.mNumValuesAndWeights];
                        morph.m_Values = new unsigned int[key.mNumValuesAndWeights]; // Which one is that
                        // Deep copy
                        std::copy(key.mWeights, key.mWeights + key.mNumValuesAndWeights, morph.m_Weights);
                        std::copy(key.mValues, key.mValues + key.mNumValuesAndWeights, morph.m_Values);
                        keyMorph.push_back(morph);
                    }

                    m_Morph[channelName] = keyMorph;
                }
            }
        }
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
                                             //| aiProcess_FlipUVs
                                             | aiProcess_CalcTangentSpace
                                             | aiProcess_JoinIdenticalVertices
                                             );
    if(! scene)
        zeno::log_error("ReadFBXPrim: Invalid assimp scene");

    Mesh mesh;
    Anim anim;

    std::filesystem::path p(fbx_path);
    mesh.fbxPath = p.remove_filename();
    //zeno::log_info("ReadFBXPrim: FBXPath {}", mesh.fbxPath.string());

    mesh.initMesh(scene);
    anim.initAnim(scene, &mesh);
    mesh.processTrans(anim.m_Morph, anim.m_Bones.AnimBoneMap, prims, datas);
    mesh.processPrim(prim);

    mesh.fbxData.iKeyMorph.value = anim.m_Morph;

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

        //zeno::log_info("ReadFBXPrim: File Path {}", path);

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
