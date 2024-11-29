#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "assimp/scene.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/cimport.h"
#include "assimp/LogStream.hpp"
#include "assimp/DefaultLogger.hpp"
#include "assimp/importerdesc.h"
#include "assimp/GenericProperty.h"
#include "assimp/Exceptional.h"
#include "assimp/BaseImporter.h"

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>

#include <stack>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <fstream>

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include "Definition.h"

namespace {

using Path = std::filesystem::path;

struct Mesh{
    FBXData fbxData;
    std::filesystem::path fbxPath;
    std::unordered_map<std::string, std::vector<unsigned int>> m_VerticesSlice;
    std::unordered_map<std::string, std::string> m_MeshNamePath;
    std::unordered_map<std::string, std::string> m_OrigMeshNamePath;
    std::unordered_map<std::string, std::vector<std::string>> m_MatMeshNames;
    std::unordered_map<std::string, std::string> m_MeshCorsName;
    std::unordered_map<std::string, std::string> m_LoadedMeshName;
    std::unordered_map<std::string, SMaterial> m_loadedMat;
    std::unordered_map<std::string, float> m_MatUdimSize;
    std::unordered_map<std::string, std::vector<std::string>> m_BSName;
    unsigned int m_VerticesIncrease = 0;
    unsigned int m_IndicesIncrease = 0;
    unsigned int m_PolysIncrease = 0;
    unsigned int m_MeshNameIncrease = 0;
    unsigned int m_MeshVertSize = 0;
    SFBXReadOption m_readOption;

    std::string createTexDir(std::string subPath){
        auto p = fbxPath;
        p += subPath;

        if(! std::filesystem::exists(p)){
            std::filesystem::create_directory(p);
        }

        return p.string();
    }

    void preInitMesh(const aiScene *scene){
        processNodePre(scene->mRootNode, scene);
        fbxData.iVertices.value.reserve(m_MeshVertSize);
        fbxData.iVertices.value.resize(m_MeshVertSize);
        std::cout << "total vert size: " << m_MeshVertSize << "\n";
    }

    void initMesh(const aiScene *scene){
        m_VerticesIncrease = 0;
        fbxData.iMeshName.value = "__root__";
        fbxData.iMeshName.value_relName = "__root__";
        fbxData.iPathName.value = "/__path__";
        fbxData.iPathName.value_oriPath = "/__path__";

        TIMER_START(InitMesh_ReadTrans)
        readTrans(scene->mRootNode, aiMatrix4x4(), "");
        TIMER_END(InitMesh_ReadTrans)

        TIMER_START(InitMesh_HandleNode)
        processNode(scene->mRootNode, scene, "");
        TIMER_END(InitMesh_HandleNode)

        // TODO read Animation of Camera property and Light property,
        // e.g. FocalLength LightIntensity
        processCamera(scene);
        readLights(scene);
        fbxData.iMeshInfo.value_corsName = m_MeshCorsName;
    }

    void readLights(const aiScene *scene){
        zeno::log_info("FBX: Num Light {}", scene->mNumLights);

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

            SLight sLig{
                lightName, l->mType, l->mPosition, l->mDirection, l->mUp,
                l->mAttenuationConstant, l->mAttenuationLinear, l->mAttenuationQuadratic,
                l->mColorDiffuse, l->mColorSpecular, l->mColorAmbient,
                l->mAngleInnerCone, l->mAngleOuterCone, l->mSize
            };

            fbxData.iLight.value[lightName] = sLig;
        }
    }

    void readTrans(const aiNode * parentNode, aiMatrix4x4 parentTransform, std::string parent_path){
        unsigned int childrenCount = parentNode->mNumChildren;
        aiMatrix4x4 transformation = parentNode->mTransformation;

        auto pathName = parent_path + "/" + std::string(parentNode->mName.C_Str());

        std::string name(parentNode->mName.data) ;

        transformation = parentTransform * transformation;
        fbxData.iPathTrans.value[pathName] = transformation;
        //std::cout << "FBX: Node " << name << "\n";

        for (int i = 0; i < childrenCount; i++) {
            readTrans(parentNode->mChildren[i], transformation, pathName);
        }
    }

    std::string getFixedMeshName(aiMesh *mesh){
        std::string meshName(mesh->mName.data);
        // Deal with face-mat
        if(m_LoadedMeshName.find(meshName) != m_LoadedMeshName.end()){
            std::string newMeshName = meshName + "_" +std::to_string(m_MeshNameIncrease);
            m_MeshCorsName[newMeshName] = meshName;
            ED_COUT << "FBX: fact-mat Name: " << meshName << " -> " << newMeshName << "\n";
            meshName = newMeshName;

            m_MeshNameIncrease++;
        }else{
            m_MeshCorsName[meshName] = meshName;
            m_LoadedMeshName[meshName] = "Hello!";
        }

        return meshName;
    }

    std::string reconstructionPathName(std::string path){
        std::vector<std::string> parts;
        std::istringstream iss(path);
        std::string token;
        while (std::getline(iss, token, '/')) {
            if (!token.empty()) {
                parts.push_back(token);
            }
        }

        std::vector<std::string> results;
        for (const auto& part : parts) {
            size_t subStrEndPos = part.find("$AssimpFbx$");
            if(subStrEndPos == std::string::npos){
                results.emplace_back(part);
            }
        }

        std::string repath = "/";
        for (const auto& result : results) {
            repath = repath + result + "/";
        }
        // Remove the last '/'
        repath.pop_back();

        return repath;
    }

    void processMesh(aiMesh *mesh, const aiScene *scene, std::string pathName) {
        std::string meshName = getFixedMeshName(mesh);
        auto numAnimMesh = mesh->mNumAnimMeshes;
        float uv_scale = 1.0f;
        auto origPathName = pathName;
        pathName = reconstructionPathName(pathName);

        //ED_COUT << "FBX: ReadMesh - Mesh name " << meshName << " VertCount " << mesh->mNumVertices << " NumAnimMesh " << numAnimMesh << " NumBone " << mesh->mNumBones << "\n";
        //ED_COUT << "FBX: ReadMesh - Path name " << pathName << "\n";

        // Material
//        TIMER_START(InitMesh_HandleNode_ReadMat)
        if(mesh->mNumVertices)
            readMaterial(mesh, meshName, scene, &uv_scale);
//        TIMER_END(InitMesh_HandleNode_ReadMat)

        // Vertices & BlendShape
//        TIMER_START(InitMesh_HandleNode_ReadVertices)

        for(unsigned int j = 0; j < mesh->mNumVertices; ++j){
            SVertex vertexInfo;

            aiVector3D vec(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            //zeno::log_info("vec: {} {} {}", vec.x, vec.y, vec.z);
            vertexInfo.position = vec;

            if (mesh->mTextureCoords[0]){
                //aiVector3D uvw(fmodf(mesh->mTextureCoords[0][j].x, 1.0f),
                //               fmodf(mesh->mTextureCoords[0][j].y, 1.0f), 0.0f);
                aiVector3D uvw(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y, 0.0f);
                // Same vert but diff uv
                // U 0.980281 0.0276042 V -0.5 -1 -0.866026
                // U 0.0325739 0.0276042 V -0.5 -1 -0.866026
                //zeno::log_info(">>>>> U {} {} V {} {} {}", uvw.x, uvw.y, vec.x, vec.y, vec.z);

                if(uv_scale > 1.0f){
                    uvw /= uv_scale;
                }
                vertexInfo.texCoord = uvw;
            }
            if (mesh->mNormals) {
                aiVector3D normal(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);
                vertexInfo.normal = normal;
            }

            //if(mesh->mTangents){
            //    aiVector3D tangent(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z);
            //    vertexInfo.tangent = tangent;
            //}
            //if(mesh->mBitangents){
            //    aiVector3D bitangent(mesh->mBitangents[j].x, mesh->mBitangents[j].y, mesh->mBitangents[j].z);
            //    vertexInfo.bitangent = bitangent;
            //}

            // TODO Support more color channel
            if(mesh->HasVertexColors(0)){
                aiColor4D cls(mesh->mColors[0][j].r, mesh->mColors[0][j].g,
                              mesh->mColors[0][j].b, mesh->mColors[0][j].a);
                vertexInfo.vectexColor = cls;
            }

//            vertexInfo.extraInfos["name"] = meshName;
//            vertexInfo.extraInfos["path"] = origPathName;

            fbxData.iVertices.value[j + m_VerticesIncrease] = vertexInfo;
        }
//        TIMER_END_MSG(InitMesh_HandleNode_ReadVertices, meshName)

        // BlendShape
//        TIMER_START(InitMesh_HandleNode_ReadBlendShape)
        if(numAnimMesh){
            // The number of vertices should be the same no matter how many anim-meshes there are
            // So let's take the first one
            unsigned int bsNumVert = mesh->mAnimMeshes[0]->mNumVertices;

            std::cout << "FBX: BS MeshName "<<meshName<<" NumAnim " <<numAnimMesh << "\n";

            std::vector<std::vector<SBSVertex>> blendShapeData;
            blendShapeData.resize(numAnimMesh);

            for(unsigned int a=0; a<numAnimMesh; a++){
                auto& animMesh = mesh->mAnimMeshes[a];
                auto animMeshName = std::string(animMesh->mName.C_Str());
                unsigned int aNumV = animMesh->mNumVertices;
                std::cout << "FBX: BSName " << animMeshName << "\n";
                blendShapeData[a].resize(aNumV);

                for(unsigned int i=0; i<aNumV; i++){
                    SBSVertex sbsVertex;

                    // assimp ignore the transformation of this mesh. we need apply those transformation in eval node.
                    sbsVertex.position = animMesh->mVertices[i];
                    sbsVertex.normal = animMesh->mNormals[i];
                    sbsVertex.deltaPosition = sbsVertex.position
                            - fbxData.iVertices.value[i+m_VerticesIncrease].position;
                    sbsVertex.deltaNormal = sbsVertex.normal
                            - fbxData.iVertices.value[i+m_VerticesIncrease].normal;
                    //sbsVertex.weight = animMesh->mWeight;
                    //std::cout << " i " << i << " " <<animMesh->mWeight<<" - ("
                    //          <<sbsVertex.deltaPosition.x<<","<<sbsVertex.deltaPosition.y<<","<<sbsVertex.deltaPosition.z<<")"
                    //          << "\n";
                    blendShapeData[a][i] = sbsVertex;
                }

                m_BSName[meshName].emplace_back(animMesh->mName.C_Str());
            }
            fbxData.iBlendSData.value[meshName] = blendShapeData;
        }
//        TIMER_END(InitMesh_HandleNode_ReadBlendShape)

        // Indices
        int start = 0;
        int polysCount = 0;
//        TIMER_START(InitMesh_HandleNode_ReadIndices)
        for(unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            //zeno::log_info("-----");
            for(unsigned int j = 0; j < face.mNumIndices; j++) {
                if(m_readOption.triangulate) {
                    fbxData.iIndices.valueTri.push_back(face.mIndices[j] + m_VerticesIncrease);
                }else{
                    fbxData.iIndices.valueLoops.push_back(face.mIndices[j] + m_VerticesIncrease);
                }
                //zeno::log_info(" {}", face.mIndices[j] + m_VerticesIncrease);
            }
            if(! m_readOption.triangulate) {
                fbxData.iIndices.valuePolys.emplace_back(start, face.mNumIndices);
            }
            start += face.mNumIndices;
            polysCount++;
        }
//        TIMER_END(InitMesh_HandleNode_ReadIndices)

        // FBXBone
//        TIMER_START(InitMesh_HandleNode_ReadBone)
        extractBone(mesh);
//        TIMER_END(InitMesh_HandleNode_ReadBone)


        int sizeOff = 0;
        if(m_readOption.triangulate){
            sizeOff = fbxData.iIndices.valueTri.size();
        }else{
            sizeOff = fbxData.iIndices.valueLoops.size();
        }
        unsigned int offsetIndices = sizeOff - m_IndicesIncrease;
        m_VerticesSlice[meshName] = std::vector<unsigned int>
                {static_cast<unsigned int>(m_VerticesIncrease),  // Vert Start
                 m_VerticesIncrease + mesh->mNumVertices,  // Vert End
                 mesh->mNumBones,
                 m_IndicesIncrease,  // Indices Start
                 static_cast<unsigned int>(m_IndicesIncrease + offsetIndices),  // Indices End
                 m_PolysIncrease,  // Mesh Polys Count Start
                 m_PolysIncrease + polysCount  // Mesh Polys Count End
                 };
        m_MeshNamePath[meshName] = pathName;
        m_OrigMeshNamePath[meshName] = origPathName;

        m_IndicesIncrease += offsetIndices;
        m_PolysIncrease += polysCount;
        m_VerticesIncrease += mesh->mNumVertices;
    }

    void processNode(aiNode *node, const aiScene *scene, std::string parent_path){
        auto path = parent_path + "/" + std::string(node->mName.C_Str());
        auto transform = node->mTransformation;

        //zeno::log_info("Node: Name {}", node->mName.data);
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMesh(scene->mMeshes[node->mMeshes[i]], scene, path);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNode(node->mChildren[i], scene, path);
    }

    void processNodePre(aiNode *node, const aiScene *scene){
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMeshPre(scene->mMeshes[node->mMeshes[i]], scene);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNodePre(node->mChildren[i], scene);
    }

    void processNodeMat(aiNode *node, const aiScene *scene){
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMeshMat(scene->mMeshes[node->mMeshes[i]], scene);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNodeMat(node->mChildren[i], scene);
    }

    void processMeshMat(aiMesh *mesh, const aiScene *scene){
        if(mesh->mNumVertices) {
            float uv_scale = 1.0f;
            std::string meshName = getFixedMeshName(mesh);
            readMaterial(mesh, meshName, scene, &uv_scale);
        }
    }

    void processMeshPre(aiMesh *mesh, const aiScene *scene){
        m_MeshVertSize += mesh->mNumVertices;
    }

    void processCamera(const aiScene *scene){
        // If Maya's camera does not have `LookAt`, it will use A `InterestPosition`

        std::cout << "FBX: Num Camera " << scene->mNumCameras << "\n";
        for(unsigned int i=0;i<scene->mNumCameras; i++){
            aiCamera* cam = scene->mCameras[i];
            std::string camName = cam->mName.data;
            aiMatrix4x4 camMatrix;
            cam->GetCameraMatrix(camMatrix);

            SCamera sCam;
            sCam.focL = cam->mFocalLength;
            sCam.filmW = cam->mFilmWidth * 25.4f; // inch to mm
            sCam.filmH = cam->mFilmHeight * 25.4f;
            std::cout << "Camera Name: " << camName << "\n";
            fbxData.iCamera.value[camName] = sCam;

        }
    }

    size_t findCaseInsensitive(std::string data, std::string toSearch, size_t pos = 0)
    {
        std::transform(data.begin(), data.end(), data.begin(), ::tolower);
        std::transform(toSearch.begin(), toSearch.end(), toSearch.begin(), ::tolower);
        return data.find(toSearch, pos);
    }
    bool setMergedImageData(uint8_t *pixels, int channel_num, uint8_t *data, int row, int col, int width, int pixel_len, int size){
        for (int j = (row * width-1), x = width-1; j >= (row-1) * width; j--, x--) {

            int ls = j * size * width * channel_num;
            int ls_ = x * width * channel_num;

            for (int k = (col-1)*width, y = 0; k < col*width; k++, y++) {
                int ir,ig,ib,ia;

                ir = data[ls_+y*channel_num+0];
                ig = data[ls_+y*channel_num+1];
                ib = data[ls_+y*channel_num+2];
                ia = data[ls_+y*channel_num+3];

                if(ls+k*channel_num < pixel_len){
                    pixels[ls+k*channel_num+0] = ir;
                    pixels[ls+k*channel_num+1] = ig;
                    pixels[ls+k*channel_num+2] = ib;
                    pixels[ls+k*channel_num+3] = ia;
                }else{
                    printf("ERROR Writing %d %d\n", ls+k*channel_num, pixel_len);
                    return false;
                }
            }
        }
        return true;
    }
    void StringSplitReverse(std::string str, const char split, std::vector<std::string> & ostrs)
    {
        std::istringstream iss(str);
        std::string token;
        std::vector<std::string> res(0);
        while(getline(iss, token, split))
        {
            res.push_back(token);
        }
        ostrs.resize(0);
        for(int i=res.size()-1; i>=0;i--)
        {
            ostrs.push_back(res[i]);
        }
    }
    void formPath(std::vector<std::string> &tokens)
    {
        for(int i=1; i<tokens.size();i++)
        {
            tokens[i] = tokens[i] + '/' + tokens[i-1];
        }
    }
    bool findFile(std::string HintPath, std::string origPath, std::string & oPath)
    {
        std::vector<std::string> paths;
        StringSplitReverse(origPath, '/', paths);
        formPath(paths);
        for(int i=0; i<paths.size(); i++)
        {
            auto filename = HintPath + '/' + paths[i];
            if(std::filesystem::exists(filename.c_str()))
            {
                oPath = filename;
                return true;
            }
        }
        return false;

    }
    void readMaterial(aiMesh* mesh, std::string relMeshName, aiScene const* scene, float *uvscale){
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

        SDefaultMatProp dMatProp;
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        std::string matName = material->GetName().data;
        //std::string meshName = mesh->mName.data;

        m_MatMeshNames[matName].push_back(relMeshName);

        if(m_loadedMat.find(matName) != m_loadedMat.end()){
            fbxData.iMaterial.value[relMeshName] = m_loadedMat[matName];
            *uvscale = m_MatUdimSize[matName];
            return;
        }

        // Use the metallic attribute to determine if the material is PBR
        //bool isPbr = false;

        SMaterial mat;
        mat.matName = matName;

        std::cout << "FBX: ReadMat - Mesh name "<<relMeshName<<" Mat name "<<matName<<"\n";

        mat.setDefaultValue(dMatProp.getUnknownProp());

        for(auto&com: mat.val)
        {
            bool texfound=false;
            bool forceD = com.second.forceDefault;

            for(int i=0;i<com.second.types.size(); i++){
                aiTextureType texType = com.second.types[i];

                // TODO Support material multi-tex
                // The first step - to find the texture
                if(material->GetTextureCount(texType)){
                    aiString texPathGet;
                    material->GetTexture(texType, 0, &texPathGet);

                    aiUVTransform uvTransform;
                    material->Get(AI_MATKEY_UVTRANSFORM(texType, 0), uvTransform);

                    auto fbxFilePath = fbxPath;
                    auto texPathStr = std::string(texPathGet.C_Str());
                    auto combinePath = (fbxFilePath += texPathStr).string();
                    std::replace(combinePath.begin(), combinePath.end(), '\\', '/');

                    auto hintPath = m_readOption.hintPath;
                    if(hintPath != "-1") {
                        std::string truePath;
                        findFile(hintPath, combinePath, truePath);

                        //std::cout << "hintPath:" << hintPath << "\n";
                        //std::cout << "truePath:" << truePath << "\n";
                        //std::cout << "fbxPath:" << fbxPath << "\n";
                        //std::cout << "combinePath:" << combinePath << "\n";

                        combinePath = truePath;
                    }

                    std::cout << " PropName "<<com.first<<" RelTexPath "<<texPathGet.data<<"\n";
                    std::cout << "  TexType "<<texType<<" MerPath "<<combinePath<<"\n";
                    std::cout << " uv transform " << uvTransform.mScaling.x << " " << uvTransform.mScaling.y
                              << " - " << uvTransform.mTranslation.x << " " << uvTransform.mTranslation.y << "\n";


                    Path file_full_path(combinePath);
                    std::string filename = Path(combinePath).filename().string();
                    Path file_path = Path(combinePath).remove_filename();

                    // Check if it is UDIM
                    int pos = 0;
                    int index = 0;
                    bool is_udim_tex = false;
                    int udim_num = 0;
                    while((index = filename.find(".10", pos)) != std::string::npos) {  // UDIM e.g. Tex_1001.png Tex_1011.png
                        if(filename.find_last_of('.') == index+5){  // index `_` pos, index+5 `.` pos
                            is_udim_tex = true;

                            udim_num = std::stoi(filename.substr(index+3, 2));
                            zeno::log_info("UDIM: Found udim tex num {}, enable {}",udim_num, m_readOption.enableUDIM);
                            break;  // for check whether is udim
                        }
                        pos = index + 1; //new position is from next element of index
                    }

                    if(is_udim_tex && m_readOption.enableUDIM){  // first udim check
                        int max_up=0, max_vp=0;
                        int sw,sh,comp;
                        std::unordered_map<std::string, std::string> udim_texs;
                        auto merged_path = Path(file_path);
                        auto fn_replace = filename;
                        merged_path+= fn_replace.replace(index+1, 4, "MERGED_UDIM");

                        if(std::filesystem::exists(Path(combinePath))) {

                            stbi_info(Path(combinePath).string().c_str(), &sw, &sh, &comp);
                            zeno::log_info("UDIM: Info {} {} {} {}", filename, sw, sh, comp);

                            // Find all udim tex
                            for (int i = 0; i < 10; i++) {
                                for (int j = 1; j < 10; j++) {
                                    std::string udnum = std::to_string(i) + std::to_string(j);
                                    auto replaced_filename = filename.replace(index + 3, 2, udnum);
                                    auto search_file_path = Path(file_path);
                                    search_file_path += replaced_filename;

                                    bool file_exists = std::filesystem::exists(search_file_path);
                                    if (file_exists) {
                                        zeno::log_info("UDIM: Unum {}, Exists texPath {}", udnum,
                                                       search_file_path.string());
                                        max_up = std::max(max_up, j);
                                        max_vp = std::max(max_vp, i + 1);

                                        udim_texs[udnum] = search_file_path.string();
                                    }
                                }
                            }
                            zeno::log_info("UDIM: Max u {} v {}, num {}", max_up, max_vp, udim_texs.size());

                            if (udim_texs.size() != 1) {  // final udim check
                                int size = std::max(max_up, max_vp);

                                if(std::filesystem::exists(merged_path)){
                                    combinePath = merged_path.string();
                                    *uvscale = float(size);
                                    m_MatUdimSize[matName] = float(size);

                                }else{

                                    int channel_num = 4;
                                    bool success = true;
                                    int fsize = sw * size;
                                    int pixel_len = fsize * fsize * channel_num;

                                    uint8_t *pixels = new uint8_t[pixel_len];

                                    for (auto &ut : udim_texs) {
                                        int width, height, n;
                                        std::string native_path = std::filesystem::u8path(ut.second).string();
                                        uint8_t *data = stbi_load(native_path.c_str(), &width, &height, &n, 4);
                                        zeno::log_info("UDIM: Read Tex {}, {} {} {}", Path(ut.second).filename().string(),
                                                       width, height, n);
                                        if (width != height) {
                                            success = false;
                                            break;
                                        }

                                        int unum = std::stoi(ut.first);
                                        int row, col;
                                        if (unum / 10.0f < 1) { // UDIM 100X ~
                                            row = size;
                                            col = unum;
                                        } else {
                                            col = unum % 10;
                                            row = size - unum / 10;
                                        }
                                        zeno::log_info("UDIM: writting data row {} col {}", row, col);
                                        bool wr =
                                            setMergedImageData(pixels, channel_num, data, row, col, width, pixel_len, size);
                                        if (!wr) {
                                            success = false;
                                            break;
                                        }

                                        delete[] data;
                                    }

                                    if (success) {
                                        zeno::log_info("UDIM: Write merged udim tex {}", merged_path.string());
                                        stbi_write_png(merged_path.string().c_str(), fsize, fsize, channel_num, pixels,
                                                       fsize * channel_num);

                                        combinePath = merged_path.string();
                                        *uvscale = float(size);
                                        m_MatUdimSize[matName] = float(size);
                                    }
                                }
                            }
                        }
                    }

                    mat.val.at(com.first).texPath = combinePath;
                    mat.val.at(com.first).uvTransform = uvTransform;

                    texfound = true;

                    break;  // for tex-types
                }
            }

            if(! texfound){
                mat.val.at(com.first).texPath = "-1";
            }

            {
                aiColor4D tmp;

                for(int i=0;i<com.second.aiNames.size(); i++){
                    auto key = com.second.aiNames[i];

                    if(!forceD && !key.empty()){
                        if(AI_SUCCESS == aiGetMaterialColor(material, key.c_str(),0,0, &tmp)){

                            //if(key == "$ai.metalness")
                            //    isPbr = true;

                            // override the default value
                            com.second.value = tmp;
                            std::cout <<" PropName "<<com.first<<" MatValue "<<tmp.r<<","<<tmp.g<<","<<tmp.b<<","<<tmp.a<<"\n";
                            break;
                        }
                    }
                }

            }

        }

        m_loadedMat[matName] = mat;
        fbxData.iMaterial.value[relMeshName] = mat;
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
                      std::shared_ptr<zeno::DictObject>& datas,
                      std::shared_ptr<zeno::DictObject>& mats,
                      std::shared_ptr<NodeTree>& nodeTree,
                      std::shared_ptr<BoneTree>& boneTree,
                      std::shared_ptr<AnimInfo>& animInfo)
    {
        auto current = 0;
        for(auto& iter: m_VerticesSlice) {
            std::cout << "curr: " << current << " total " << m_VerticesSlice.size() << "\n";
            std::string meshName = iter.first;
            std::string pathName = m_MeshNamePath[iter.first];
            std::string origPathName = m_OrigMeshNamePath[iter.first];
            std::string relMeshName = m_MeshCorsName[iter.first];

            std::vector<unsigned int> verSlice = iter.second;
            unsigned int verStart = verSlice[0];
            unsigned int verEnd = verSlice[1];
            unsigned int verBoneNum = verSlice[2];
            unsigned int indicesStart = verSlice[3];
            unsigned int indicesEnd = verSlice[4];
            unsigned int polysCountStart = verSlice[5];
            unsigned int polysCountEnd = verSlice[6];

            // TODO full support blend bone-animation and mesh-animation, See SimTrans.fbx
            bool foundMeshBone = bones.find(relMeshName) != bones.end();

            auto sub_data = std::make_shared<FBXData>();
            std::vector<SVertex> sub_vertices;
            std::vector<unsigned int> sub_TriIndices;
            std::vector<unsigned int> sub_LoopsIndices;
            std::vector<zeno::vec2i> sub_PolysIndices;

            if(m_readOption.triangulate) {
                for (unsigned int i = indicesStart; i < indicesEnd; i += 3) {
                    auto i1 = fbxData.iIndices.valueTri[i] - verStart;
                    auto i2 = fbxData.iIndices.valueTri[i + 1] - verStart;
                    auto i3 = fbxData.iIndices.valueTri[i + 2] - verStart;
                    zeno::vec3i incs(i1, i2, i3);
                    sub_TriIndices.push_back(i1);
                    sub_TriIndices.push_back(i2);
                    sub_TriIndices.push_back(i3);
                }
            }else{
                for (unsigned int i = polysCountStart; i < polysCountEnd; ++i) {
                    sub_PolysIndices.emplace_back(fbxData.iIndices.valuePolys[i]);
                }
                for (unsigned int i = indicesStart; i < indicesEnd; ++i) {
                    auto l1 = fbxData.iIndices.valueLoops[i] - verStart;
                    sub_LoopsIndices.push_back(l1);
                }

            }

            for(unsigned int i=verStart; i< verEnd; ++i){
                sub_vertices.push_back(fbxData.iVertices.value[i]);
            }

            IPathTrans piecePathTrans{};
            piecePathTrans.value[origPathName] = fbxData.iPathTrans.value[origPathName];

            sub_data->iIndices.valueTri = sub_TriIndices;
            sub_data->iIndices.valueLoops = sub_LoopsIndices;
            sub_data->iIndices.valuePolys = sub_PolysIndices;
            sub_data->iVertices.value = sub_vertices;
            sub_data->iPathName.value = pathName;
            sub_data->iPathName.value_oriPath = origPathName;
            sub_data->iMeshName.value = meshName;
            sub_data->iMeshName.value_relName = relMeshName;
            sub_data->iMeshName.value_matName = fbxData.iMaterial.value[meshName].matName;
            // TODO Currently it is the sub-data that has all the data
            sub_data->iBoneOffset = fbxData.iBoneOffset;
            sub_data->iBlendSData = fbxData.iBlendSData;
            sub_data->iPathTrans = piecePathTrans;
            sub_data->iKeyMorph.value = morph;
            sub_data->iMeshInfo.value_corsName = m_MeshCorsName;

            sub_data->boneTree = boneTree;
            sub_data->nodeTree = nodeTree;
            sub_data->animInfo = animInfo;

            datas->lut[meshName] = sub_data;

            ++current;
        }

        for(auto [k, v]:m_loadedMat){
            auto mat_data = std::make_shared<MatData>();

            mat_data->sMaterial = v;
            for(auto l: m_MatMeshNames[k]){
                auto fbx_data = zeno::safe_dynamic_cast<FBXData>(datas->lut[l]);
                mat_data->iFbxData.value[l] = fbx_data;
            }
            mats->lut[k] = mat_data;
        }

    }

    void processPrim(std::shared_ptr<zeno::PrimitiveObject>& prim){
        auto &ver = prim->verts;
        auto &ind = prim->tris;
        auto &polys = prim->polys;
        auto &loops = prim->loops;
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

        if(m_readOption.triangulate) {
            for (unsigned int i = 0; i < fbxData.iIndices.valueTri.size(); i += 3) {
                zeno::vec3i incs(fbxData.iIndices.valueTri[i], fbxData.iIndices.valueTri[i + 1],
                                 fbxData.iIndices.valueTri[i + 2]);
                ind.push_back(incs);
            }
        }else{
            for (unsigned int i = 0; i < fbxData.iIndices.valueLoops.size(); i ++) {
                loops.emplace_back(fbxData.iIndices.valueLoops[i]);
            }
            for(int i=0; i<fbxData.iIndices.valuePolys.size(); i++){
                polys.emplace_back(fbxData.iIndices.valuePolys[i]);
            }
        }
    }
};

struct Anim{
    NodeTree m_RootNode;
    BoneTree m_Bones;
    SFBXReadOption m_readOption;

    std::unordered_map<std::string, std::vector<SKeyMorph>> m_Morph;  // Value: NumKeys
    std::unordered_map<std::string, std::vector<std::string>> m_MeshBSName;

    AnimInfo m_animInfo;

    void initAnim(aiScene const*scene, Mesh* model){
        zeno::log_info("FBX: Init Animation");

        m_animInfo.duration = 0.0f;
        m_animInfo.tick = 0.0f;
        m_animInfo.maxTimeStamp = std::numeric_limits<float>::min();
        m_animInfo.minTimeStamp = std::numeric_limits<float>::max();

        readHierarchyData(m_RootNode, scene->mRootNode);

        if(m_readOption.printTree)
            Helper::printNodeTree(&m_RootNode, 0);

        if(scene->mNumAnimations){
            // TODO Handle more animation if have
            // TODO Blend Shape Initial Key data. e.g. 0.6, not animated.
            for(unsigned int i=0; i<scene->mNumAnimations; i++){
                auto animation = scene->mAnimations[i];
                m_animInfo.duration = animation->mDuration;
                m_animInfo.tick = animation->mTicksPerSecond;

                std::cout << "FBX Anim: Name " << animation->mName.data << " NumChannel " << animation->mNumChannels << "\n";
                std::cout << "FBX Anim: NumMeshChannel " << animation->mNumMeshChannels << " NumMorphMeshChannel " << animation->mNumMorphMeshChannels << "\n";
                std::cout << "FBX Anim: Duration " << animation->mDuration << " TicksPerSecond " << animation->mTicksPerSecond << "\n";

                setupBones(animation);
                setupBlendShape(animation);

                break;
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
        double tickOffset = m_readOption.offsetInSeconds * animation->mTicksPerSecond;

        for (int i = 0; i < size; i++) {
            auto channel = animation->mChannels[i];
            std::string boneName(channel->mNodeName.data);
            //zeno::log_info("----- Name {}", boneName);
            ED_COUT << " Anim Bone Name: " << boneName << " tickOffset " << tickOffset << "\n";

            SAnimBone bone;
            bone.initBone(boneName, channel, tickOffset);

            m_Bones.AnimBoneMap[boneName] = bone;

            m_animInfo.maxTimeStamp = std::max(m_animInfo.maxTimeStamp, bone.m_MaxTimeStamp);
            m_animInfo.minTimeStamp = std::min(m_animInfo.minTimeStamp, bone.m_MinTimeStamp);
        }
        // XXX -5 ~ 18  ---> 5 -> 28
        //zeno::log_info("SetupBones: Num AnimBoneMap: {}", m_Bones.AnimBoneMap.size());
        std::cout << "FBX: MinTimeStamp " << m_animInfo.minTimeStamp << " MaxTimeStamp " << m_animInfo.maxTimeStamp << std::endl;
    }

    void setupBlendShape(const aiAnimation *animation){
        double tickOffset = m_readOption.offsetInSeconds * animation->mTicksPerSecond;

        auto NumMorphChannel = animation->mNumMorphMeshChannels;
        std::cout << "FBX: BlendShape NumMorphChannel " << NumMorphChannel << "\n";
        if(NumMorphChannel){
            for(int j=0; j<NumMorphChannel; j++){
                aiMeshMorphAnim* channel = animation->mMorphMeshChannels[j];
                std::string channelName(channel->mName.data);  // pPlane1*0 with *0
                channelName = channelName.substr(0, channelName.find_last_of('*'));
                std::cout << "FBX: BS Channel Name " << channelName << "\n";
                std::vector<SKeyMorph> keyMorph;

                if(channel->mNumKeys) {
                    std::cout << "FBX: BS NumKeys " << channel->mNumKeys << " NumVal&Wei " << channel->mKeys[0].mNumValuesAndWeights << "\n";

                    for (int i = 0; i < channel->mNumKeys; ++i) {
                        SKeyMorph morph{};
                        auto &key = channel->mKeys[i];
                        morph.m_NumValuesAndWeights = key.mNumValuesAndWeights; // e.g. blendShape pPlane1 -> pPlane2 pPlane3...
                        morph.m_Time = key.mTime + tickOffset;  // Frame
                        morph.m_Weights = new double[key.mNumValuesAndWeights];
                        morph.m_Values = new unsigned int[key.mNumValuesAndWeights]; // Which one is that
                        // Deep copy
                        std::copy(key.mWeights, key.mWeights + key.mNumValuesAndWeights, morph.m_Weights);
                        std::copy(key.mValues, key.mValues + key.mNumValuesAndWeights, morph.m_Values);
                        keyMorph.push_back(morph);

                        std::cout << "BlendShape idx " << i << " Time " << morph.m_Time << "\n";
                    }

                    m_Morph[channelName] = keyMorph;
                }else{
                    std::cout << "FBX: BS Channel " << channelName << "\n";
                    if(m_MeshBSName.find(channelName) != m_MeshBSName.end()){
                        auto& bsData = m_MeshBSName[channelName];

                        for(int i=0; i<bsData.size(); i++){
                            SKeyMorph morph{};
                            morph.m_Time = 0.0;
                            morph.m_Weights = new double[bsData.size()];
                            morph.m_Values = new unsigned int[bsData.size()];
                            morph.m_Weights[i] = 0.0;
                            morph.m_Values[i] = 0;
                            keyMorph.push_back(morph);
                        }
                        m_Morph[channelName] = keyMorph;
                    }else{
                        zeno::log_info("FBX: BlendShape {} Not Found", channelName);
                    }
                }
            }
        }
    }
};

void readFBXFile(
        std::shared_ptr<zeno::DictObject>& datas,
        std::shared_ptr<NodeTree>& nodeTree,
        std::shared_ptr<FBXData>& data,
        std::shared_ptr<BoneTree>& boneTree,
        std::shared_ptr<AnimInfo>& animInfo,
        const char *fbx_path,
        std::shared_ptr<zeno::PrimitiveObject>& prim,
        std::shared_ptr<zeno::DictObject>& mats,
        SFBXReadOption readOption
    )
{
    Assimp::Importer importer;
    aiScene const* scene;
    Mesh mesh;
    mesh.m_readOption = readOption;
    std::filesystem::path p(fbx_path);
    mesh.fbxPath = p.remove_filename();
    Anim anim;
    anim.m_readOption = readOption;

    if(readOption.generate){
        TIMER_START(GenerateRead)
        scene = importer.ReadFile(fbx_path, 0);
        if(scene == nullptr){
            std::cout << "Read empty fbx scene\n";
            return;
        }

        //mesh.createTexDir("valueTex");
        mesh.processNodeMat(scene->mRootNode, scene);
        for(auto const&[key, value]:mesh.m_loadedMat){
            mats->lut[key] = value.clone();
        }
        TIMER_END(GenerateRead)

        return;
    }

    TIMER_START(ImporterRead)
    importer.SetPropertyInteger(AI_CONFIG_PP_PTV_NORMALIZE, true);
    if(readOption.triangulate){
        scene = importer.ReadFile(fbx_path, aiProcess_Triangulate
                                                //| aiProcess_FlipUVs
                                                //| aiProcess_CalcTangentSpace
                                                | aiProcess_ImproveCacheLocality
                                                | aiProcess_JoinIdenticalVertices);
    }else{
        scene = importer.ReadFile(fbx_path, aiProcess_ImproveCacheLocality | aiProcess_JoinIdenticalVertices);
    }
    TIMER_END(ImporterRead)

    if(! scene)
        zeno::log_error("FBX: Invalid assimp scene");

    // When reading a particularly large model, the reading speed of the vertices is slow in space allocation,
    // so we first get the entire required space and then set the data directly by indexing it
    TIMER_START(PreInit)
    mesh.preInitMesh(scene);
    TIMER_END(PreInit)

    TIMER_START(InitMesh)
    mesh.initMesh(scene);
    TIMER_END(InitMesh)

    TIMER_START(InitAnim)
    anim.m_MeshBSName = mesh.m_BSName;
    anim.initAnim(scene, &mesh);
    TIMER_END(InitAnim)

    TIMER_START(HandleTrans)
    mesh.processTrans(anim.m_Morph, anim.m_Bones.AnimBoneMap, datas, mats, nodeTree, boneTree, animInfo);
    mesh.fbxData.iKeyMorph.value = anim.m_Morph;
    TIMER_END(HandleTrans)

    if(readOption.indepData) {
        TIMER_START(CopyData)
        *data = mesh.fbxData;
        *nodeTree = anim.m_RootNode;
        *boneTree = anim.m_Bones;
        *animInfo = anim.m_animInfo;
        TIMER_END(CopyData)

        data->animInfo = animInfo;
        data->boneTree = boneTree;
        data->nodeTree = nodeTree;
    }

    if(readOption.makePrim){
        mesh.processPrim(prim);
        if(prim->verts->empty()){
            zeno::log_error("empty prim");
            prim->verts.emplace_back(zeno::vec3f(0.0f, 0.0f, 0.0f));
            prim->verts.add_attr<zeno::vec3f>("nrm").emplace_back(0.0f, 0.0f, 0.0f);
            prim->verts.add_attr<zeno::vec3f>("uv").emplace_back(0.0f, 0.0f, 0.0f);
        }
    }

    zeno::log_info("FBX: Num Animation {}", scene->mNumAnimations);
    zeno::log_info("FBX: Total Vertices count {}", mesh.fbxData.iVertices.value.size());
    zeno::log_info("FBX: Total Indices count {}", mesh.fbxData.iIndices.valueTri.size());
    zeno::log_info("FBX: Total Loops count {}", mesh.fbxData.iIndices.valueLoops.size());
}

struct ReadFBXPrim : zeno::INode {

    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto hintPath = get_input<zeno::StringObject>("hintPath")->get();
        std::shared_ptr<zeno::DictObject> datas = std::make_shared<zeno::DictObject>();
        auto nodeTree = std::make_shared<NodeTree>();
        auto animInfo = std::make_shared<AnimInfo>();
        auto data = std::make_shared<FBXData>();
        auto boneTree = std::make_shared<BoneTree>();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::shared_ptr<zeno::DictObject> mats = std::make_shared<zeno::DictObject>();

        auto fbxFileName = Path(path)
                               .replace_extension("")
                               .filename()
                               .string();
        std::replace(fbxFileName.begin(),fbxFileName.end(), ' ', '_');
        zeno::log_info("FBX: File path {}, Replaced FBXName {}", path,fbxFileName);

        SFBXReadOption readOption;
        std::shared_ptr<zeno::DictObject> visibility;

        auto udim = get_param<std::string>("udim");
        auto primitive = get_param<bool>("primitive");
        auto generate = get_input2<bool>("generate");
        auto offsetInSeconds = get_input2<float>("offset");
        auto triangulate = get_param<bool>("triangulate");
        auto printTree = get_param<bool>("printTree");
        auto indepData = get_param<bool>("indepData");

        if(has_input("visibility")){
            visibility = get_input2<zeno::DictObject>("visibility");
        }else {
            visibility = std::make_shared<zeno::DictObject>();
        }

        readOption.offsetInSeconds = offsetInSeconds;
        if (udim == "ENABLE")
            readOption.enableUDIM = true;
        if(primitive)
            readOption.makePrim = true;
        if(generate)
            readOption.generate = true;
        if(indepData)
            readOption.indepData = true;
        if(triangulate)
            readOption.triangulate = true;
        if(printTree)
            readOption.printTree = true;
        readOption.hintPath = hintPath;

        readFBXFile(datas, nodeTree, data, boneTree, animInfo,
                    path.c_str(), prim, mats, readOption);

        if(generate){
            int count = 0;
            for (auto &[k, v]: mats->lut) {
                auto vc = zeno::safe_dynamic_cast<SMaterial>(v);
                std::cout << "setting user data: " << count  << " name " << k << "\n";
                prim->userData().set2(std::to_string(count), k);

                std::vector<std::string> texList{};
                std::map<std::string, int> texMap{};
                std::map<std::string, aiUVTransform> texUv{};
                std::map<std::string, aiColor4D> matValue{};

                vc->getSimplestTexList(texList, texMap, texUv, matValue);
                for(int i=0;i<texList.size();i++){
                    prim->userData().set2(std::to_string(count) + "_tex_" + std::to_string(i), texList[i]);
                }

                count++;
            }
            prim->userData().setLiterial("matNum", count);
            prim->userData().setLiterial("fbxName", fbxFileName);
        }

        data->iVisibility = *visibility;
        for(auto&[key, value]: datas->lut){
            auto data = zeno::safe_dynamic_cast<FBXData>(value);
            data->iVisibility = *visibility;
        }

        set_output("data", std::move(data));
        set_output("datas", std::move(datas));
        set_output("animinfo", std::move(animInfo));
        set_output("nodetree", std::move(nodeTree));
        set_output("bonetree", std::move(boneTree));
        set_output("prim", std::move(prim));
        set_output("mats", std::move(mats));
    }
};

ZENDEFNODE(ReadFBXPrim,
           {       /* inputs: */
               {
                   {"readpath", "path"},
                   {"readpath", "hintPath", "-1"},
                   {"bool", "generate", "false"},
                   {"float", "offset", "0.0"},
                   {"DictObject", "visibility", ""}
               },  /* outputs: */
               {
                    "prim",
                    "data",
                    {"dict", "datas", ""},
                    {"dict", "mats", ""},
                "animinfo", "nodetree", "bonetree",
               },  /* params: */
               {
                {"enum ENABLE DISABLE", "udim", "DISABLE"},
                {"bool", "primitive", "false"},
                {"bool", "triangulate", "true"},
                {"bool", "printTree", "false"},
                {"bool", "indepData", "false"},
               },  /* category: */
               {
                   "FBX",
               }
           });

struct ReadLightFromFile : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        zeno::log_info("Light: File path {}", path);

       auto posList = std::make_shared<zeno::ListObject>();
       auto rotList = std::make_shared<zeno::ListObject>();
       auto sclList = std::make_shared<zeno::ListObject>();
       auto colList = std::make_shared<zeno::ListObject>();
       auto intList = std::make_shared<zeno::ListObject>();
       auto expList = std::make_shared<zeno::ListObject>();

       std::ifstream infile(path);
       if (infile.is_open()) {
           std::string line;
           int num = 0;
           int dl = 7;
           while (std::getline(infile, line)) {
               // using printf() in all tests for consistency
               std::string l = line.c_str();
               printf("Light: Processing %s\n", l.c_str());
               if(num%dl==0){
                   LIGHT_STR_SPLIT_V3F
                   //printf("Light: Pos %.2f %.2f %.2f\n", tmp[0], tmp[1], tmp[2]);
                   posList->arr.push_back(no);
               }
               if(num%dl==1){
                   LIGHT_STR_SPLIT_V3F
                   //printf("Light: Rot %.2f %.2f %.2f\n", tmp[0], tmp[1], tmp[2]);
                   rotList->arr.push_back(no);
               }
               if(num%dl==2){
                   LIGHT_STR_SPLIT_V3F
                   //printf("Light: Scl %.2f %.2f %.2f\n", tmp[0], tmp[1], tmp[2]);
                   sclList->arr.push_back(no);
               }
               if(num%dl==3){
                   LIGHT_STR_SPLIT_V3F
                   //printf("Light: Col %.2f %.2f %.2f\n", tmp[0], tmp[1], tmp[2]);
                   colList->arr.push_back(no);
               }
               if(num%dl==4){
                   auto no = std::make_shared<zeno::NumericObject>();
                   float tmp = (float)atof(l.c_str());
                   no->set(tmp);
                   //printf("Light: Int %.2f\n", tmp);
                   intList->arr.push_back(no);
               }
               if(num%dl==5){
                   auto no = std::make_shared<zeno::NumericObject>();
                   float tmp = (float)atof(l.c_str());
                   no->set(tmp);
                   //printf("Light: Exp %.2f\n", tmp);
                   expList->arr.push_back(no);
               }

               num++;
           }
           infile.close();
       }

       set_output("posList", std::move(posList));
       set_output("rotList", std::move(rotList));
       set_output("sclList", std::move(sclList));
       set_output("colList", std::move(colList));
       set_output("intList", std::move(intList));
       set_output("expList", std::move(expList));
    }
};
ZENDEFNODE(ReadLightFromFile,
           {       /* inputs: */
            {
                {"readpath", "path"},
            },  /* outputs: */
            {
                "posList", "rotList", "sclList", "colList", "intList", "expList"
            },  /* params: */
            {
            },  /* category: */
            {
                "FBX",
            }
           });

struct ReadSTL : zeno::INode {
    virtual void apply() override {
        auto filepath = get_input2<std::string>("path");

        Assimp::Importer importer;

        // Import the STL file with triangulation and joining identical vertices
        const aiScene* scene = importer.ReadFile(filepath, 0);

        // Check if the import was successful
        if (!scene) {
            std::cerr << "Failed to load model: " << importer.GetErrorString() << std::endl;
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();

        // Iterate through all meshes in the scene
        for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
            aiMesh* mesh = scene->mMeshes[meshIndex];
            prim->verts.resize(mesh->mNumVertices);
            std::cout << "Mesh " << meshIndex << ": " << mesh->mNumVertices << " vertices, " << mesh->mNumFaces << " faces." << std::endl;

            // Iterate through all vertices in the mesh
            for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex) {
                aiVector3D position = mesh->mVertices[vertexIndex];
                prim->verts[vertexIndex] = {position.x, position.y, position.z};
            }

            prim->loops.reserve(mesh->mNumFaces * 3);
            prim->polys.resize(mesh->mNumFaces);

            int counter = 0;

            // Iterate through all faces in the mesh
            for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex) {
                aiFace face = mesh->mFaces[faceIndex];

                for (unsigned int i = 0; i < face.mNumIndices; ++i) {

                    prim->loops.push_back(face.mIndices[i]);
                }

                prim->polys[faceIndex] = {counter, int(face.mNumIndices)};
                counter += face.mNumIndices;
            }
        }
        set_output("prim", prim);
    }
};
ZENDEFNODE(ReadSTL, {
    {
        {"readpath", "path"},
    },
    {
        {"prim"},
    },
    {},
    {"primitive"},
});
}
