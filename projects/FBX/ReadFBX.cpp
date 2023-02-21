#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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
    std::unordered_map<std::string, std::vector<std::string>> m_MatMeshNames;
    std::unordered_map<std::string, std::string> m_MeshCorsName;
    std::unordered_map<std::string, std::string> m_LoadedMeshName;
    std::unordered_map<std::string, aiMatrix4x4> m_TransMatrix;
    std::unordered_map<std::string, SMaterial> m_loadedMat;
    std::unordered_map<std::string, float> m_MatUdimSize;
    std::unordered_map<std::string, std::vector<std::string>> m_BSName;
    unsigned int m_VerticesIncrease = 0;
    unsigned int m_IndicesIncrease = 0;
    unsigned int m_PolysIncrease = 0;
    unsigned int m_MeshNameIncrease = 0;
    SFBXReadOption m_readOption;

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
        fbxData.iMeshName.value_relName = "__root__";

        createTexDir("valueTex");
        readTrans(scene->mRootNode, aiMatrix4x4());
        processNode(scene->mRootNode, scene);
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

    std::string getFixedMeshName(aiMesh *mesh){
        std::string meshName(mesh->mName.data);
        // Deal with face-mat
        if(m_LoadedMeshName.find(meshName) != m_LoadedMeshName.end()){
            std::string newMeshName = meshName + "_" +std::to_string(m_MeshNameIncrease);
            m_MeshCorsName[newMeshName] = meshName;
            meshName = newMeshName;

            m_MeshNameIncrease++;
        }else{
            m_MeshCorsName[meshName] = meshName;
            m_LoadedMeshName[meshName] = "Hello!";
        }

        return meshName;
    }

    void processMesh(aiMesh *mesh, const aiScene *scene) {
        std::string meshName = getFixedMeshName(mesh);

        auto numAnimMesh = mesh->mNumAnimMeshes;
        float uv_scale = 1.0f;

        zeno::log_info("FBX: Mesh name {}, vert count {} NumAnimMesh {}", meshName, mesh->mNumVertices, numAnimMesh);

        // Material
        if(mesh->mNumVertices)
            readMaterial(mesh, meshName, scene, &uv_scale);

        // Vertices & BlendShape
        for(unsigned int j = 0; j < mesh->mNumVertices; j++){
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

            zeno::log_info("FBX: BS MeshName {} NumAnim {}", meshName, numAnimMesh);

            std::vector<std::vector<SBSVertex>> blendShapeData;
            blendShapeData.resize(numAnimMesh);

            for(unsigned int a=0; a<numAnimMesh; a++){
                auto& animMesh = mesh->mAnimMeshes[a];
                unsigned int aNumV = animMesh->mNumVertices;
                zeno::log_info("FBX: BSName {}", animMesh->mName.C_Str());
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

                m_BSName[meshName].push_back(std::string(animMesh->mName.C_Str()));
            }
            fbxData.iBlendSData.value[meshName] = blendShapeData;
        }

        // Indices
        int start = 0;
        int polysCount = 0;
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
            start+=face.mNumIndices;
            polysCount++;
        }

        // FBXBone
        extractBone(mesh);
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

        m_IndicesIncrease += offsetIndices;
        m_PolysIncrease += polysCount;
        m_VerticesIncrease += mesh->mNumVertices;
    }

    void processNode(aiNode *node, const aiScene *scene){

        //zeno::log_info("Node: Name {}", node->mName.data);
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
            processMesh(scene->mMeshes[node->mMeshes[i]], scene);
        for(unsigned int i = 0; i < node->mNumChildren; i++)
            processNode(node->mChildren[i], scene);
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

    void processCamera(const aiScene *scene){
        // If Maya's camera does not have `LookAt`, it will use A `InterestPosition`

        zeno::log_info("FBX: Num Camera {}", scene->mNumCameras);
        for(unsigned int i=0;i<scene->mNumCameras; i++){
            auto& cam = scene->mCameras[i];
            std::string camName = cam->mName.data;
            aiMatrix4x4 camMatrix;
            cam->GetCameraMatrix(camMatrix);

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

            zeno::log_info("{} hfov {} fl {} ap {} n {} f {}\n\tfw {} fh {} x {} y {} z {}",
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
        bool isPbr = false;

        SMaterial mat;
        mat.matName = matName;

        std::string tmpMatName = matName;
        std::replace(tmpMatName.begin(), tmpMatName.end(), ':', '_');
        std::string vmPath = createTexDir("valueTex/" + tmpMatName);

        zeno::log_info("FBX: Mesh name {} Mat name {}", relMeshName, matName);

        if( findCaseInsensitive(matName, "SKIN") != std::string::npos ){
            zeno::log_info("FBX: SKIN");
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }else if( findCaseInsensitive(matName, "CLOTH") != std::string::npos ){
            zeno::log_info("FBX: CLOTH");
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }else if( findCaseInsensitive(matName, "HAIR") != std::string::npos ){
            zeno::log_info("FBX: HAIR");
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }else{
            mat.setDefaultValue(dMatProp.getUnknownProp());
        }

        for(auto&com: mat.val)
        {
            bool texfound=false;
            bool forceD = com.second.forceDefault;

            for(int i=0;i<com.second.types.size(); i++){
                aiTextureType texType = com.second.types[i];

                // TODO Support material multi-tex
                // The first step - to find the texture
                if(material->GetTextureCount(texType)){
                    aiString str;
                    material->GetTexture(texType, 0, &str);
                    auto p = fbxPath;
                    auto s = std::string(str.C_Str());
                    auto c = (p += s).string();
                    std::replace(c.begin(), c.end(), '\\', '/');
                    zeno::log_info("FBX: Mat name {}, PropName {} RelTexPath {} TexType {} MerPath {}",matName,com.first, str.data, texType, c);

                    Path file_full_path(c);
                    std::string filename = Path(c).filename().string();
                    Path file_path = Path(c).remove_filename();

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

                        if(std::filesystem::exists(Path(c))) {

                            stbi_info(Path(c).string().c_str(), &sw, &sh, &comp);
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
                                    c = merged_path.string();
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

                                        uint8_t *data = stbi_load(ut.second.c_str(), &width, &height, &n, 4);
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

                                        c = merged_path.string();
                                        *uvscale = float(size);
                                        m_MatUdimSize[matName] = float(size);
                                    }
                                }
                            }
                        }
                    }

                    mat.val.at(com.first).texPath = c;
                    texfound = true;

                    break;  // for tex-types
                }
            }
            if(! texfound)
            // The second step - to find the material-prop and to generate a value-based texture
            {
                aiColor4D tmp;
                bool found = false;

                for(int i=0;i<com.second.aiNames.size(); i++){
                    auto key = com.second.aiNames[i];
                    //zeno::log_info("Getting {}", key);
                    if(!forceD && !key.empty()){
                        if(AI_SUCCESS == aiGetMaterialColor(material,
                                                            key.c_str(),0,0,
                                                            &tmp)){
                            found = true;
                            if(key == "$ai.metalness")
                                isPbr = true;
                            // override the default value
                            zeno::log_info("FBX: Mat name {}, PropName {}, MatValue {} {} {} {}",matName, com.first,tmp.r, tmp.g,tmp.b,tmp.a);
                            com.second.value = tmp;
                        }
                    }

                    if(found){
                        break;
                    }
                }

                if(isPbr && com.first == "opacity" && m_readOption.invertOpacity){
                    zeno::log_info("FBX: Convert PBR opacity");
                    com.second.value = std::any_cast<aiColor4D>(com.second.value)*-1.0f+1.0f;
                }

                auto v = std::any_cast<aiColor4D>(com.second.value);

                // XXX
                if(com.first == "opacity" && v.r>0.99f && v.g>0.99f && v.b>0.99f)
                    v *= 0.8;

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
                      std::shared_ptr<zeno::DictObject>& prims,
                      std::shared_ptr<zeno::DictObject>& mats,
                      std::shared_ptr<NodeTree>& nodeTree,
                      std::shared_ptr<BoneTree>& boneTree,
                      std::shared_ptr<AnimInfo>& animInfo)
    {
        std::unordered_map<std::string, std::shared_ptr<FBXData>> tmpFbxData;

        for(auto& iter: m_VerticesSlice) {
            std::string meshName = iter.first;
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
                        // XXX processing model transformation
                        vertex.position = m_TransMatrix[relMeshName] * fbxData.iVertices.value[i].position;
                    }
                }
            }

            // Sub-prims (applied node transform)
            auto sub_prim = std::make_shared<zeno::PrimitiveObject>();
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
                    sub_prim->tris.push_back(incs);
                    sub_TriIndices.push_back(i1);
                    sub_TriIndices.push_back(i2);
                    sub_TriIndices.push_back(i3);
                }
            }else{
                for (unsigned int i = polysCountStart; i < polysCountEnd; i++) {
                    sub_PolysIndices.emplace_back(fbxData.iIndices.valuePolys[i]);
                }
                for (unsigned int i = indicesStart; i < indicesEnd; i++) {
                    auto l1 = fbxData.iIndices.valueLoops[i] - verStart;
                    sub_LoopsIndices.push_back(l1);
                }

            }

            for(unsigned int i=verStart; i< verEnd; i++){
                sub_prim->verts.emplace_back(fbxData.iVertices.value[i].position.x,
                                             fbxData.iVertices.value[i].position.y,
                                             fbxData.iVertices.value[i].position.z);
                sub_vertices.push_back(fbxData.iVertices.value[i]);
            }

            sub_data->iIndices.valueTri = sub_TriIndices;
            sub_data->iIndices.valueLoops = sub_LoopsIndices;
            sub_data->iIndices.valuePolys = sub_PolysIndices;
            sub_data->iVertices.value = sub_vertices;
            sub_data->iMeshName.value = meshName;
            sub_data->iMeshName.value_relName = relMeshName;
            sub_data->iMeshName.value_matName = fbxData.iMaterial.value[meshName].matName;
            // TODO Currently it is the sub-data that has all the data
            sub_data->iBoneOffset = fbxData.iBoneOffset;
            sub_data->iBlendSData = fbxData.iBlendSData;
            sub_data->iKeyMorph.value = morph;
            sub_data->iMeshInfo.value_corsName = m_MeshCorsName;

            sub_data->boneTree = boneTree;
            sub_data->nodeTree = nodeTree;
            sub_data->animInfo = animInfo;

            prims->lut[meshName] = sub_prim;
            datas->lut[meshName] = sub_data;
            tmpFbxData[meshName] = sub_data;
        }

        for(auto [k, v]:m_loadedMat){
            auto mat_data = std::make_shared<MatData>();

            mat_data->sMaterial = v;
            for(auto l: m_MatMeshNames[k]){
                mat_data->iFbxData.value[l] = tmpFbxData[l];
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

        m_animInfo.duration = 0.0f;
        m_animInfo.tick = 0.0f;
        m_animInfo.maxTimeStamp = std::numeric_limits<float>::min();
        m_animInfo.minTimeStamp = std::numeric_limits<float>::max();

        readHierarchyData(m_RootNode, scene->mRootNode);

        if(m_readOption.printTree)
            Helper::printNodeTree(&m_RootNode, 0);

        if(scene->mNumAnimations){
            // TODO handle more animation if have
            for(unsigned int i=0; i<scene->mNumAnimations; i++){
                auto animation = scene->mAnimations[i];
                m_animInfo.duration = animation->mDuration;
                m_animInfo.tick = animation->mTicksPerSecond;
                zeno::log_info("FBX: AniName {} NC {} NMC {} NMMC {} D {} T {}",
                               animation->mName.data,
                               animation->mNumChannels,
                               animation->mNumMeshChannels,
                               animation->mNumMorphMeshChannels,
                               animation->mDuration,
                               animation->mTicksPerSecond
                               );

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

        for (int i = 0; i < size; i++) {
            auto channel = animation->mChannels[i];
            std::string boneName(channel->mNodeName.data);
            //zeno::log_info("----- Name {}", boneName);

            SAnimBone bone;
            bone.initBone(boneName, channel);

            m_Bones.AnimBoneMap[boneName] = bone;

            m_animInfo.maxTimeStamp = std::max(m_animInfo.maxTimeStamp, bone.m_MaxTimeStamp);
            m_animInfo.minTimeStamp = std::min(m_animInfo.minTimeStamp, bone.m_MinTimeStamp);
        }
        // XXX -5 ~ 18  ---> 5 -> 28
        //zeno::log_info("SetupBones: Num AnimBoneMap: {}", m_Bones.AnimBoneMap.size());
        //std::cout << "FBX: MinTimeStamp " << m_animInfo.minTimeStamp
        //          << " MaxTimeStamp " << m_animInfo.maxTimeStamp << std::endl;
    }

    void setupBlendShape(const aiAnimation *animation){
        auto NumMorphChannel = animation->mNumMorphMeshChannels;
        zeno::log_info("FBX: BlendShape NumMorphChannel {}", NumMorphChannel);
        if(NumMorphChannel){
            for(int j=0; j<NumMorphChannel; j++){
                aiMeshMorphAnim* channel = animation->mMorphMeshChannels[j];
                std::string channelName(channel->mName.data);  // pPlane1*0 with *0
                channelName = channelName.substr(0, channelName.find_last_of('*'));
                zeno::log_info("FBX: BS Channel Name {}", channelName);
                std::vector<SKeyMorph> keyMorph;

                if(channel->mNumKeys) {
                    zeno::log_info("FBX: BS NumKeys {} NumVal&Wei {}", channel->mNumKeys,
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
                }else{
                    if(m_MeshBSName.find(channelName) != m_MeshBSName.end()){
                        auto& bsData = m_MeshBSName[channelName];

                        for(int i=0; i<bsData.size(); i++){
                            SKeyMorph morph{};
                            morph.m_Time = (float)i;
                            morph.m_Weights = new double[bsData.size()];
                            morph.m_Values = new unsigned int[bsData.size()];
                            morph.m_Weights[i] = 0.0; morph.m_Values[i] = 0;
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
        std::shared_ptr<zeno::DictObject>& prims,
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
        scene = importer.ReadFile(fbx_path, 0);
        mesh.createTexDir("valueTex");
        mesh.processNodeMat(scene->mRootNode, scene);
        for(auto const&[key, value]:mesh.m_loadedMat){
            mats->lut[key] = value.clone();
        }
        return;
    }

    if(true) {
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
    }else{
            bool nopointslines = false;
            float g_smoothAngle = 80.f;
            // default pp steps
            unsigned int ppsteps = aiProcess_CalcTangentSpace | // calculate tangents and bitangents if possible
                                   aiProcess_JoinIdenticalVertices    | // join identical vertices/ optimize indexing
                                   aiProcess_ValidateDataStructure    | // perform a full validation of the loader's output
                                   aiProcess_ImproveCacheLocality     | // improve the cache locality of the output vertices
                                   aiProcess_RemoveRedundantMaterials | // remove redundant materials
                                   aiProcess_FindDegenerates          | // remove degenerated polygons from the import
                                   aiProcess_FindInvalidData          | // detect invalid model data, such as invalid normal vectors
                                   aiProcess_GenUVCoords              | // convert spherical, cylindrical, box and planar mapping to proper UVs
                                   aiProcess_TransformUVCoords        | // preprocess UV transformations (scaling, translation ...)
                                   aiProcess_FindInstances            | // search for instanced meshes and remove them by references to one master
                                   aiProcess_LimitBoneWeights         | // limit bone weights to 4 per vertex
                                   aiProcess_OptimizeMeshes		   | // join small meshes, if possible;
                                   aiProcess_SplitByBoneCount         | // split meshes with too many bones. Necessary for our (limited) hardware skinning shader
                                   0;
            aiPropertyStore* props = aiCreatePropertyStore();
            aiSetImportPropertyInteger(props,AI_CONFIG_IMPORT_TER_MAKE_UVS,1);
            aiSetImportPropertyFloat(props,AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE,g_smoothAngle);
            aiSetImportPropertyInteger(props,AI_CONFIG_PP_SBP_REMOVE,nopointslines ? aiPrimitiveType_LINE | aiPrimitiveType_POINT : 0 );
            aiSetImportPropertyInteger(props,AI_CONFIG_GLOB_MEASURE_TIME,1);
            scene = (aiScene*)aiImportFileExWithProperties(fbx_path,
                                                                 ppsteps | /* configurable pp steps */
                                                                     aiProcess_GenSmoothNormals		   | // generate smooth normal vectors if not existing
                                                                     aiProcess_SplitLargeMeshes         | // split large, unrenderable meshes into submeshes
                                                                     aiProcess_Triangulate			   | // triangulate polygons with more than 3 edges
                                                                     aiProcess_ConvertToLeftHanded	   | // convert everything to D3D left handed space
                                                                     aiProcess_SortByPType              | // make 'clean' meshes which consist of a single typ of primitives
                                                                     0,
                                                                 nullptr,
                                                                 props);
            aiReleasePropertyStore(props);
    }

    if(! scene)
        zeno::log_error("FBX: Invalid assimp scene");

    mesh.initMesh(scene);
    anim.m_MeshBSName = mesh.m_BSName;
    anim.initAnim(scene, &mesh);

    mesh.processTrans(anim.m_Morph, anim.m_Bones.AnimBoneMap, datas, prims, mats, nodeTree, boneTree, animInfo);
    mesh.fbxData.iKeyMorph.value = anim.m_Morph;

    *data = mesh.fbxData;
    *nodeTree = anim.m_RootNode;
    *boneTree = anim.m_Bones;
    *animInfo = anim.m_animInfo;

    data->animInfo = animInfo;
    data->boneTree = boneTree;
    data->nodeTree = nodeTree;

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
        std::shared_ptr<zeno::DictObject> datas = std::make_shared<zeno::DictObject>();
        auto nodeTree = std::make_shared<NodeTree>();
        auto animInfo = std::make_shared<AnimInfo>();
        auto data = std::make_shared<FBXData>();
        auto boneTree = std::make_shared<BoneTree>();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::shared_ptr<zeno::DictObject> prims = std::make_shared<zeno::DictObject>();
        std::shared_ptr<zeno::DictObject> mats = std::make_shared<zeno::DictObject>();

        auto fbxFileName = Path(path)
                               .replace_extension("")
                               .filename()
                               .string();
        std::replace(fbxFileName.begin(),fbxFileName.end(), ' ', '_');
        zeno::log_info("FBX: File path {}, Replaced FBXName {}", path,fbxFileName);

        SFBXReadOption readOption;
        auto udim = get_param<std::string>("udim");
        auto primitive = get_param<bool>("primitive");
        auto generate = get_input2<bool>("generate");
        auto invOpacity = get_param<bool>("invOpacity");
        auto triangulate = get_param<bool>("triangulate");
        auto printTree = get_param<bool>("printTree");
        if (udim == "ENABLE")
            readOption.enableUDIM = true;
        if(invOpacity)
            readOption.invertOpacity = true;
        if(primitive)
            readOption.makePrim = true;
        if(generate)
            readOption.generate = true;
        if(triangulate)
            readOption.triangulate = true;
        if(printTree)
            readOption.printTree = true;

        zeno::log_info("FBX: UDIM {} PRIM {} INVERT {}", readOption.enableUDIM,readOption.makePrim,readOption.invertOpacity);

        readFBXFile(datas, nodeTree, data, boneTree, animInfo,
                    path.c_str(), prim, prims, mats, readOption);

        if(generate){
            int count = 0;
            for (auto &[k, v]: mats->lut) {
                auto vc = zeno::safe_dynamic_cast<SMaterial>(v);
                zeno::log_info("FBX: Setting user data {} {}", count, k);
                prim->userData().setLiterial(std::to_string(count), k);

                auto texLists = vc->getSimplestTexList();
                for(int i=0;i<texLists.size();i++){
                    prim->userData().setLiterial(
                        std::to_string(count)+"_tex_"+std::to_string(i),
                        texLists[i]);
                }

                count++;
            }
            prim->userData().setLiterial("matNum", count);
            prim->userData().setLiterial("fbxName", fbxFileName);
        }
        set_output("data", std::move(data));
        set_output("datas", std::move(datas));
        set_output("animinfo", std::move(animInfo));
        set_output("nodetree", std::move(nodeTree));
        set_output("bonetree", std::move(boneTree));
        set_output("prim", std::move(prim));
        set_output("prims", std::move(prims));
        set_output("mats", std::move(mats));
    }
};

ZENDEFNODE(ReadFBXPrim,
           {       /* inputs: */
               {
                   {"readpath", "path"},
                   {"bool", "generate", "false"}
               },  /* outputs: */
               {
                   "prim", "prims", "data", "datas", "mats",
                   "animinfo", "nodetree", "bonetree",
               },  /* params: */
               {
                {"enum ENABLE DISABLE", "udim", "DISABLE"},
                {"bool", "invOpacity", "true"},
                {"bool", "primitive", "false"},
                {"bool", "triangulate", "true"},
                {"bool", "printTree", "false"},
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
}
