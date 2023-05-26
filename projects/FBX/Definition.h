#ifndef ZENO_FBX_DEFINITION_H
#define ZENO_FBX_DEFINITION_H

#define ED_EMPTY
//#define ED_DEFINE_COUT

#ifdef ED_DEFINE_COUT
#define ED_COUT             std::cout
#define ED_CERR             std::cerr
#else
#define ED_COUT             /ED_EMPTY/
#define ED_CERR             /ED_EMPTY/
#endif

#include <limits>
#include <iostream>
#include <algorithm>
#include <zeno/utils/log.h>
#include <zeno/utils/vec.h>
#include <zeno/core/IObject.h>

inline namespace ZenoFBXDefinition {

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
#define COMMON_DEFAULT_transmissionColor aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)

#define COMMON_DEFAULT_defaultColor aiColor4D(0.0f, 0.0f, 0.0f, 1.0f)

#define LIGHT_STR_SPLIT_V3F                             \
    auto sl = Helper::splitStr(l, ',');                 \
    zeno::vec3f tmp{                                    \
        (float)atof(sl[0].c_str()),                     \
        (float)atof(sl[1].c_str()),                     \
        (float)atof(sl[2].c_str())};                    \
    auto no = std::make_shared<zeno::NumericObject>();  \
    no->set(tmp);

struct SFBXReadOption {
    bool invertOpacity = false;
    bool makePrim = false;
    bool enableUDIM = false;
    bool generate = false;
    bool triangulate = false;
    bool printTree = false;
    std::string hintPath = "";
    float offsetInSeconds = 0.0f;
};

struct SFBXEvalOption {
    bool writeData = false;
    bool interAnimData = false;
    bool printAnimData = false;
    float globalScale = 1.0f;
};

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

struct SKeyMorph {
    double m_Time;
    double *m_Weights;
    unsigned int *m_Values;
    unsigned int m_NumValuesAndWeights;
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
    float m_MaxTimeStamp = std::numeric_limits<float>::min();
    float m_MinTimeStamp = std::numeric_limits<float>::max();

    void initBone(std::string name, const aiNodeAnim* channel, double tickOffset){
        m_Name = name;
        m_NumPositions = channel->mNumPositionKeys;
        for (int positionIndex = 0; positionIndex < m_NumPositions; ++positionIndex) {
            aiVector3D aiPosition = channel->mPositionKeys[positionIndex].mValue;
            float timeStamp = channel->mPositionKeys[positionIndex].mTime + tickOffset;

            SKeyPosition data;
            data.position = aiPosition;
            data.timeStamp = timeStamp;
            m_Positions.push_back(data);

            //std::cout << " BoneAnim: index " << positionIndex << " " << timeStamp << "\n";
            //std::cout << "  Position: (" << aiPosition.x << ", " << aiPosition.y << ", " << aiPosition.z << ")\n";

            m_MaxTimeStamp = std::max(m_MaxTimeStamp, timeStamp);
            m_MinTimeStamp = std::min(m_MinTimeStamp, timeStamp);
        }

        //std::cout << " -----\n";

        m_NumRotations = channel->mNumRotationKeys;
        for (int rotationIndex = 0; rotationIndex < m_NumRotations; ++rotationIndex) {
            aiQuaternion aiOrientation = channel->mRotationKeys[rotationIndex].mValue;
            float timeStamp = channel->mRotationKeys[rotationIndex].mTime + tickOffset;

            SKeyRotation data;
            data.orientation = aiOrientation;
            data.timeStamp = timeStamp;
            m_Rotations.push_back(data);

            //std::cout << " BoneAnim: index " << rotationIndex << " " << timeStamp << "\n";
            //std::cout << "  Rotation: (" << aiOrientation.x << ", " << aiOrientation.y << ", " << aiOrientation.z << ", " << aiOrientation.w << ")\n";

            m_MaxTimeStamp = std::max(m_MaxTimeStamp, timeStamp);
            m_MinTimeStamp = std::min(m_MinTimeStamp, timeStamp);
        }

        //std::cout << " -----\n";

        m_NumScalings = channel->mNumScalingKeys;
        for (int scaleIndex = 0; scaleIndex < m_NumScalings; ++scaleIndex) {
            aiVector3D scale = channel->mScalingKeys[scaleIndex].mValue;
            float timeStamp = channel->mScalingKeys[scaleIndex].mTime + tickOffset;

            SKeyScale data;
            data.scale = scale;
            data.timeStamp = timeStamp;
            m_Scales.push_back(data);

            //std::cout << " BoneAnim: index " << scaleIndex << " " << timeStamp << "\n";
            //std::cout << "  Scale: (" << scale.x << ", " << scale.y << ", " << scale.z << ")\n";

            m_MaxTimeStamp = std::max(m_MaxTimeStamp, timeStamp);
            m_MinTimeStamp = std::min(m_MinTimeStamp, timeStamp);
        }

        //zeno::log_info("----- N {} NP {} NR {} NS {}",
        //               m_Name, m_NumPositions, m_NumRotations, m_NumScalings);
        //std::cout << "FBX: Anim Bone MaxTimeStamp " << m_MaxTimeStamp
        //          << " MinTimeStamp " << m_MinTimeStamp << std::endl;
    }

    void update(float animationTime) {
        aiMatrix4x4 translation = interpolatePosition(animationTime);
        aiMatrix4x4 rotation = interpolateRotation(animationTime);
        aiMatrix4x4 scale = interpolateScaling(animationTime);

        m_LocalTransform = translation * rotation * scale;
    }

    int getPositionIndex(float animationTime) {
        for (int index = 0; index < m_NumPositions - 1; ++index) {
            if (animationTime <= m_Positions[index + 1].timeStamp)
                return index;
        }
        return m_NumPositions-1;
    }
    int getRotationIndex(float animationTime) {
        for (int index = 0; index < m_NumRotations - 1; ++index) {
            if (animationTime <= m_Rotations[index + 1].timeStamp)
                return index;
        }
        return m_NumRotations-1;
    }
    int getScaleIndex(float animationTime) {
        for (int index = 0; index < m_NumScalings - 1; ++index) {
            if (animationTime <= m_Scales[index + 1].timeStamp)
                return index;
        }
        return m_NumScalings-1;
    }

    aiMatrix4x4 interpolatePosition(float animationTime) {
        aiMatrix4x4 result;

        if (1 == m_NumPositions) {
            aiMatrix4x4::Translation(m_Positions[0].position, result);
            return result;
        }

        int p0Index = getPositionIndex(animationTime);
        int p1Index = p0Index + 1;
        if(p1Index == m_NumPositions){
            p1Index = p0Index;
        }
        float scaleFactor = getScaleFactor(
                m_Positions[p0Index].timeStamp,
                m_Positions[p1Index].timeStamp,
                animationTime);
        //std::cout << "Interpolate Position: Index " << p0Index << " time " << animationTime << " factor " << scaleFactor << "\n";
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
        if(p1Index == m_NumRotations){
            p1Index = p0Index;
        }
        float scaleFactor = getScaleFactor(
                m_Rotations[p0Index].timeStamp,
                m_Rotations[p1Index].timeStamp,
                animationTime);
        //std::cout << "Interpolate Rotation: Index " << p0Index << " time " << animationTime << " factor " << scaleFactor << "\n";
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
        if(p1Index == m_NumScalings){
            p1Index = p0Index;
        }
        float scaleFactor = getScaleFactor(
                m_Scales[p0Index].timeStamp,
                m_Scales[p1Index].timeStamp,
                animationTime);
        //std::cout << "Interpolate Scaling : Index " << p0Index << " time " << animationTime << " factor " << scaleFactor << "\n";
        aiVector3D finalScale = m_Scales[p0Index].scale *  (1.0f - scaleFactor) + m_Scales[p1Index].scale * scaleFactor;
        aiMatrix4x4::Scaling(finalScale, result);

        return result;
    }

    float getScaleFactor(float lastTimeStamp, float nextTimeStamp, float animationTime) {
        //std::cout << " Stamp Factor " << lastTimeStamp << " " << nextTimeStamp << " " << animationTime << "\n";
        if(animationTime <= lastTimeStamp){
            return 0.0f;
        }else if(animationTime >= nextTimeStamp){
            return 1.0f;
        }

        // e.g. last: 1, next: 2, time: 1.5  -> (1.5-1)/(2-1)=0.5
        float midWayLength = animationTime - lastTimeStamp;
        float framesDiff = nextTimeStamp - lastTimeStamp;

        return midWayLength / framesDiff;
    }
};

struct SVertex{
    aiVector3D position;
    aiVector3D texCoord;
    aiVector3D normal;
    aiVector3D tangent;
    aiVector3D bitangent;
    aiColor4D vectexColor;
    std::unordered_map<std::string, float> boneWeights;
    float numAnimMesh;
};

struct SBSVertex{
    aiVector3D position;
    aiVector3D deltaPosition;
    aiVector3D normal;
    aiVector3D deltaNormal;
    float weight;
};

struct SMaterialProp{
    int order;
    bool forceDefault;
    std::any value;
    std::vector<aiTextureType> types;
    std::vector<std::string> aiNames;
    std::string texPath;
};

struct SDefaultMatProp{
    std::map<std::string, aiColor4D> getUnknownProp(){
        std::map<std::string, aiColor4D> val;

        val.emplace("basecolor", COMMON_DEFAULT_basecolor);
        val.emplace("metallic", COMMON_DEFAULT_metallic);
        val.emplace("diffuseRoughness", COMMON_DEFAULT_roughness);
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
        val.emplace("transmissionColor", COMMON_DEFAULT_transmissionColor);
        val.emplace("specularRoughness", COMMON_DEFAULT_defaultColor);
        val.emplace("displacement", COMMON_DEFAULT_defaultColor);
        return val;
    }
};


struct SFBXData : zeno::IObjectClone<SFBXData>{
    int jointIndices_elementSize = 0;
    std::vector<std::string> jointNames;
    std::vector<std::string> joints;
    std::vector<std::string> blendShapes;
    std::unordered_map<int, std::vector<float>> blendShapeWeights_timeSamples;
    std::vector<aiMatrix4x4> bindTransforms;
    std::vector<aiMatrix4x4> restTransforms;
    std::unordered_map<int, std::vector<zeno::vec4f>> rotations_timeSamples;
    std::unordered_map<int, std::vector<zeno::vec3f>> scales_timeSamples;
    std::unordered_map<int, std::vector<zeno::vec3f>> translations_timeSamples;
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
        // TODO trick - We use some unused tex properties to set some tex
        //
        val.emplace("basecolor", SMaterialProp{0,           false, aiColor4D(), {aiTextureType_BASE_COLOR, aiTextureType_DIFFUSE}, {"$ai.base", "$clr.diffuse"}});
        val.emplace("metallic", SMaterialProp{1,            false, aiColor4D(), {aiTextureType_METALNESS}, {"$ai.metalness"}});
        val.emplace("diffuseRoughness", SMaterialProp{2,    false, aiColor4D(), {aiTextureType_DIFFUSE_ROUGHNESS}, {"$ai.diffuseRoughness"}});
        val.emplace("specular", SMaterialProp{3,            false, aiColor4D(), {aiTextureType_SPECULAR}, {"$ai.specular", "$clr.specular"}});
        val.emplace("subsurface", SMaterialProp{4,          false, aiColor4D(), {aiTextureType_NONE}, {"$ai.subsurfaceFactor"}});
        val.emplace("thinkness", SMaterialProp{5,           true, aiColor4D(), {aiTextureType_NONE}, {"",}});
        val.emplace("sssParam", SMaterialProp{6,            false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("sssColor", SMaterialProp{7,            false, aiColor4D(), {aiTextureType_REFLECTION}, {"$ai.subsurface"}});
        val.emplace("foliage", SMaterialProp{8,             false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("skin", SMaterialProp{9,                false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("curvature", SMaterialProp{10,          false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("specularTint", SMaterialProp{11,       false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("anisotropic", SMaterialProp{12,        false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("sheen", SMaterialProp{13,              false, aiColor4D(), {aiTextureType_SHININESS}, {"$ai.sheen"}});
        val.emplace("sheenTint", SMaterialProp{14,          false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("clearcoat", SMaterialProp{15,          false, aiColor4D(), {aiTextureType_AMBIENT}, {"$ai.coat"}});
        val.emplace("clearcoatGloss", SMaterialProp{16,     true, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("normal", SMaterialProp{17,             false, aiColor4D(), {aiTextureType_NORMAL_CAMERA, aiTextureType_NORMALS}, {"",}});
        val.emplace("emission", SMaterialProp{18,           false, aiColor4D(), {aiTextureType_EMISSIVE, aiTextureType_EMISSION_COLOR}, {"$ai.emission", "$clr.emissive"}});
        val.emplace("exposure", SMaterialProp{19,           false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("ao", SMaterialProp{20,                 false, aiColor4D(), {aiTextureType_AMBIENT_OCCLUSION}, {""}});
        val.emplace("toon", SMaterialProp{21,               false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("stroke", SMaterialProp{22,             false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("shape", SMaterialProp{23,              false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("style", SMaterialProp{24,              false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("strokeNoise", SMaterialProp{25,        false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("shad", SMaterialProp{26,               false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("strokeTint", SMaterialProp{27,         false, aiColor4D(), {aiTextureType_NONE}, {""}});
        val.emplace("opacity", SMaterialProp{28,            false, aiColor4D(), {aiTextureType_LIGHTMAP}, {"$ai.opacity", "$clr.transparent"}});
        val.emplace("transmissionColor", SMaterialProp{29,  false, aiColor4D(), {aiTextureType_OPACITY}, {"$ai.transmission", "$clr.transparent"}});
        val.emplace("specularRoughness", SMaterialProp{30,  false, aiColor4D(), {aiTextureType_HEIGHT}, {"$ai.specularRoughness",}});
        val.emplace("displacement", SMaterialProp{31,       false, aiColor4D(), {aiTextureType_DISPLACEMENT}, {"",}});
    }

    std::vector<std::string> getTexList(){
        std::vector<std::string> tl;

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

    void getSimplestTexList(std::vector<std::string>& texList, std::map<std::string, int>& texMap){
        texList.clear();
        texMap.clear();

        std::map<std::string, SMaterialProp> val_tmp;
        val_tmp.emplace("basecolor", val["basecolor"]);                             //0
        val_tmp.emplace("metallic", val["metallic"]);                               //1
        val_tmp.emplace("diffuseRoughness", val["diffuseRoughness"]);               //2
        val_tmp.emplace("specular", val["specular"]);                               //3
        val_tmp.emplace("subsurface", val["subsurface"]);                           //4
        val_tmp.emplace("sssColor", val["sssColor"]);                               //5
        val_tmp.emplace("sheen", val["sheen"]);                                     //6
        val_tmp.emplace("clearcoat", val["clearcoat"]);                             //7
        val_tmp.emplace("normal", val["normal"]);                                   //8
        val_tmp.emplace("emission", val["emission"]);                               //9
        val_tmp.emplace("ao", val["ao"]);                                           //10
        val_tmp.emplace("opacity", val["opacity"]);                                 //11
        val_tmp.emplace("transmissionColor", val["transmissionColor"]);             //12
        val_tmp.emplace("specularRoughness", val["specularRoughness"]);             //13
        val_tmp.emplace("displacement", val["displacement"]);                       //14

        std::vector<pair> val_vec_tmp;

        std::copy(val_tmp.begin(),
                  val_tmp.end(),
                  std::back_inserter<std::vector<pair>>(val_vec_tmp));

        std::sort(val_vec_tmp.begin(), val_vec_tmp.end(),
                  [](const pair &l, const pair &r)
                  {
                      if (l.second.order != r.second.order) {
                          return l.second.order < r.second.order;
                      }

                      return l.first < r.first;
                  });
        for (int i=0; i<val_vec_tmp.size(); i++) {
            auto& p = val_vec_tmp[i];
            texList.emplace_back(p.second.texPath);
            texMap[p.first] = i;
        }
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
    float filmW;
    float filmH;
    float pNear;
    float pFar;
    zeno::vec3f interestPos;
    /*zeno::vec3f lookAt;*/
    zeno::vec3f pos;
    zeno::vec3f up;
    zeno::vec3f view;
    /*aiMatrix4x4 camM;*/
};

struct SLight{
    std::string mName;
    aiLightSourceType mType;
    aiVector3D mPosition;
    aiVector3D mDirection;
    aiVector3D mUp;
    float mAttenuationConstant;
    float mAttenuationLinear;
    float mAttenuationQuadratic;
    aiColor3D mColorDiffuse;
    aiColor3D mColorSpecular;
    aiColor3D mColorAmbient;
    float mAngleInnerCone;
    float mAngleOuterCone;
    aiVector2D mSize;
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
    float maxTimeStamp;
    float minTimeStamp;
};

struct IMaterial : zeno::IObjectClone<IMaterial>{
    std::unordered_map<std::string, SMaterial> value;  // key: meshName
};

struct IBoneOffset : zeno::IObjectClone<IBoneOffset>{
    std::unordered_map<std::string, SBoneOffset> value;
};

struct IPathTrans : zeno::IObjectClone<IPathTrans>{
    std::unordered_map<std::string, aiMatrix4x4> value;
};

struct ICamera : zeno::IObjectClone<ICamera>{
    std::unordered_map<std::string, SCamera> value;
};

struct ILight : zeno::IObjectClone<ILight>{
    std::unordered_map<std::string, SLight> value;
};

struct IVertices : zeno::IObjectClone<IVertices>{
    std::vector<SVertex> value;
};

struct IIndices : zeno::IObjectClone<IIndices>{
    std::vector<unsigned int> valueTri;
    std::vector<unsigned int> valueLoops;
    std::vector<zeno::vec2i> valuePolys;
};

struct IBlendSData : zeno::IObjectClone<IBlendSData>{
    // value: one-dimensional: Anim Mesh, two-dimensional: Mesh Vertices
    std::unordered_map<std::string, std::vector<std::vector<SBSVertex>>> value;
};

struct IKeyMorph : zeno::IObjectClone<IKeyMorph>{
    std::unordered_map<std::string, std::vector<SKeyMorph>> value;
};

struct IMeshName : zeno::IObjectClone<IMeshName>{
    std::string value;
    std::string value_relName;
    std::string value_matName;
};

struct IPathName : zeno::IObjectClone<IMeshName>{
    std::string value;
    std::string value_oriPath;
};

struct IMeshInfo : zeno::IObjectClone<IMeshInfo>{
    std::unordered_map<std::string, std::string> value_corsName;
};

struct FBXData : zeno::IObjectClone<FBXData>{
    IMeshName iMeshName;
    IPathName iPathName;
    IMeshInfo iMeshInfo;
    IVertices iVertices;
    IIndices iIndices;
    IBlendSData iBlendSData;
    IKeyMorph iKeyMorph;
    IBoneOffset iBoneOffset;
    IPathTrans iPathTrans;

    IMaterial iMaterial;
    ICamera iCamera;
    ILight iLight;

    zeno::DictObject iVisibility;

    std::shared_ptr<BoneTree> boneTree;
    std::shared_ptr<NodeTree> nodeTree;
    std::shared_ptr<AnimInfo> animInfo;
};

struct IFBXData : zeno::IObjectClone<IFBXData>{
    std::unordered_map<std::string, std::shared_ptr<FBXData>> value;
};

struct MatData : zeno::IObjectClone<MatData>{
    IFBXData iFbxData;
    SMaterial sMaterial;
};

struct IMatData : zeno::IObjectClone<IMatData>{
    std::unordered_map<std::string, MatData> value;
};

struct Helper{
    static void printAiMatrix(aiMatrix4x4 m, bool transpose = false){

        std::cout.precision(4);
        std::cout << std::fixed;
        std::cout << " ("<<m[0][0]<<","<<m[0][1]<<","<<m[0][2]<<","<<m[0][3]<<", "
                         <<m[1][0]<<","<<m[1][1]<<","<<m[1][2]<<","<<m[1][3]<<", "
                         <<m[2][0]<<","<<m[2][1]<<","<<m[2][2]<<","<<m[2][3]<<", "
                         <<m[3][0]<<","<<m[3][1]<<","<<m[3][2]<<","<<m[3][3]
                         <<")\n";

        aiVector3t<float> trans;
        aiQuaterniont<float> rotate;
        aiVector3t<float> scale;
        m.Decompose(scale, rotate, trans);
        std::cout << std::fixed;
        std::cout << " T ("<<trans.x<<","<<trans.y<<","<<trans.z<<")"
                  << " R ("<<rotate.x<<","<<rotate.y<<","<<rotate.z<<","<<rotate.w<<")"
                  << " S ("<<scale.x<<","<<scale.y<<","<<scale.z<<")"
                <<")\n";

        aiMatrix3x3 r = rotate.GetMatrix();
        if (transpose)
            r = rotate.GetMatrix().Transpose();
        std::cout << std::fixed;
        std::cout << " ("<<r[0][0]<<","<<r[0][1]<<","<<r[0][2]<<", "
                         <<r[1][0]<<","<<r[1][1]<<","<<r[1][2]<<", "
                         <<r[2][0]<<","<<r[2][1]<<","<<r[2][2]
                         <<")\n";
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
            std::cout << " ";
        auto t = root->transformation;
        std::cout.precision(4);
        std::cout << root->name <<" ("<<t[0][0]<<","<<t[0][1]<<","<<t[0][2]<<","<<t[0][3]<<", "
                                      <<t[1][0]<<","<<t[1][1]<<","<<t[1][2]<<","<<t[1][3]<<", "
                                      <<t[2][0]<<","<<t[2][1]<<","<<t[2][2]<<","<<t[2][3]<<", "
                                      <<t[3][0]<<","<<t[3][1]<<","<<t[3][2]<<","<<t[3][3]
                                <<")\n";
    }

    static std::vector<std::string> splitStr(std::string str, char c){
        std::stringstream test(str);
        std::string segment;
        std::vector<std::string> seglist;

        while(std::getline(test, segment, c))
        {
            seglist.push_back(segment);
        }
        return seglist;
    }
};

struct BezierCompute{
    template <class T>
    static T interpolation(T p1, T p2, float t){
        return (1-t)*p1+t*p2;
    }

    template <class T>
    static T bezier( std::vector<T> const&p, float t ){
        std::vector<T> ps = p;
        auto iter = ps.size();
        for(int z=0; z<iter; z++){
            auto n=ps.size();
            std::vector<T> tmp;
            for(int i=0;i<n-1;i++){
                auto cr = T(interpolation(ps[i], ps[i+1], t));
                tmp.push_back(cr);
            }
            ps=tmp;
            iter--;
        }
        return interpolation(ps[0], ps[1], t);
    }

    static zeno::vec3f compute(float c1of, float c2of, float factor, zeno::vec3f n, zeno::vec3f nm){
        std::vector<zeno::vec3f> v_x;
        std::vector<zeno::vec3f> v_y;
        std::vector<zeno::vec3f> v_z;
        v_x.push_back(zeno::vec3f(0.0f, nm[0], 0.0f));
        v_x.push_back(zeno::vec3f(c1of, nm[0], 0.0f));
        v_x.push_back(zeno::vec3f(c2of, n[0], 0.0f));
        v_x.push_back(zeno::vec3f(1.0f, n[0], 0.0f));
        v_y.push_back(zeno::vec3f(0.0f, nm[1], 0.0f));
        v_y.push_back(zeno::vec3f(c1of, nm[1], 0.0f));
        v_y.push_back(zeno::vec3f(c2of, n[1], 0.0f));
        v_y.push_back(zeno::vec3f(1.0f, n[1], 0.0f));
        v_z.push_back(zeno::vec3f(0.0f, nm[2], 0.0f));
        v_z.push_back(zeno::vec3f(c1of, nm[2], 0.0f));
        v_z.push_back(zeno::vec3f(c2of, n[2], 0.0f));
        v_z.push_back(zeno::vec3f(1.0f, n[2], 0.0f));
        auto b_pos_x = BezierCompute::bezier(v_x, factor);
        auto b_pos_y = BezierCompute::bezier(v_y, factor);
        auto b_pos_z = BezierCompute::bezier(v_z, factor);
        auto result = zeno::vec3f(b_pos_x[1], b_pos_y[1], b_pos_z[1]);

        return result;
    }

    static float compute(float c1of, float c2of, float factor, float n, float nm){
        std::vector<zeno::vec3f> tp_v;
        tp_v.push_back(zeno::vec3f(0.0f, nm, 0.0f));
        tp_v.push_back(zeno::vec3f(c1of, nm, 0.0f));
        tp_v.push_back(zeno::vec3f(c2of, n, 0.0f));
        tp_v.push_back(zeno::vec3f(1.0f, n, 0.0f));
        auto b_v = BezierCompute::bezier(tp_v, factor);

        return b_v[1];
    }

};

}

#endif //ZENO_FBX_DEFINITION_H
