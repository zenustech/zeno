#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/UserData.h>

#include "assimp/scene.h"

#include "Definition.h"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <unordered_map>

namespace {

struct EvalAnim{
    float m_CurrentFrame;
    float m_DeltaTime;

    SFBXEvalOption m_evalOption;
    NodeTree m_RootNode;
    SFBXData m_FbxData;
    AnimInfo m_animInfo;

    std::unordered_map<std::string, aiMatrix4x4> m_Transforms;
    std::unordered_map<std::string, aiMatrix4x4> m_LazyTransforms;
    std::unordered_map<std::string, SBoneOffset> m_BoneOffset;
    std::unordered_map<std::string, SAnimBone> m_AnimBones;

    std::vector<SVertex> m_Vertices;
    std::vector<unsigned int> m_Indices;

    std::unordered_map<std::string, int> m_JointCorrespondingIndex;

    void initAnim(std::shared_ptr<NodeTree>& nodeTree,
                  std::shared_ptr<BoneTree>& boneTree,
                  std::shared_ptr<FBXData>& fbxData,
                  std::shared_ptr<AnimInfo>& animInfo){
        m_animInfo = *animInfo;

        m_Vertices = fbxData->iVertices.value;
        m_Indices = fbxData->iIndices.value;

        m_RootNode = *nodeTree;
        m_AnimBones = boneTree->AnimBoneMap;
        m_BoneOffset = fbxData->iBoneOffset.value;

        m_CurrentFrame = 0.0f;
    }

    void updateAnimation(int fi, std::shared_ptr<zeno::PrimitiveObject>& prim, float fps) {
        // TODO Use the actual frame number
        float dt = fi / fps;
        m_DeltaTime = dt;
        m_CurrentFrame += m_animInfo.tick * dt;
        m_CurrentFrame = fmod(m_CurrentFrame, m_animInfo.duration);

        //zeno::log_info("Update: F {} D {} C {}", fi, dt, m_CurrentFrame);
        //std::cout << "FBX: CurrentFrame " << m_CurrentFrame << " Duration "
        //          << m_animInfo.duration << " DeltaTime " << m_DeltaTime << std::endl;

        if(m_evalOption.writeData){
            expandBoneTransform(&m_RootNode, "", aiMatrix4x4());
            calculateMaxBoneInfluence();

            for(float s = m_animInfo.minTimeStamp; s<=m_animInfo.maxTimeStamp; s+=1.0f){
                //std::cout << "FBX: Calculate Anim Transform Time " << s << std::endl;
                calculateAnimTransform(&m_RootNode, s);
            }
        }

        calculateBoneTransform(&m_RootNode, aiMatrix4x4());
        calculateFinal(prim);
    }

    void decomposeAnimation(std::shared_ptr<zeno::DictObject> &t,
                            std::shared_ptr<zeno::DictObject> &r,
                            std::shared_ptr<zeno::DictObject> &s){

        for(auto& m:m_Transforms){
            //zeno::log_info("A {}", m.first);
            aiVector3t<float> trans;
            aiQuaterniont<float> rotate;
            aiVector3t<float> scale;
            m.second.Decompose(scale, rotate, trans);
            //zeno::log_info("    T {: f} {: f} {: f}", trans.x, trans.y, trans.z);
            //zeno::log_info("    R {: f} {: f} {: f} {: f}", rotate.x, rotate.y, rotate.z, rotate.w);
            //zeno::log_info("    S {: f} {: f} {: f}", scale.x, scale.y, scale.z);

            auto nt = std::make_shared<zeno::NumericObject>();
            nt->value = zeno::vec3f(trans.x, trans.y, trans.z);

            auto nr = std::make_shared<zeno::NumericObject>();
            nr->value = zeno::vec4f(rotate.x, rotate.y, rotate.z, rotate.w);

            auto ns = std::make_shared<zeno::NumericObject>();
            ns->value = zeno::vec3f(scale.x, scale.y, scale.z);

            t->lut[m.first] = nt;
            r->lut[m.first] = nr;
            s->lut[m.first] = ns;
        }
    }

    void calculateMaxBoneInfluence(){
        for(unsigned int i=0; i<m_Vertices.size(); i++) {
            int s = m_Vertices[i].boneWeights.size();
            m_FbxData.jointIndices_elementSize = std::max(s, m_FbxData.jointIndices_elementSize);
        }
        //std::cout << "FBX: MaxJointInfluence " << m_FbxData.jointIndices_elementSize << std::endl;
    }

    void calculateAnimTransform(const NodeTree *node, float timeCode){
        std::string nodeName = node->name;
        aiVector3t<float> trans{0.0f,0.0f,0.0f};
        aiQuaterniont<float> rotate;
        aiVector3t<float> scale{1.0f,1.0f,1.0f};

        if (m_AnimBones.find(nodeName) != m_AnimBones.end()) {
            auto& bone = m_AnimBones[nodeName];
            bone.update(timeCode);
            bone.m_LocalTransform.Decompose(scale, rotate, trans);
        }

        m_FbxData.rotations_timeSamples[timeCode].emplace_back(rotate.x,rotate.y,rotate.z,rotate.w);
        m_FbxData.translations_timeSamples[timeCode].emplace_back(trans.x,trans.y,trans.z);
        m_FbxData.scales_timeSamples[timeCode].emplace_back(scale.x,scale.y,scale.z);

        for (int i = 0; i < node->childrenCount; i++)
            calculateAnimTransform(&node->children[i], timeCode);
    }

    void expandBoneTransform(const NodeTree *node, std::string pPath, aiMatrix4x4 parentTransform) {
        std::string nodeName = node->name;
        std::string cName;
        pPath.empty() ? cName = nodeName : cName = pPath + "/" + nodeName;
        aiMatrix4x4 nodeTransform = node->transformation;

        m_FbxData.joints.push_back(cName);
        m_FbxData.jointNames.push_back(nodeName);
        m_JointCorrespondingIndex[nodeName] = m_FbxData.joints.size()-1;
        //std::cout << "FBX: Bone name " << nodeName << " " << cName << " " << m_FbxData.joints.size() << std::endl;

        aiMatrix4x4 globalTransformation = parentTransform * nodeTransform;
        m_FbxData.restTransforms.push_back(globalTransformation);
        m_FbxData.bindTransforms.push_back(nodeTransform);

        if (m_BoneOffset.find(nodeName) != m_BoneOffset.end()) {  // found
            std::string boneName = m_BoneOffset[nodeName].name;
            aiMatrix4x4 boneOffset = m_BoneOffset[nodeName].offset;
            m_Transforms[boneName] = globalTransformation * boneOffset;
        }

        for (int i = 0; i < node->childrenCount; i++)
            expandBoneTransform(&node->children[i], cName, globalTransformation);
    }

    void calculateBoneTransform(const NodeTree *node, aiMatrix4x4 parentTransform) {
        std::string nodeName = node->name;
        aiMatrix4x4 nodeTransform = node->transformation;

        //zeno::log_info("***** {}", nodeName);
        // Any object that just has the key-anim is a bone
        if (m_AnimBones.find(nodeName) != m_AnimBones.end()) {
            auto& bone = m_AnimBones[nodeName];

            bone.update(m_CurrentFrame);
            nodeTransform = bone.m_LocalTransform;

            //std::cout << "FBX: Anim Node Name " << nodeName << std::endl;
            //Helper::printAiMatrix(nodeTransform);
        }
        aiMatrix4x4 globalTransformation = parentTransform * nodeTransform;

        // XXX Lazy Transform
        if (m_BoneOffset.find(nodeName) != m_BoneOffset.end()) {  // found
            std::string boneName = m_BoneOffset[nodeName].name;
            aiMatrix4x4 boneOffset = m_BoneOffset[nodeName].offset;
            //std::cout << "FBX: Bone Node Name " << nodeName << std::endl;
            //Helper::printAiMatrix(boneOffset);

            m_Transforms[boneName] = globalTransformation * boneOffset;
        }else{
            // The child is already applied the parent transformation by the tree struct.
            m_LazyTransforms[nodeName] = globalTransformation;
            //std::cout << "FBX: Lazy Node Name " << nodeName << std::endl;
        }

        for (int i = 0; i < node->childrenCount; i++)
            calculateBoneTransform(&node->children[i], globalTransformation);
    }

    void updateCameraAndLight(std::shared_ptr<FBXData>& fbxData,
                              std::shared_ptr<ICamera>& iCamera,
                              std::shared_ptr<ILight>& iLight)
    {
        float s = m_evalOption.globalScale;
        for(auto& m: m_LazyTransforms){
            if(fbxData->iCamera.value.find(m.first) != fbxData->iCamera.value.end()){
                //zeno::log_info("----- LT Camera {}", m.first);
                //Helper::printAiMatrix(m.second, true);

                SCamera cam = fbxData->iCamera.value.at(m.first);

                aiVector3t<float> trans;
                aiQuaterniont<float> rotate;
                aiVector3t<float> scale;
                m.second.Decompose(scale, rotate, trans);
                cam.pos = zeno::vec3f(trans.x*s, trans.y*s, trans.z*s);
                aiMatrix3x3 r = rotate.GetMatrix().Transpose();
                cam.view = zeno::vec3f(r.a1, r.a2, r.a3);
                cam.up = zeno::vec3f(r.b1, r.b2, r.b3);

                iCamera->value[m.first] = cam;
            }else if(fbxData->iLight.value.find(m.first) != fbxData->iLight.value.end()){
                //zeno::log_info("+++++ LT Light {}", m.first);
                //Helper::printAiMatrix(m.second, true);
            }
        }
    }

    void calculateFinal(std::shared_ptr<zeno::PrimitiveObject>& prim){
        auto &ver = prim->verts;
        auto &ind = prim->tris;
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &posb = prim->verts.add_attr<zeno::vec3f>("posb");
        auto &clr0 = prim->verts.add_attr<zeno::vec3f>("clr0");
        for(int i=0;i<m_FbxData.jointIndices_elementSize;i++){
            prim->verts.add_attr<float>("jointIndice_" + std::to_string(i));
            prim->verts.add_attr<float>("jointWeight_" + std::to_string(i));
        }
        int jie = m_FbxData.jointIndices_elementSize;
        prim->userData().set2("jointIndicesElementSize", std::move(jie));
        float s = m_evalOption.globalScale;

        for(unsigned int i=0; i<m_Vertices.size(); i++){
            auto& bwe = m_Vertices[i].boneWeights;
            auto& pos = m_Vertices[i].position;
            auto& uvw = m_Vertices[i].texCoord;
            auto& nor = m_Vertices[i].normal;
            auto& vco = m_Vertices[i].vectexColor;

            glm::vec4 tpos(0.0f, 0.0f, 0.0f, 0.0f);

            bool infd = false;
            int bCount = 0;
            for(auto& b: bwe){
                if(jie){
                    float bIndex = (float)m_JointCorrespondingIndex[b.first];
                    prim->verts.attr<float>("jointIndice_" + std::to_string(bCount)).push_back(bIndex);
                    prim->verts.attr<float>("jointWeight_" + std::to_string(bCount)).push_back(b.second);
                }
                //std::cout << "FBX: Vert " << i << " name " << b.first << " bIndex " << bIndex << std::endl;
                infd = true;
                auto& tr = m_Transforms[b.first];
                glm::mat4 trans = glm::mat4(tr.a1,tr.b1,tr.c1,tr.d1,
                                            tr.a2,tr.b2,tr.c2,tr.d2,
                                            tr.a3,tr.b3,tr.c3,tr.d3,
                                            tr.a4,tr.b4,tr.c4,tr.d4);
                glm::vec4 lpos = trans * glm::vec4(pos.x, pos.y, pos.z, 1.0f);
                tpos += lpos * b.second;
                bCount += 1;
            }
            // Supplement, joint index supplement 0, weight 0
            for(int z=bCount; z<m_FbxData.jointIndices_elementSize; z++){
                prim->verts.attr<float>("jointIndice_" + std::to_string(bCount)).push_back(0);
                prim->verts.attr<float>("jointWeight_" + std::to_string(bCount)).push_back(0.0f);
            }
            if(! infd)
                tpos = glm::vec4(pos.x, pos.y, pos.z, 1.0f);

            glm::vec3 fpos = glm::vec3(tpos.x/tpos.w, tpos.y/tpos.w, tpos.z/tpos.w);

            ver.emplace_back(fpos.x*s, fpos.y*s, fpos.z*s);
            posb.emplace_back(0.0f, 0.0f, 0.0f);
            uv.emplace_back(uvw.x, uvw.y, uvw.z);
            norm.emplace_back(nor.x, nor.y, nor.z);
            clr0.emplace_back(vco.r, vco.g, vco.b);
        }

        for(unsigned int i=0; i<m_Indices.size(); i+=3){
            zeno::vec3i incs(m_Indices[i],m_Indices[i+1],m_Indices[i+2]);
            ind.push_back(incs);
        }

        auto &uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
        auto &uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
        auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
        for(unsigned int i=0; i<ind.size(); i++){
            unsigned int _i1 = ind[i][0];
            unsigned int _i2 = ind[i][1];
            unsigned int _i3 = ind[i][2];
            uv0[i] = zeno::vec3f(m_Vertices[_i1].texCoord[0], m_Vertices[_i1].texCoord[1], 0);
            uv1[i] = zeno::vec3f(m_Vertices[_i2].texCoord[0], m_Vertices[_i2].texCoord[1], 0);
            uv2[i] = zeno::vec3f(m_Vertices[_i3].texCoord[0], m_Vertices[_i3].texCoord[1], 0);

        }
    }
};

struct EvalFBXAnim : zeno::INode {

    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input<zeno::NumericObject>("frameid")->get<int>();
        } else {
            frameid = getGlobalState()->frameid;
        }

        SFBXEvalOption evalOption;
        auto fbxData = get_input<FBXData>("data");
        auto fps = get_input2<float>("fps");
        auto unit = get_param<std::string>("unit");
        auto interAnimData = get_param<std::string>("interAnimData");
        auto paramWriteData = get_param<bool>("writeData");
        unit == "FROM_MAYA" ? evalOption.globalScale = 0.01f : evalOption.globalScale = 1.0f;
        interAnimData == "TRUE" ? evalOption.interAnimData = true : evalOption.interAnimData = false;
        paramWriteData == true ? evalOption.writeData = true : evalOption.writeData = false;

        auto nodeTree = evalOption.interAnimData ? fbxData->nodeTree : get_input<NodeTree>("nodetree");
        auto boneTree = evalOption.interAnimData ? fbxData->boneTree : get_input<BoneTree>("bonetree");
        auto animInfo = evalOption.interAnimData ? fbxData->animInfo : get_input<AnimInfo>("animinfo");

        if(nodeTree == nullptr || boneTree == nullptr || animInfo == nullptr){
            zeno::log_error("FBX: Empty NodeTree, BoneTree or AnimInfo");
        }

        zeno::log_info("FBX: Eval Option InterAnimData {} WriteData {} UnitScale {}",
                       evalOption.interAnimData, evalOption.writeData, evalOption.globalScale);

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto transDict = std::make_shared<zeno::DictObject>();
        auto quatDict = std::make_shared<zeno::DictObject>();
        auto scaleDict = std::make_shared<zeno::DictObject>();
        auto iCamera = std::make_shared<ICamera>();
        auto iLight = std::make_shared<ILight>();
        auto matName = std::make_shared<zeno::StringObject>();
        auto outMeshName = std::make_shared<zeno::StringObject>();

        EvalAnim anim;
        anim.m_evalOption = evalOption;
        anim.initAnim(nodeTree, boneTree, fbxData, animInfo);
        anim.updateAnimation(frameid, prim, fps);
        anim.updateCameraAndLight(fbxData, iCamera, iLight);
        anim.decomposeAnimation(transDict, quatDict, scaleDict);

        auto bsPrims = std::make_shared<zeno::ListObject>();
        auto bsPrimsOrigin = std::make_shared<zeno::ListObject>();
        auto& meshName = fbxData->iMeshName.value_relName;

        matName->set(fbxData->iMeshName.value_matName);
        outMeshName->set(meshName);

        auto& kmValue = fbxData->iKeyMorph.value;
        auto& bsValue = fbxData->iBlendSData.value;
        float gScale = evalOption.globalScale;

        for(auto const& [bsName, bss]: bsValue){
            std::cout << "BlendShape Key " << bsName << "\n";
            for(int i=0; i< bss.size(); i++){
                auto bsprim = std::make_shared<zeno::PrimitiveObject>();
                auto &verAttr = bsprim->verts;
                auto &indAttr = bsprim->tris;
                auto &nrmAttr = bsprim->verts.add_attr<zeno::vec3f>("nrm");
                auto &dnrmAttr = bsprim->verts.add_attr<zeno::vec3f>("dnrm");
                auto &dposAttr = bsprim->verts.add_attr<zeno::vec3f>("dpos");
                auto& bs = bss[i];
                for(unsigned int j=0; j<bs.size(); j++){ // Mesh Vert
                    auto& vdata = bs[j];
                    auto& pos = vdata.position;
                    auto& nrm = vdata.normal;
                    auto& dpos = vdata.deltaPosition;
                    auto& dnrm = vdata.deltaNormal;
                    verAttr.emplace_back(pos.x * gScale, pos.y * gScale, pos.z * gScale);
                    nrmAttr.emplace_back(nrm.x, nrm.y, nrm.z);
                    dposAttr.emplace_back(dpos.x * gScale, dpos.y * gScale, dpos.z * gScale);
                    dnrmAttr.emplace_back(dnrm.x, dnrm.y, dnrm.z);
                }

                bsPrimsOrigin->arr.emplace_back(bsprim);
            }
        }

        // TODO FBXData Write BlendShape
        if(bsValue.find(meshName) != bsValue.end()){
            auto& b = bsValue[meshName];

            if(kmValue.find(meshName) != kmValue.end()){
                auto& k = kmValue[meshName];
                unsigned int ki = 0;
                unsigned int kin;
                for(unsigned int i=0; i<k.size()-1; i++){  // Find keyMorph index
                    if(anim.m_CurrentFrame < k[i+1].m_Time){ // Animation must occur between at least two frames
                        ki = i;
                    }
                }
                kin = ki+1;

                auto& kd = k[ki];
                auto& kdn = k[kin];
                float factor = (anim.m_CurrentFrame - kd.m_Time) / (kdn.m_Time - kd.m_Time);
                for(unsigned int i=0; i<b.size(); i++){ // Anim Mesh & Same as BlendShape WeightsAndValues
                    auto bsprim = std::make_shared<zeno::PrimitiveObject>();
                    auto &ver = bsprim->verts;
                    auto &ind = bsprim->tris;
                    auto &norm = bsprim->verts.add_attr<zeno::vec3f>("nrm");
                    auto &posb = bsprim->verts.add_attr<zeno::vec3f>("posb");
                    auto &bsw = bsprim->verts.add_attr<float>("bsw");
                    double w = kd.m_Weights[i] * (1.0f - factor) + kdn.m_Weights[i] * factor;
                    auto& v = b[i];
                    for(unsigned int j=0; j<v.size(); j++){ // Mesh Vert
                        auto& vpos = v[j].deltaPosition;
                        //v[j].position;
                        auto& vnor = v[j].deltaNormal;
                        ver.emplace_back(vpos.x*gScale, vpos.y*gScale, vpos.z*gScale);
                        posb.emplace_back(0.0f, 0.0f, 0.0f);
                        bsw.emplace_back((float)w);
                        norm.emplace_back(vnor.x, vnor.y, vnor.z);
                    }

                    bsPrims->arr.emplace_back(bsprim);
                }
            }else{
                zeno::log_info("BlendShape NotFound MorphKey {}", meshName);
            }
        }

        //zeno::log_info("Frame {} Prims Num {} Mesh Name {}", anim.m_CurrentFrame, bsPrims->arr.size(), meshName);

        auto writeData = std::make_shared<SFBXData>();
        *writeData = anim.m_FbxData;

        set_output("prim", std::move(prim));
        set_output("bsPrims", std::move(bsPrims));
        set_output("bsPrimsOrigin", std::move(bsPrimsOrigin));
        set_output("camera", std::move(iCamera));
        set_output("light", std::move(iLight));
        set_output("matName", std::move(matName));
        set_output("meshName", std::move(outMeshName));
        set_output("transDict", std::move(transDict));
        set_output("quatDict", std::move(quatDict));
        set_output("scaleDict", std::move(scaleDict));
        set_output("writeData", std::move(writeData));
    }
};
ZENDEFNODE(EvalFBXAnim,
           {       /* inputs: */
               {
                   {"frameid"},
                   {"float", "fps", "24.0"},
                   "data", "animinfo", "nodetree", "bonetree",
               },  /* outputs: */
               {
                   "prim", "camera", "light", "matName", "meshName", "bsPrims", "bsPrimsOrigin",
                   "transDict", "quatDict", "scaleDict",
                   "writeData"
               },  /* params: */
               {
                   {"enum FROM_MAYA DEFAULT", "unit", "FROM_MAYA"},
                   {"enum TRUE FALSE", "interAnimData", "FALSE"},
                   {"bool", "writeData", "false"},
               },  /* category: */
               {
                   "FBX",
               }
           });

}
