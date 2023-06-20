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
    IMeshName m_meshName;
    IPathName m_pathName;
    IPathTrans m_PathTrans;

    std::unordered_map<std::string, aiMatrix4x4> m_BoneTransforms;
    std::unordered_map<std::string, aiMatrix4x4> m_LazyTransforms;
    std::unordered_map<std::string, SBoneOffset> m_BoneOffset;
    std::unordered_map<std::string, SAnimBone> m_AnimBones;

    std::vector<SVertex> m_Vertices;
    std::vector<unsigned int> m_IndicesTris;
    std::vector<unsigned int> m_IndicesLoops;
    std::vector<zeno::vec2i> m_IndicesPolys;

    std::unordered_map<std::string, int> m_JointCorrespondingIndex;

    void initAnim(std::shared_ptr<NodeTree>& nodeTree,
                  std::shared_ptr<BoneTree>& boneTree,
                  std::shared_ptr<FBXData>& fbxData,
                  std::shared_ptr<AnimInfo>& animInfo){
        m_animInfo = *animInfo;

        m_Vertices = fbxData->iVertices.value;
        m_IndicesTris = fbxData->iIndices.valueTri;
        m_IndicesPolys = fbxData->iIndices.valuePolys;
        m_IndicesLoops = fbxData->iIndices.valueLoops;

        m_meshName = fbxData->iMeshName;
        m_pathName = fbxData->iPathName;
        m_PathTrans = fbxData->iPathTrans;

        m_RootNode = *nodeTree;
        m_AnimBones = boneTree->AnimBoneMap;
        m_BoneOffset = fbxData->iBoneOffset.value;

        m_CurrentFrame = 0.0f;
    }

    void updateAnimation(int fi, std::shared_ptr<zeno::PrimitiveObject>& prim) {
        // TODO Use the actual frame number
        auto tick_fps = m_animInfo.tick;
        float dt = (float)fi / tick_fps;
        m_DeltaTime = dt;
        m_CurrentFrame += tick_fps * dt;
        //m_CurrentFrame = fmod(m_CurrentFrame, m_animInfo.duration);

        //zeno::log_info("Update: F {} D {} C {}", fi, dt, m_CurrentFrame);
        //ED_COUT << "FBX: Frame " << m_CurrentFrame << " Duration " << m_animInfo.duration << " " << m_pathName.value << std::endl;
        //ED_COUT << "FBX: FrameID " << fi << " Tick " << m_animInfo.tick << " DeltaTime " << dt << std::endl;

        if(m_evalOption.writeData){
            expandBoneTransform(&m_RootNode, "", aiMatrix4x4());
            calculateMaxBoneInfluence();

            for(float s = m_animInfo.minTimeStamp; s<=m_animInfo.maxTimeStamp; s+=1.0f){
                //std::cout << "FBX: Calculate Anim Transform Time " << s << std::endl;
                calculateAnimTransform(&m_RootNode, s);
            }
        }

        if(m_evalOption.printAnimData) {
            std::cout << "----- >" << m_pathName.value << "\n";
        }
//        TIMER_START(UpdateAnim_CalcTrans)
        calculateBoneTransform(&m_RootNode, aiMatrix4x4(), "");
//        TIMER_END(UpdateAnim_CalcTrans)

//        TIMER_START(UpdateAnim_CalcPrim)
        calculateFinal(prim);
//        TIMER_END(UpdateAnim_CalcPrim)

        if(m_evalOption.printAnimData) {
            std::cout << "===== <" << m_pathName.value << "\n";
        }
    }

    void decomposeAnimation(std::shared_ptr<zeno::DictObject> &t,
                            std::shared_ptr<zeno::DictObject> &r,
                            std::shared_ptr<zeno::DictObject> &s){

        for(auto& m:m_BoneTransforms){
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
            m_BoneTransforms[boneName] = globalTransformation * boneOffset;
        }

        for (int i = 0; i < node->childrenCount; i++)
            expandBoneTransform(&node->children[i], cName, globalTransformation);
    }

    void calculateBoneTransform(const NodeTree *node, aiMatrix4x4 parentTransform, std::string parent_path) {
        std::string nodeName = node->name;
        aiMatrix4x4 nodeTransform = node->transformation;
        auto pathName = parent_path + "/" + nodeName;

        if(m_evalOption.printAnimData) {
            std::cout << "---------- ---------- ----------\n";
            std::cout << "FBX: ***** Node Name " << nodeName << std::endl;
            Helper::printAiMatrix(nodeTransform);
            std::cout << "++++++++++ ++++++++++ ++++++++++\n";
            Helper::printAiMatrix(parentTransform);
            std::cout << "---------- ---------- ----------\n";
        }

        // Any object that just has the key-anim is a bone
        if (m_AnimBones.find(nodeName) != m_AnimBones.end()) {
            auto& bone = m_AnimBones[nodeName];

            bone.update(m_CurrentFrame);
            nodeTransform = bone.m_LocalTransform;

            if(m_evalOption.printAnimData) {
                std::cout << "FBX: Anim Node Name " << nodeName << std::endl;
                //Helper::printAiMatrix(nodeTransform);
            }
        }
        aiMatrix4x4 globalTransformation = parentTransform * nodeTransform;

        // XXX Lazy Transform
        if (m_BoneOffset.find(nodeName) != m_BoneOffset.end()) {  // found
            std::string boneName = m_BoneOffset[nodeName].name;
            aiMatrix4x4 boneOffset = m_BoneOffset[nodeName].offset;

            if(m_evalOption.printAnimData) {
                std::cout << "FBX: Bone Node Name " << nodeName << std::endl;
                //Helper::printAiMatrix(boneOffset);
            }

            m_BoneTransforms[boneName] = globalTransformation * boneOffset;
            if(m_evalOption.printAnimData) {
                //std::cout << "FBX: Transform " << boneName << "\n";
                //Helper::printAiMatrix(m_BoneTransforms[boneName]);
            }
        }

            // The child is already applied the parent transformation by the tree struct.
            m_LazyTransforms[pathName] = globalTransformation;

            if(m_evalOption.printAnimData) {
                std::cout << std::fixed << "FBX: Lazy Node Name " << nodeName << std::endl;
                Helper::printAiMatrix(globalTransformation);
            }

        for (int i = 0; i < node->childrenCount; i++)
            calculateBoneTransform(&node->children[i], globalTransformation, pathName);
    }

    void updateCameraAndLight(std::shared_ptr<FBXData>& fbxData,
                              std::shared_ptr<ICamera>& iCamera,
                              std::shared_ptr<ILight>& iLight)
    {
        float gscale = m_evalOption.globalScale;
        // TODO We didn't consider that the camera might be in the hierarchy
        for(auto& ltrans: m_LazyTransforms){
            auto namePath = ltrans.first;

            for(auto &[camName, camObj]: fbxData->iCamera.value){
                if(namePath.find(camName) != std::string::npos){

                    SCamera cam = fbxData->iCamera.value.at(camName);

                    aiVector3t<float> trans;
                    aiQuaterniont<float> rotate;
                    aiVector3t<float> scale;

                    ltrans.second.Decompose(scale, rotate, trans);

                    cam.pos = zeno::vec3f(trans.x * gscale, trans.y * gscale, trans.z * gscale);
                    aiMatrix3x3 r = rotate.GetMatrix().Transpose();
                    cam.view = zeno::vec3f(r.a1, r.a2, r.a3);
                    cam.up = zeno::vec3f(r.b1, r.b2, r.b3);

                    iCamera->value[camName] = cam;
                }
            }
        }
    }

    void getPathTrans(std::string pathName, glm::mat4& pathTrans, int& tranType){

        if(m_LazyTransforms.find(pathName) != m_LazyTransforms.end()){
            auto& tr = m_LazyTransforms[pathName];
            pathTrans = glm::mat4(tr.a1,tr.b1,tr.c1,tr.d1,
                                  tr.a2,tr.b2,tr.c2,tr.d2,
                                  tr.a3,tr.b3,tr.c3,tr.d3,
                                  tr.a4,tr.b4,tr.c4,tr.d4);
            if(m_evalOption.printAnimData) {
                std::cout << "Eval Lazy Trans\n";
                Helper::printAiMatrix(tr);
                std::cout << "===============\n";
            }
            tranType = 0;

        }else if(m_PathTrans.value.find(pathName) != m_PathTrans.value.end()) {
            auto& tr = m_PathTrans.value[pathName];
            pathTrans = glm::mat4(tr.a1,tr.b1,tr.c1,tr.d1,
                                  tr.a2,tr.b2,tr.c2,tr.d2,
                                  tr.a3,tr.b3,tr.c3,tr.d3,
                                  tr.a4,tr.b4,tr.c4,tr.d4);
            if(m_evalOption.printAnimData) {
                std::cout << "Eval Path Trans\n";
                Helper::printAiMatrix(tr);
                std::cout << "===============\n";
            }
            tranType = 1;
        }else{
            pathTrans = glm::mat4(1.0);
            std::cout << "Eval: Trans None " << pathName << "\n";
            tranType = 2;
        }
    }

    void calculateFinal(std::shared_ptr<zeno::PrimitiveObject>& prim){
        auto &ver = prim->verts;
        auto &trisInd = prim->tris;
        auto &polys = prim->polys;
        auto &loops = prim->loops;
        auto &uvs = prim->uvs;
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &posb = prim->verts.add_attr<zeno::vec3f>("posb");
        auto &clr0 = prim->verts.add_attr<zeno::vec3f>("clr0");
        bool isTris = false;

        //std::cout << "Eval name: " << m_meshName.value << "\n";
        //std::cout << "Eval name: " << m_meshName.value_matName << "\n";
        //std::cout << "Eval name: " << m_meshName.value_relName << "\n";
        //std::cout << "Eval name: " << m_pathName.value << "\n";
        //std::cout << "Eval name: " << m_pathName.value_oriPath << "\n";

        if(m_IndicesLoops.size() == 0){
            isTris = true;
        }else{
            isTris = false;
        }
        //std::cout << "mesh size loops " << m_IndicesLoops.size() << " tris " << m_IndicesTris.size() << " is tris " << isTris <<"\n";
        for(int i=0;i<m_FbxData.jointIndices_elementSize;i++){
            prim->verts.add_attr<float>("jointIndice_" + std::to_string(i));
            prim->verts.add_attr<float>("jointWeight_" + std::to_string(i));
        }
        int elemSize = m_FbxData.jointIndices_elementSize;
        prim->userData().set2("jointIndicesElementSize", elemSize);
        float gscale = m_evalOption.globalScale;

        // Trans
        auto pathName = m_pathName.value_oriPath;
        glm::mat4 pathTrans(1.0);
        int tranType = -1;
        if(pathName != "/__path__") {
            getPathTrans(pathName, pathTrans, tranType);
        }

        for(unsigned int i=0; i<m_Vertices.size(); i++){
            auto& bwe = m_Vertices[i].boneWeights;
            auto& pos = m_Vertices[i].position;
            auto& uvw = m_Vertices[i].texCoord;
            auto& nor = m_Vertices[i].normal;
            auto& vco = m_Vertices[i].vectexColor;

//            auto& exi = m_Vertices[i].extraInfos;
//            if(pathName == "/__path__") {
//                getPathTrans(exi["path"], pathTrans, tranType);
//            }

            glm::vec4 tpos(0.0f, 0.0f, 0.0f, 0.0f);

            bool boneInflued = false;
            int bCount = 0;
            // Influence
            for(auto& b: bwe){
                // e.g. b.first -> joint1
                if(elemSize){
                    float bIndex = (float)m_JointCorrespondingIndex[b.first];
                    prim->verts.attr<float>("jointIndice_" + std::to_string(bCount)).push_back(bIndex);
                    prim->verts.attr<float>("jointWeight_" + std::to_string(bCount)).push_back(b.second);
                }
                //std::cout << "FBX: Vert " << i << " name " << b.first << " bIndex " << bIndex << std::endl;
                boneInflued = true;
                auto& tr = m_BoneTransforms[b.first];
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

            // TODO (Bone Influence) Skeleton + Transform
            //  If remove follow `if`, we will get full transform animation, but the skel animation is gone
            if(! boneInflued) {
                tpos = pathTrans * glm::vec4(pos.x, pos.y, pos.z, 1.0f);
            }

            glm::vec3 fpos = glm::vec3(tpos.x/tpos.w, tpos.y/tpos.w, tpos.z/tpos.w);

            ver.emplace_back(fpos.x * gscale, fpos.y * gscale, fpos.z * gscale);
            posb.emplace_back(0.0f, 0.0f, 0.0f);
            uvs.emplace_back(uvw.x, uvw.y, uvw.z);
            uv.emplace_back(uvw.x, uvw.y, uvw.z);
            norm.emplace_back(nor.x, nor.y, nor.z);
            clr0.emplace_back(vco.r, vco.g, vco.b);
        }

        if(isTris) {
            for (unsigned int i = 0; i < m_IndicesTris.size(); i += 3) {
                zeno::vec3i incs(m_IndicesTris[i], m_IndicesTris[i + 1], m_IndicesTris[i + 2]);
                trisInd.push_back(incs);
            }
            uvs.clear();
        }else{
            for (unsigned int i = 0; i < m_IndicesLoops.size(); i ++) {
                loops.emplace_back(m_IndicesLoops[i]);
            }
            for (unsigned int i = 0; i < m_IndicesPolys.size(); i ++) {
                polys.emplace_back(m_IndicesPolys[i]);
            }
            uv.clear();
        }

        // Processing UV data
        if(isTris) {
            auto &uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
            auto &uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
            auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
            for (unsigned int i = 0; i < trisInd.size(); i++) {
                unsigned int _i1 = trisInd[i][0];
                unsigned int _i2 = trisInd[i][1];
                unsigned int _i3 = trisInd[i][2];
                uv0[i] = zeno::vec3f(m_Vertices[_i1].texCoord[0], m_Vertices[_i1].texCoord[1], 0);
                uv1[i] = zeno::vec3f(m_Vertices[_i2].texCoord[0], m_Vertices[_i2].texCoord[1], 0);
                uv2[i] = zeno::vec3f(m_Vertices[_i3].texCoord[0], m_Vertices[_i3].texCoord[1], 0);
            }
        }else{
            // Crash
            //if(prim->uvs.size()) {
            //    prim->loops.add_attr<int>("uvs");
            //    for (auto i = 0; i < prim->loops.size(); i++) {
            //        prim->loops.attr<int>("uvs")[i] = /*prim->loops[i]*/ i;
            //    }
            //}
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
        auto unit = get_param<std::string>("unit");
        auto interAnimData = get_param<std::string>("interAnimData");
        auto writeData = get_param<bool>("writeData");
        auto printAnimData = get_param<bool>("printAnimData");
        auto evalBlendShape = get_param<bool>("evalBlendShape");

        unit == "FROM_MAYA" ? evalOption.globalScale = 0.01f : evalOption.globalScale = 1.0f;
        interAnimData == "TRUE" ? evalOption.interAnimData = true : evalOption.interAnimData = false;
        if(writeData)
            evalOption.writeData = true;
        if(printAnimData)
            evalOption.printAnimData = true;
        if(evalBlendShape)
            evalOption.evalBlendShape = true;

        auto nodeTree = evalOption.interAnimData ? fbxData->nodeTree : get_input<NodeTree>("nodetree");
        auto boneTree = evalOption.interAnimData ? fbxData->boneTree : get_input<BoneTree>("bonetree");
        auto animInfo = evalOption.interAnimData ? fbxData->animInfo : get_input<AnimInfo>("animinfo");

        if(nodeTree == nullptr || boneTree == nullptr || animInfo == nullptr){
            zeno::log_error("FBX: Empty NodeTree, BoneTree or AnimInfo");
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto transDict = std::make_shared<zeno::DictObject>();
        auto quatDict = std::make_shared<zeno::DictObject>();
        auto scaleDict = std::make_shared<zeno::DictObject>();
        auto iCamera = std::make_shared<ICamera>();
        auto iLight = std::make_shared<ILight>();
        auto matName = std::make_shared<zeno::StringObject>();
        auto pathName = std::make_shared<zeno::StringObject>();
        auto outMeshName = std::make_shared<zeno::StringObject>();
        auto isVisibility = std::make_shared<zeno::NumericObject>();

        auto path = fbxData->iPathName.value;
        if(! fbxData->iVisibility.lut.empty()){
            if(fbxData->iVisibility.lut.find(path) != fbxData->iVisibility.lut.end()){
                auto visibilityData = fbxData->iVisibility.lut[path];
                auto listVisData = zeno::safe_dynamic_cast<zeno::ListObject>(visibilityData);
                bool is_visibility;
                auto length_m1 = listVisData->arr.size() - 1;
                if(frameid <=0){
                    is_visibility = zeno::safe_dynamic_cast<zeno::NumericObject>(listVisData->arr[0])->get<int>();
                }else if(frameid >= (length_m1)){
                    is_visibility = zeno::safe_dynamic_cast<zeno::NumericObject>(listVisData->arr[length_m1])->get<int>();
                }else{
                    is_visibility = zeno::safe_dynamic_cast<zeno::NumericObject>(listVisData->arr[frameid])->get<int>();
                }
                isVisibility->set(is_visibility);
            }else{
                std::cout << " Visibility Not Found " << path << "\n";
                isVisibility->set(1);
            }
        }else{
            isVisibility->set(1);
        }

        //ED_COUT << " Visibility " << isVisibility->get<int>() << " Path " << fbxData->iPathName.value << "\n";

//        TIMER_START(EvalAnim)
        EvalAnim anim;
        anim.m_evalOption = evalOption;
        anim.initAnim(nodeTree, boneTree, fbxData, animInfo);
//        TIMER_END(EvalAnim)

//        TIMER_START(UpdateAnim)
        anim.updateAnimation(frameid, prim);
//        TIMER_END(UpdateAnim)
        anim.updateCameraAndLight(fbxData, iCamera, iLight);
        anim.decomposeAnimation(transDict, quatDict, scaleDict);

        auto bsPrims = std::make_shared<zeno::ListObject>();
        auto bsPrimsOrigin = std::make_shared<zeno::ListObject>();
        auto meshName = fbxData->iMeshName.value_relName;

        matName->set(fbxData->iMeshName.value_matName);
        pathName->set(fbxData->iPathName.value);
        outMeshName->set(meshName);

        auto bsValue = fbxData->iBlendSData.value;
        float gScale = evalOption.globalScale;

        // XXX When the input data is single partten
//        TIMER_START(BlendShapeCreate)
        for(auto & [bsName, nameOfBlendShapes]: bsValue){
            glm::mat4 pathTrans(1.0);
            auto pathName = fbxData->iPathName.value_oriPath;
            int tranType = -1;
            anim.getPathTrans(pathName, pathTrans, tranType);

            //std::cout << "BlendShape Key " << bsName << "\n";
            for(int i=0; i< nameOfBlendShapes.size(); i++){
                auto bsprim = std::make_shared<zeno::PrimitiveObject>();
                auto &verAttr = bsprim->verts;
                auto &indAttr = bsprim->tris;
                auto &nrmAttr = bsprim->verts.add_attr<zeno::vec3f>("nrm");
                auto &dnrmAttr = bsprim->verts.add_attr<zeno::vec3f>("dnrm");
                auto &dposAttr = bsprim->verts.add_attr<zeno::vec3f>("dpos");
                auto& blendShapeData = nameOfBlendShapes[i];
                //std::cout << " BlendShape " << i << "\n";

                auto pathTransScale = glm::vec3(glm::length(glm::vec3(pathTrans[0])),
                                                glm::length(glm::vec3(pathTrans[1])),
                                                glm::length(glm::vec3(pathTrans[2])));

                for(unsigned int j=0; j<blendShapeData.size(); j++){ // Mesh Vert
                    auto& vdata = blendShapeData[j];
                    auto& pos = vdata.position;
                    auto& nrm = vdata.normal;
                    auto& dpos = vdata.deltaPosition;
                    auto& dnrm = vdata.deltaNormal;

                    // TODO BlendShape Normal Compute
                    glm::vec4 adpos = glm::vec4(pathTransScale, 1.0f) * glm::vec4(dpos.x, dpos.y, dpos.z, 1.0f);
                    dpos = aiVector3D(adpos.x/adpos.w, adpos.y/adpos.w, adpos.z/adpos.w);

                    verAttr.emplace_back(pos.x * gScale, pos.y * gScale, pos.z * gScale);
                    nrmAttr.emplace_back(nrm.x, nrm.y, nrm.z);
                    dposAttr.emplace_back(adpos.x * gScale, adpos.y * gScale, adpos.z * gScale);
                    dnrmAttr.emplace_back(dnrm.x, dnrm.y, dnrm.z);
                }

                bsPrimsOrigin->arr.emplace_back(bsprim);
            }
        }
//        TIMER_END(BlendShapeCreate)

//        TIMER_START(BlendShapeEval)
        // TODO FBXData Write BlendShape
        if(bsValue.find(meshName) != bsValue.end()){
            if(fbxData->iKeyMorph.value.find(meshName) != fbxData->iKeyMorph.value.end()){

                auto blendShapeData = bsValue[meshName];
                auto keyMorphs = fbxData->iKeyMorph.value[meshName];

                unsigned int kstart = 0;
                unsigned int kend;
                bool found = false;
                for(unsigned int i=0; i<keyMorphs.size()-1; i++){
                    //std::cout << " i " << i << " " << keyMorphs[i+1].m_Time << " current " << anim.m_CurrentFrame << "\n";
                    if(anim.m_CurrentFrame <= keyMorphs[i+1].m_Time){
                        kstart = i;
                        found = true;
                        break;
                    }
                }
                kend = kstart+1;
                if(! found){
                    kstart = keyMorphs.size()-2;
                    kend = kstart+1;
                }

                auto& kdstart = keyMorphs[kstart];
                auto& kdend = keyMorphs[kend];
                float factor = (anim.m_CurrentFrame - kdstart.m_Time) / (kdend.m_Time - kdstart.m_Time);
                std::cout << "Eval BlendShape " << meshName << " Factor " << factor << "\n";
                std::cout << "Eval BlendShape Index " << kstart << " " << kend << "\n";
                std::cout << "Eval BlendShape Time " << kdstart.m_Time << " " << kdend.m_Time << " " << anim.m_CurrentFrame << "\n";

                if(factor < 0.0){
                    factor = 0.0;
                }
                if(factor > 1.0){
                    factor = 1.0;
                }

                for(unsigned int i=0; i<blendShapeData.size(); i++){ // Anim Mesh & Same as BlendShape WeightsAndValues
                    auto bsprim = std::make_shared<zeno::PrimitiveObject>();
                    auto &verAttr = bsprim->verts;
                    auto &nrmAttr = bsprim->verts.add_attr<zeno::vec3f>("nrm");
                    auto &norb = bsprim->verts.add_attr<zeno::vec3f>("nrmb");
                    auto &posb = bsprim->verts.add_attr<zeno::vec3f>("posb");
                    auto &bsw = bsprim->verts.add_attr<float>("bsw");
                    double w = kdstart.m_Weights[i] * (1.0f - factor) + kdend.m_Weights[i] * factor;
                    auto& bsdata = blendShapeData[i];
                    for(unsigned int j=0; j<bsdata.size(); j++){
                        auto& pos = bsdata[j].position;
                        auto& nrm = bsdata[j].normal;
                        auto& dpos = bsdata[j].deltaPosition;
                        auto& dnor = bsdata[j].deltaNormal;

                        verAttr.emplace_back(pos.x * gScale, pos.y * gScale, pos.z * gScale);
                        nrmAttr.emplace_back(nrm.x, nrm.y, nrm.z);
                        posb.emplace_back(dpos.x * gScale, dpos.y * gScale, dpos.z * gScale);
                        norb.emplace_back(dnor.x, dnor.y, dnor.z);
                        bsw.emplace_back((float)w);
                    }

                    if(evalBlendShape){
                        for(unsigned int j=0; j<bsdata.size(); j++) {
                            //std::cout << " " << j << " " << posb[j][0] << ","<<posb[j][1] <<","<<posb[j][2] << " - " << w << "\n";
                            prim->verts[j] = prim->verts[j] + posb[j] * w;
                        }
                    }

                    bsPrims->arr.emplace_back(bsprim);
                }
            }else{
                std::cout << "BlendShape NotFound MorphKey " << meshName << "\n";
            }
        }
//        TIMER_END(BlendShapeEval)

        //zeno::log_info("Frame {} Prims Num {} Mesh Name {}", anim.m_CurrentFrame, bsPrims->arr.size(), meshName);
        auto data2write = std::make_shared<SFBXData>();
        *data2write = anim.m_FbxData;

        set_output("prim", std::move(prim));
        set_output("bsPrims", std::move(bsPrims));
        set_output("bsPrimsOrigin", std::move(bsPrimsOrigin));
        set_output("camera", std::move(iCamera));
        set_output("light", std::move(iLight));
        set_output("matName", std::move(matName));
        set_output("pathName", std::move(pathName));
        set_output("meshName", std::move(outMeshName));
        set_output("transDict", std::move(transDict));
        set_output("quatDict", std::move(quatDict));
        set_output("scaleDict", std::move(scaleDict));
        set_output("writeData", std::move(data2write));
        set_output("visibility", std::move(isVisibility));
    }
};
ZENDEFNODE(EvalFBXAnim,
           {       /* inputs: */
               {
                   {"frameid"},
                   //{"float", "fps", "24.0"},
                   "data", "animinfo", "nodetree", "bonetree",
               },  /* outputs: */
               {
                   "prim",
                   "camera", "light", "matName", "meshName", "pathName", "bsPrimsOrigin",
                   {"list", "bsPrims", ""},
                   "transDict", "quatDict", "scaleDict",
                   "writeData", "visibility"
               },  /* params: */
               {
                   {"enum FROM_MAYA DEFAULT", "unit", "FROM_MAYA"},
                   {"enum TRUE FALSE", "interAnimData", "TRUE"},
                   {"bool", "writeData", "false"},
                   {"bool", "printAnimData", "false"},
                   {"bool", "evalBlendShape", "true"},
               },  /* category: */
               {
                   "FBX",
               }
           });

}
