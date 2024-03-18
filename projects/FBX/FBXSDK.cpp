#include <iostream>
#include <memory>
#include <sstream>
#include <stack>

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>

#ifdef ZENO_FBXSDK
#include <fbxsdk.h>
#include "zeno/utils/log.h"
#include <zeno/types/UserData.h>
#include "zeno/types/PrimitiveObject.h"
#include "zeno/utils/scope_exit.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace FBX{
    void GetChildNodePathRecursive(FbxNode* node, std::string& path) {
        if (node->GetParent()) {
            GetChildNodePathRecursive(node->GetParent(), path);
            path += "/";
        }
        path += node->GetName();
    }


    std::string GetLastComponent(const std::string& str, char delimiter) {
        std::stringstream ss(str);
        std::string item;
        std::vector<std::string> components;

        while (std::getline(ss, item, delimiter)) {
            components.push_back(item);
        }

        if (!components.empty()) {
            return components.back();
        }

        return "";
    }

    std::string GetDifference(const std::string& str1, const std::string& str2) {
        if (str1.size() >= str2.size() && str1.compare(0, str2.size(), str2) == 0) {
            // Extract the difference from str1
            return str1.substr(str2.size());
        }

        // No common prefix or str2 is longer than str1
        return "";
    }

    bool CheckInherit(std::vector<std::string>& mayInherits, std::string check, std::string& find_path){
        // Check if the substring is found in the string
        for(auto& may: mayInherits) {
            if (may != check && check.find(may) != std::string::npos) {

                auto diff = GetDifference(check, may);
                char delimiter = '/';
                std::string lastComponent = GetLastComponent(check, delimiter);
                std::string pathedLast = "/" + lastComponent;
                //std::cout << "  Check: " << check << "\n";
                //std::cout << "  Maybe: " << may << "\n";
                //std::cout << "  Diff: " << diff << " last " << pathedLast << "\n";

                if(pathedLast == diff) {
                    find_path = may;
                    return true;
                }
            }
        }
        return false;
    }

    void EvalInheritVisibilityCurve(int startFrame,
                                    int endFrame,
                                    FbxAnimCurve* visCurve,
                                    std::shared_ptr<zeno::DictObject> dictObj,
                                    std::string full_path,
                                    FbxTime::EMode eFrame){
        // Visibility animation curve found, process the keyframes
        // ...
        //std::cout << " Visibility Curve" << "\n";

        // Eval Visibility
        //FbxTime::EMode timeMode = FbxTime::eFrames24;
        FbxTime::EMode timeMode = eFrame;
        auto vis = std::make_shared<zeno::ListObject>();
        for(int frameNumber=startFrame; frameNumber<=endFrame; frameNumber++) {
            double frameRate = FbxTime::GetFrameRate(timeMode);
            double frameTime = static_cast<double>(frameNumber) / frameRate;
            FbxTime time;
            time.SetSecondDouble(frameTime);
            bool visibility = visCurve->Evaluate(time);
            auto no_vis = std::make_shared<zeno::NumericObject>();
            no_vis->set(visibility);
            //std::cout << "visibility: " << visibility << " time: " << frameNumber << "\n";
            vis->arr.push_back(no_vis);
        }
        dictObj->lut[full_path] = vis;

        // Get Visibility
        int numKeys = visCurve->KeyGetCount();
        for (int keyIndex = 0; keyIndex < numKeys; keyIndex++) {
            FbxTime keyTime = visCurve->KeyGetTime(keyIndex);

            FbxLongLong frameNumber = keyTime.GetFrameCount();
            FbxAnimCurveKey key = visCurve->KeyGet(keyIndex);

            bool visibility = key.GetValue();  // Retrieve visibility value (true/false)
            //std::cout << "  visibility: " << visibility << " time: " << frameNumber << "\n";
            // Process visibility keyframe at keyTime
            // ...
        }
    }
}

struct FBXSDKVisibility : zeno::INode {

    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto inheritVisibility = get_input2<bool>("inherit");
        auto vis_dict = std::make_shared<zeno::DictObject>();
        auto inherit_dict = std::make_shared<zeno::DictObject>();
        auto fps = get_param<std::string>("fps");
        FbxTime::EMode eFrame = FbxTime::eFrames30;
        fps == "e30" ? eFrame = FbxTime::eFrames30 : eFrame = FbxTime::eFrames24;
        std::cout << "EFrame " << eFrame << " FPS " << fps << "\n";
        FbxManager* manager = FbxManager::Create();
        FbxIOSettings* ios = FbxIOSettings::Create(manager, IOSROOT);
        manager->SetIOSettings(ios);

        FbxImporter* importer = FbxImporter::Create(manager, "");
        importer->Initialize(path.c_str(), -1, manager->GetIOSettings());
        FbxScene* scene = FbxScene::Create(manager, "sceneName");

        FbxTime::EMode GlobalTimeMode = scene->GetGlobalSettings().GetTimeMode(); // Get the time mode of the scene

        importer->Import(scene);
        importer->Destroy();

        int startFrame = 0;
        int endFrame = 0;
        int numAnimStacks = scene->GetSrcObjectCount<FbxAnimStack>();
        for (int stackIndex = 0; stackIndex < numAnimStacks; stackIndex++) {
            FbxAnimStack* animStack = scene->GetSrcObject<FbxAnimStack>(stackIndex);
            // Process animStack
            FbxTimeSpan timeSpan = animStack->GetLocalTimeSpan();
            FbxTime start = timeSpan.GetStart();
            FbxTime end = timeSpan.GetStop();

            //startFrame = start.GetFrameCount();
            endFrame = end.GetFrameCount();
            //std::cout << " Anim Stack: " << stackIndex << " " << animStack->GetName() << "\n";
            //std::cout << "  Start: " << startFrame << " End " << endFrame << "\n";
        }

        // We will convert to 24 FPS, so if the frame is ended by 30 FPS, we need handle it.
        auto length = endFrame - startFrame;
        if (GlobalTimeMode == FbxTime::eFrames30) {
            if(eFrame == FbxTime::eFrames24) {
                endFrame = (int)((float)length / 30.0 * 24.0);
            }
            std::cout << "FBX Frame 30, End frame " << endFrame << "\n";
        } else if (GlobalTimeMode == FbxTime::eFrames24) {
            std::cout << "FBX Frame 24\n";
        } else {
            std::cout << "FBX Frame else " << GlobalTimeMode << "\n";
        }

        FbxNode* rootNode = scene->GetRootNode();
        if (rootNode) {
            //std::cout << " Root Node: " << rootNode->GetName() << "\n";
            std::vector<std::string> paths{};
            std::vector<std::string> checkInherit{};
            std::stack<FbxNode*> stack;
            stack.push(rootNode);

            while(!stack.empty()){
                FbxNode* node = stack.top();
                stack.pop();

                std::string full_path = "/";
                FBX::GetChildNodePathRecursive(node, full_path);
                //std::cout << " Child: " << node->GetName() << " Full: " << full_path << "\n";

                auto it = std::find(paths.begin(), paths.end(), full_path);
                if(it != paths.end()){
                    std::cout << " Element found at index: " << std::distance(paths.begin(), it) << std::endl;
                }else{
                    paths.emplace_back(full_path);
                }

                int numAnimLayers = scene->GetSrcObjectCount<FbxAnimLayer>();

                // Only handle One AnimLayer for now
                for (int layerIndex = 0; layerIndex < numAnimLayers; layerIndex++) {
                    FbxAnimLayer* animLayer = scene->GetSrcObject<FbxAnimLayer>(layerIndex);

                    // Channel
                    FbxAnimCurve* visCurve = node->Visibility.GetCurve(animLayer);
                    bool visNode = node->Visibility.Get();
                    //std::cout << "  visibility node: " << visNode << "\n";

                    // AttributeEditor (Anim Curve Attributes)
                    FbxProperty showProperty = node->FindProperty("Show");
                    FbxBool showValue = true;
                    if (showProperty.IsValid()) {
                        showValue = showProperty.Get<FbxBool>();
                        //std::cout << " Show: " << showValue << "\n";
                    }

                    if(visCurve){
                        //std::cout << " Node VisCurve " << full_path << "\n";
                        checkInherit.emplace_back(full_path);
                        FBX::EvalInheritVisibilityCurve(startFrame, endFrame, visCurve, inherit_dict, full_path, eFrame);
                    }

                    // 1. Show Property, The Highest priority
                    if(! showValue){
                        auto vis = std::make_shared<zeno::ListObject>();
                        for(int frameNumber=startFrame; frameNumber<=endFrame; frameNumber++) {
                            auto no_vis = std::make_shared<zeno::NumericObject>();
                            no_vis->set(showValue);
                            vis->arr.push_back(no_vis);
                        }
                        vis_dict->lut[full_path] = vis;
                    }
                    // 2. Visibility Key Frame
                    else if (visCurve)
                    {
                        FBX::EvalInheritVisibilityCurve(startFrame, endFrame, visCurve, vis_dict, full_path, eFrame);
                    }
                    // 3. Node Visibility
                    else
                    {
                        auto vis = std::make_shared<zeno::ListObject>();
                        for(int frameNumber=startFrame; frameNumber<=endFrame; frameNumber++) {
                            auto no_vis = std::make_shared<zeno::NumericObject>();
                            no_vis->set(visNode);
                            vis->arr.push_back(no_vis);
                        }
                        vis_dict->lut[full_path] = vis;
                    }
                    break;
                }

                int childCount = node->GetChildCount();
                for (int i = childCount - 1; i >= 0; --i) {
                    FbxNode* child = node->GetChild(i);
                    stack.push(child);
                }
            }

            if(inheritVisibility) {
                // Check Worst-case scenario n^2 algorithm complexity
                for (auto &path : paths) {
                    std::string find_path{};
                    bool inherit = FBX::CheckInherit(checkInherit, path, find_path);
                    //std::cout << " Check " << path << " Inherit " << inherit << "\n";
                    if (inherit) {
                        //std::cout << "  Find: " << find_path << "\n";
                        vis_dict->lut[path] = inherit_dict->lut[find_path];
                    }
                }
            }
        }

        manager->Destroy();

        set_output("visibility", std::move(vis_dict));
    }
};

ZENDEFNODE(FBXSDKVisibility,
           {       /* inputs: */
            {
                {"readpath", "path"},
                {"bool", "inherit", "false"},
            },  /* outputs: */
            {
                {"DictObject", "visibility", ""},
            },  /* params: */
            {
                {"enum e24 e30", "fps", "e24"},
            },  /* category: */
            {
                "FBX",
            }
           });

namespace zeno {
/**
* Return a string-based representation based on the attribute type.
*/
FbxString GetAttributeTypeName(FbxNodeAttribute::EType type) {
    switch(type) {
        case FbxNodeAttribute::eUnknown: return "unidentified";
        case FbxNodeAttribute::eNull: return "null";
        case FbxNodeAttribute::eMarker: return "marker";
        case FbxNodeAttribute::eSkeleton: return "skeleton";
        case FbxNodeAttribute::eMesh: return "mesh";
        case FbxNodeAttribute::eNurbs: return "nurbs";
        case FbxNodeAttribute::ePatch: return "patch";
        case FbxNodeAttribute::eCamera: return "camera";
        case FbxNodeAttribute::eCameraStereo: return "stereo";
        case FbxNodeAttribute::eCameraSwitcher: return "camera switcher";
        case FbxNodeAttribute::eLight: return "light";
        case FbxNodeAttribute::eOpticalReference: return "optical reference";
        case FbxNodeAttribute::eOpticalMarker: return "marker";
        case FbxNodeAttribute::eNurbsCurve: return "nurbs curve";
        case FbxNodeAttribute::eTrimNurbsSurface: return "trim nurbs surface";
        case FbxNodeAttribute::eBoundary: return "boundary";
        case FbxNodeAttribute::eNurbsSurface: return "nurbs surface";
        case FbxNodeAttribute::eShape: return "shape";
        case FbxNodeAttribute::eLODGroup: return "lodgroup";
        case FbxNodeAttribute::eSubDiv: return "subdiv";
        default: return "unknown";
    }
}

/**
* Print an attribute.
*/
void PrintAttribute(FbxNodeAttribute* pAttribute) {
    if(!pAttribute) return;

    FbxString typeName = GetAttributeTypeName(pAttribute->GetAttributeType());
    FbxString attrName = pAttribute->GetName();
    // Note: to retrieve the character array of a FbxString, use its Buffer() method.
    printf("<attribute type='%s' name='%s'/>\n", typeName.Buffer(), attrName.Buffer());
}

//void PrintNode(FbxNode* pNode) {
//    const char* nodeName = pNode->GetName();
//    FbxDouble3 translation = pNode->LclTranslation.Get();
//    FbxDouble3 rotation = pNode->LclRotation.Get();
//    FbxDouble3 scaling = pNode->LclScaling.Get();
//
//    // Print the contents of the node.
//    printf("<node name='%s' translation='(%f, %f, %f)' rotation='(%f, %f, %f)' scaling='(%f, %f, %f)'>\n",
//           nodeName,
//           translation[0], translation[1], translation[2],
//           rotation[0], rotation[1], rotation[2],
//           scaling[0], scaling[1], scaling[2]
//    );
//
//    // Print the node's attributes.
//    for(int i = 0; i < pNode->GetNodeAttributeCount(); i++)
//        PrintAttribute(pNode->GetNodeAttributeByIndex(i));
//
//    // Recursively print the children.
//    for(int j = 0; j < pNode->GetChildCount(); j++)
//        PrintNode(pNode->GetChild(j));
//
//    printf("</node>\n");
//}
bool GetMesh(FbxNode* pNode, std::shared_ptr<PrimitiveObject> prim) {
    FbxMesh* pMesh = pNode->GetMesh();
    if (!pMesh) return false;

    FbxAMatrix bindMatrix = pNode->EvaluateGlobalTransform();
    auto s = bindMatrix.GetS();
    auto t = bindMatrix.GetT();
//    zeno::log_info("s {} {} {}", s[0], s[1], s[2]);
//    zeno::log_info("t {} {} {}", t[0], t[1], t[2]);

    int numVertices = pMesh->GetControlPointsCount();
    FbxVector4* vertices = pMesh->GetControlPoints();
    prim->verts.resize(numVertices);

    for (int i = 0; i < numVertices; ++i) {
        auto pos = bindMatrix.MultT( FbxVector4(vertices[i][0], vertices[i][1], vertices[i][2], 1.0));
        prim->verts[i] = vec3f(pos[0], pos[1], pos[2]);
    }
    int numPolygons = pMesh->GetPolygonCount();
    prim->polys.resize(numPolygons);
    std::vector<int> loops;
    loops.reserve(numPolygons * 4);
    int count = 0;
    for (int i = 0; i < numPolygons; ++i) {
        int numVertices = pMesh->GetPolygonSize(i);
        for (int j = 0; j < numVertices; ++j) {
            int vertexIndex = pMesh->GetPolygonVertex(i, j);
            loops.push_back(vertexIndex);
        }
        prim->polys[i] = {count, numVertices};
        count += numVertices;
    }
    loops.shrink_to_fit();
    prim->loops.values = loops;
//    zeno::log_info("pMesh->GetDeformerCount(FbxDeformer::eSkin) {}", pMesh->GetDeformerCount(FbxDeformer::eSkin));
    auto &ud = prim->userData();
    if (pMesh->GetDeformerCount(FbxDeformer::eSkin)) {
        auto &bi = prim->verts.add_attr<vec4i>("boneName");
        std::fill(bi.begin(), bi.end(), vec4i(-1, -1, -1, -1));
        auto &bw = prim->verts.add_attr<vec4f>("boneWeight");
        std::fill(bw.begin(), bw.end(), vec4f(-1.0, -1.0, -1.0, -1.0));

        FbxSkin* pSkin = (FbxSkin*)pMesh->GetDeformer(0, FbxDeformer::eSkin);
        std::vector<std::string> bone_names;

        // Iterate over each cluster (bone)
        for (int j = 0; j < pSkin->GetClusterCount(); ++j) {
            FbxCluster* pCluster = pSkin->GetCluster(j);

            // Get the link node (bone)
            FbxNode* pBoneNode = pCluster->GetLink();
            if (!pBoneNode) continue;

            // Get the bone weights
            int numIndices = pCluster->GetControlPointIndicesCount();
            int* indices = pCluster->GetControlPointIndices();
            double* weights = pCluster->GetControlPointWeights();

            bone_names.emplace_back(pBoneNode->GetName());
            for (int k = 0; k < numIndices; ++k) {
                for (auto l = 0; l < 4; l++) {
                    if (bi[indices[k]][l] == -1) {
                        bi[indices[k]][l] = j;
                        bw[indices[k]][l] = weights[k];
                        break;
                    }
                }
            }
        }
        ud.set2("boneName_count", int(bone_names.size()));
        for (auto i = 0; i < bone_names.size(); i++) {
            ud.set2(zeno::format("boneName_{}", i), bone_names[i]);
        }
    }
    return true;
}

struct NewFBXImportSkin : INode {
    virtual void apply() override {
        // Change the following filename to a suitable filename value.
        auto lFilename = get_input2<std::string>("path");

        // Initialize the SDK manager. This object handles all our memory management.
        FbxManager* lSdkManager = FbxManager::Create();

        // Create the IO settings object.
        FbxIOSettings *ios = FbxIOSettings::Create(lSdkManager, IOSROOT);
        lSdkManager->SetIOSettings(ios);

        // Create an importer using the SDK manager.
        FbxImporter* lImporter = FbxImporter::Create(lSdkManager,"");

        // Use the first argument as the filename for the importer.
        if(!lImporter->Initialize(lFilename.c_str(), -1, lSdkManager->GetIOSettings())) {
            printf("Call to FbxImporter::Initialize() failed.\n");
            printf("Error returned: %s\n\n", lImporter->GetStatus().GetErrorString());
            exit(-1);
        }
        int major, minor, revision;
        lImporter->GetFileVersion(major, minor, revision);

        // Create a new scene so that it can be populated by the imported file.
        FbxScene* lScene = FbxScene::Create(lSdkManager,"myScene");

        // Import the contents of the file into the scene.
        lImporter->Import(lScene);

        // The file is imported; so get rid of the importer.
        lImporter->Destroy();

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();
        auto &ud = prim->userData();
        ud.set2("version", vec3i(major, minor, revision));
        FbxNode* lRootNode = lScene->GetRootNode();
        if(lRootNode) {
            for(int i = 0; i < lRootNode->GetChildCount(); i++) {
                if (GetMesh(lRootNode->GetChild(i), prim)) {
                    break;
                }
            }
        }
        if (get_input2<bool>("ConvertUnits")) {
            for (auto & v: prim->verts) {
                v = v * 0.01;
            }
        }
        set_output("prim", prim);
        // Destroy the SDK manager and all the other objects it was handling.
        lSdkManager->Destroy();
    }
};

ZENDEFNODE(NewFBXImportSkin, {
    {
        {"readpath", "path"},
        {"bool", "ConvertUnits", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

struct NewFBXImportSkeleton : INode {
    virtual void apply() override {
        // Change the following filename to a suitable filename value.
        auto lFilename = get_input2<std::string>("path");

        // Initialize the SDK manager. This object handles all our memory management.
        FbxManager* lSdkManager = FbxManager::Create();

        // Create the IO settings object.
        FbxIOSettings *ios = FbxIOSettings::Create(lSdkManager, IOSROOT);
        lSdkManager->SetIOSettings(ios);

        // Create an importer using the SDK manager.
        FbxImporter* lImporter = FbxImporter::Create(lSdkManager,"");

        // Use the first argument as the filename for the importer.
        if(!lImporter->Initialize(lFilename.c_str(), -1, lSdkManager->GetIOSettings())) {
            printf("Call to FbxImporter::Initialize() failed.\n");
            printf("Error returned: %s\n\n", lImporter->GetStatus().GetErrorString());
            exit(-1);
        }
        int major, minor, revision;
        lImporter->GetFileVersion(major, minor, revision);

        // Create a new scene so that it can be populated by the imported file.
        FbxScene* lScene = FbxScene::Create(lSdkManager,"myScene");

        // Import the contents of the file into the scene.
        lImporter->Import(lScene);

        // The file is imported; so get rid of the importer.
        lImporter->Destroy();

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();
        auto &ud = prim->userData();
        ud.set2("version", vec3i(major, minor, revision));

        auto pose_count = lScene->GetPoseCount();
        bool found_bind_pose = false;
        for (auto i = 0; i < pose_count; i++) {
            auto pose = lScene->GetPose(i);
            if (pose == nullptr || !pose->IsBindPose()) {
                continue;
            }
            found_bind_pose = true;
        }
        if (found_bind_pose == false) {
            lSdkManager->CreateMissingBindPoses(lScene);
        }
        pose_count = lScene->GetPoseCount();
        for (auto i = 0; i < pose_count; i++) {
            auto pose = lScene->GetPose(i);
            if (pose == nullptr || !pose->IsBindPose()) {
                continue;
            }
            std::string name = pose->GetName();
            prim->verts.resize(pose->GetCount() - 1);
            std::vector<std::string> bone_names;
            auto &boneNames = prim->verts.add_attr<int>("boneName");
            auto &transform_r0 = prim->verts.add_attr<vec3f>("transform_r0");
            auto &transform_r1 = prim->verts.add_attr<vec3f>("transform_r1");
            auto &transform_r2 = prim->verts.add_attr<vec3f>("transform_r2");
            for (int j = 1; j < pose->GetCount(); ++j) {
                FbxMatrix transformMatrix = pose->GetMatrix(j);
                auto t = transformMatrix.GetRow(3);
                prim->verts[j - 1] = vec3f(t[0], t[1], t[2]);

                auto r0 = transformMatrix.GetRow(0);
                auto r1 = transformMatrix.GetRow(1);
                auto r2 = transformMatrix.GetRow(2);
                transform_r0[j - 1] = vec3f(r0[0], r0[1], r0[2]);
                transform_r1[j - 1] = vec3f(r1[0], r1[1], r1[2]);
                transform_r2[j - 1] = vec3f(r2[0], r2[1], r2[2]);

                bone_names.emplace_back(pose->GetNode(j)->GetName());
                boneNames[j - 1] = j - 1;
            }
            std::vector<int> bone_connects;
            for (int j = 1; j < pose->GetCount(); ++j) {
                auto parent_name = pose->GetNode(j)->GetParent()->GetName();
                auto index = std::find(bone_names.begin(), bone_names.end(), parent_name) - bone_names.begin();
                if (index < bone_names.size()) {
                    bone_connects.push_back(index);
                    bone_connects.push_back(j - 1);
                }
            }
            {
                prim->loops.values = bone_connects;
                prim->polys.resize(bone_connects.size() / 2);
                for (auto j = 0; j < bone_connects.size() / 2; j++) {
                    prim->polys[j] = {j * 2, 2};
                }
            }
            ud.set2("boneName_count", int(bone_names.size()));
            for (auto i = 0; i < bone_names.size(); i++) {
                ud.set2(zeno::format("boneName_{}", i), bone_names[i]);
            }
        }

        if (get_input2<bool>("ConvertUnits")) {
            for (auto & v: prim->verts) {
                v = v * 0.01;
            }
        }
        set_output("prim", prim);
        // Destroy the SDK manager and all the other objects it was handling.
        lSdkManager->Destroy();
    }
};

ZENDEFNODE(NewFBXImportSkeleton, {
    {
        {"readpath", "path"},
        {"bool", "ConvertUnits", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

struct NewFBXImportAnimation : INode {
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = std::lround(get_input2<float>("frameid"));
        } else {
            frameid = getGlobalState()->frameid;
        }
        float fps = get_input2<float>("fps");
        float t = float(frameid) / fps;
        FbxTime curTime;       // The time for each key in the animation curve(s)
        curTime.SetSecondDouble(t);   // Starting time

        // Change the following filename to a suitable filename value.
        auto lFilename = get_input2<std::string>("path");

        // Initialize the SDK manager. This object handles all our memory management.
        FbxManager* lSdkManager = FbxManager::Create();

        // Create the IO settings object.
        FbxIOSettings *ios = FbxIOSettings::Create(lSdkManager, IOSROOT);
        lSdkManager->SetIOSettings(ios);
        // Destroy the SDK manager and all the other objects it was handling.
        zeno::scope_exit sp([=]() { lSdkManager->Destroy(); });

        // Create an importer using the SDK manager.
        FbxImporter* lImporter = FbxImporter::Create(lSdkManager,"");

        // Use the first argument as the filename for the importer.
        if(!lImporter->Initialize(lFilename.c_str(), -1, lSdkManager->GetIOSettings())) {
            printf("Call to FbxImporter::Initialize() failed.\n");
            printf("Error returned: %s\n\n", lImporter->GetStatus().GetErrorString());
            exit(-1);
        }
        int major, minor, revision;
        lImporter->GetFileVersion(major, minor, revision);

        // Create a new scene so that it can be populated by the imported file.
        FbxScene* lScene = FbxScene::Create(lSdkManager,"myScene");

        // Import the contents of the file into the scene.
        lImporter->Import(lScene);

        // The file is imported; so get rid of the importer.
        lImporter->Destroy();

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();
        auto &ud = prim->userData();
        ud.set2("version", vec3i(major, minor, revision));

        FbxArray<FbxString*> animationStackNames;
        std::vector<std::string> clip_names;
        lScene->FillAnimStackNameArray(animationStackNames);
        for (auto i = 0; i < animationStackNames.GetCount(); i++) {
            clip_names.emplace_back(animationStackNames[i]->Buffer());
        }
        for (auto i = 0; i < clip_names.size(); i++) {
            ud.set2(format("avail_anim_clip_{}", i), clip_names[i]);
        }
        ud.set2("avail_anim_clip_count", int(clip_names.size()));

        auto clip_name = get_input2<std::string>("clipName");
        if (clip_name == "") {
            clip_name = lScene->ActiveAnimStackName.Get().Buffer();
        }

        int stack_index = int(std::find(clip_names.begin(), clip_names.end(), clip_name) - clip_names.begin());
        if (stack_index == clip_names.size()) {
            zeno::log_error("FBX: Can not find clip name");
        }
//        zeno::log_info("stack_index: {}", stack_index);


        FbxAnimStack* animStack = lScene->GetSrcObject<FbxAnimStack>(stack_index);
        ud.set2("clipinfo.name", std::string(animStack->GetName()));
//        zeno::log_info("animStack: {}", animStack->GetName());


        lScene->SetCurrentAnimationStack(animStack);

        FbxTime mStart, mStop;
        FbxTakeInfo* lCurrentTakeInfo = lScene->GetTakeInfo(*(animationStackNames[stack_index]));
        float src_fps = 0;
        if (lCurrentTakeInfo)  {
            mStart = lCurrentTakeInfo->mLocalTimeSpan.GetStart();
            mStop = lCurrentTakeInfo->mLocalTimeSpan.GetStop();
            int frameCount = lCurrentTakeInfo->mLocalTimeSpan.GetDuration().GetFrameCount();
            src_fps = frameCount / lCurrentTakeInfo->mLocalTimeSpan.GetDuration().GetSecondDouble();
        }
        else {
            // Take the time line value
            FbxTimeSpan lTimeLineTimeSpan;
            lScene->GetGlobalSettings().GetTimelineDefaultTimeSpan(lTimeLineTimeSpan);

            mStart = lTimeLineTimeSpan.GetStart();
            mStop  = lTimeLineTimeSpan.GetStop();
            int frameCount = lTimeLineTimeSpan.GetDuration().GetFrameCount();
            src_fps = frameCount / lTimeLineTimeSpan.GetDuration().GetSecondDouble();
        }

        ud.set2("clipinfo.source_range", vec2f(mStart.GetSecondDouble(), mStop.GetSecondDouble()));
        ud.set2("clipinfo.source_fps", src_fps);
        ud.set2("clipinfo.fps", fps);

        {
            auto node_count = lScene->GetNodeCount();
            prim->verts.resize(node_count);
            std::vector<std::string> bone_names;
            auto &boneNames = prim->verts.add_attr<int>("boneName");
            auto &transform_r0 = prim->verts.add_attr<vec3f>("transform_r0");
            auto &transform_r1 = prim->verts.add_attr<vec3f>("transform_r1");
            auto &transform_r2 = prim->verts.add_attr<vec3f>("transform_r2");
            for (int j = 0; j < node_count; ++j) {
                auto pNode = lScene->GetNode(j);
                FbxAMatrix lGlobalPosition = pNode->EvaluateGlobalTransform(curTime);
                FbxMatrix transformMatrix;
                memcpy(&transformMatrix, &lGlobalPosition, sizeof(FbxMatrix));
                auto t = transformMatrix.GetRow(3);
                prim->verts[j] = vec3f(t[0], t[1], t[2]);

                auto r0 = transformMatrix.GetRow(0);
                auto r1 = transformMatrix.GetRow(1);
                auto r2 = transformMatrix.GetRow(2);
                transform_r0[j] = vec3f(r0[0], r0[1], r0[2]);
                transform_r1[j] = vec3f(r1[0], r1[1], r1[2]);
                transform_r2[j] = vec3f(r2[0], r2[1], r2[2]);

                bone_names.emplace_back(pNode->GetName());
                boneNames[j] = j;
            }
            std::vector<int> bone_connects;
            for (int j = 0; j < node_count; ++j) {
                auto pNode = lScene->GetNode(j);
                if (pNode->GetParent()) {
                    auto parent_name = pNode->GetParent()->GetName();
                    auto index = std::find(bone_names.begin(), bone_names.end(), parent_name) - bone_names.begin();
                    if (index < bone_names.size()) {
                        bone_connects.push_back(index);
                        bone_connects.push_back(j);
                    }
                }
            }
            {
                prim->loops.values = bone_connects;
                prim->polys.resize(bone_connects.size() / 2);
                for (auto j = 0; j < bone_connects.size() / 2; j++) {
                    prim->polys[j] = {j * 2, 2};
                }
            }
            ud.set2("boneName_count", int(bone_names.size()));
            for (auto i = 0; i < bone_names.size(); i++) {
                ud.set2(zeno::format("boneName_{}", i), bone_names[i]);
            }
        }

        if (get_input2<bool>("ConvertUnits")) {
            for (auto & v: prim->verts) {
                v = v * 0.01;
            }
        }
        set_output("prim", prim);
    }
};

ZENDEFNODE(NewFBXImportAnimation, {
    {
        {"readpath", "path"},
        {"string", "clipName", ""},
        {"frameid"},
        {"float", "fps", "25"},
        {"bool", "ConvertUnits", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});
struct NewFBXBoneDeform : INode {
    std::vector<std::string> getBoneNames(PrimitiveObject *prim) {
        auto boneName_count = prim->userData().get2<int>("boneName_count");
        std::vector<std::string> boneNames;
        boneNames.reserve(boneName_count);
        for (auto i = 0; i < boneName_count; i++) {
            auto boneName = prim->userData().get2<std::string>(format("boneName_{}", i));
            boneNames.emplace_back(boneName);
        }
        return boneNames;
    }
    std::vector<int> getBoneMapping(std::vector<std::string> &old, std::vector<std::string> &_new) {
        std::vector<int> mapping;
        mapping.reserve(old.size());
        for (auto i = 0; i < old.size(); i++) {
            auto index = std::find(_new.begin(), _new.end(), old[i]) - _new.begin();
            if (index == _new.size()) {
                index = -1;
                zeno::log_error("connot find bone: {}, {}", i, old[i]);
            }
            mapping.push_back(index);
        }
        return mapping;
    }
    std::vector<glm::mat4> getBoneMatrix(PrimitiveObject *prim) {
        std::vector<glm::mat4> matrixs;
        auto &verts = prim->verts;
        auto &transform_r0 = prim->verts.add_attr<vec3f>("transform_r0");
        auto &transform_r1 = prim->verts.add_attr<vec3f>("transform_r1");
        auto &transform_r2 = prim->verts.add_attr<vec3f>("transform_r2");
        for (auto i = 0; i < prim->verts.size(); i++) {
            glm::mat4 matrix;
            matrix[0] = {transform_r0[i][0], transform_r0[i][1], transform_r0[i][2], 0};
            matrix[1] = {transform_r1[i][0], transform_r1[i][1], transform_r1[i][2], 0};
            matrix[2] = {transform_r2[i][0], transform_r2[i][1], transform_r2[i][2], 0};
            matrix[3] = {verts[i][0], verts[i][1], verts[i][2], 1};
            matrixs.push_back(matrix);
        }
        return matrixs;
    }
    std::vector<glm::mat4> getInvertedBoneMatrix(PrimitiveObject *prim) {
        std::vector<glm::mat4> inv_matrixs;
        auto matrixs = getBoneMatrix(prim);
        for (auto i = 0; i < matrixs.size(); i++) {
            auto m = matrixs[i];
            auto inv_m = glm::inverse(m);
            inv_matrixs.push_back(inv_m);
        }
        return inv_matrixs;
    }
    vec3f transform_pos(glm::mat4 &transform, vec3f pos) {
        auto p = transform * glm::vec4(pos[0], pos[1], pos[2], 1);
        return {p.x, p.y, p.z};
    }
    virtual void apply() override {
        auto geometryToDeform = get_input2<PrimitiveObject>("GeometryToDeform");
        auto geometryToDeformBoneNames = getBoneNames(geometryToDeform.get());
        auto restPointTransformsPrim = get_input2<PrimitiveObject>("RestPointTransforms");
        auto restPointTransformsBoneNames = getBoneNames(restPointTransformsPrim.get());
        auto restPointTransformsBoneMapping = getBoneMapping(geometryToDeformBoneNames, restPointTransformsBoneNames);
        auto restPointTransformsInv = getInvertedBoneMatrix(restPointTransformsPrim.get());
        auto deformPointTransformsPrim = get_input2<PrimitiveObject>("DeformPointTransforms");
        auto deformPointTransformsBoneNames = getBoneNames(deformPointTransformsPrim.get());
        auto deformPointTransformsBoneMapping = getBoneMapping(geometryToDeformBoneNames, deformPointTransformsBoneNames);
        auto deformPointTransforms = getBoneMatrix(deformPointTransformsPrim.get());

        std::vector<glm::mat4> matrixs;
        matrixs.reserve(geometryToDeformBoneNames.size());
        for (auto i = 0; i < geometryToDeformBoneNames.size(); i++) {
            auto res_inv_matrix = restPointTransformsInv[restPointTransformsBoneMapping[i]];
            auto deform_matrix = deformPointTransforms[deformPointTransformsBoneMapping[i]];
            auto matrix = deform_matrix * res_inv_matrix;
            matrixs.push_back(matrix);
        }

        auto prim = std::dynamic_pointer_cast<PrimitiveObject>(geometryToDeform->clone());

        auto &bi = prim->verts.add_attr<vec4i>("boneName");
        auto &bw = prim->verts.add_attr<vec4f>("boneWeight");
        size_t vert_count = prim->verts.size();
        #pragma omp parallel for
        for (auto i = 0; i < vert_count; i++) {
            auto opos = prim->verts[i];
            vec3f pos = {};
            float w = 0;
            for (auto j = 0; j < 4; j++) {
                if (bi[i][j] < 0) {
                    continue;
                }
                auto matrix = matrixs[bi[i][j]];
                pos += transform_pos(matrix, opos) * bw[i][j];
                w += bw[i][j];
            }
            prim->verts[i] = pos / w;
        }

        set_output("prim", prim);
    }
};

ZENDEFNODE(NewFBXBoneDeform, {
    {
        "GeometryToDeform",
        "RestPointTransforms",
        "DeformPointTransforms",
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});
}
#endif