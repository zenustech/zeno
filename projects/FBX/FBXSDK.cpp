#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <numeric>

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>

#include "zeno/utils/log.h"
#include <zeno/types/UserData.h>
#include "zeno/types/PrimitiveObject.h"
#include "zeno/utils/scope_exit.h"
#include "zeno/funcs/PrimitiveUtils.h"
#include "zeno/utils/string.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include "DualQuaternion.h"

#ifdef ZENO_FBXSDK
#include <fbxsdk.h>

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
            vis->push_back(no_vis);
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
                            vis->push_back(no_vis);
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
                            vis->push_back(no_vis);
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
                {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
                {gParamType_Bool, "inherit", "false"},
            },  /* outputs: */
            {
                {gParamType_Dict,"visibility", ""},
            },  /* params: */
            {
                {"enum e24 e30", "fps", "e24"},
            },  /* category: */
            {
                "FBX",
            }
           });

namespace zeno {
struct FBXObject : PrimitiveObject {
    FbxManager* lSdkManager = nullptr;
    FbxScene* lScene = nullptr;
};

struct ReadFBXFile: INode {
    std::shared_ptr<FBXObject> _inner_fbx_object;
    std::string usedPath;
    virtual void apply() override {
        // Change the following filename to a suitable filename value.
        auto lFilename = get_input2<std::string>("path");
        if (lFilename == usedPath && _inner_fbx_object != nullptr) {
            set_output("fbx_object", _inner_fbx_object);
            return;
        }

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
        auto fbx_object = std::make_shared<FBXObject>();
        fbx_object->lSdkManager = lSdkManager;
        // Create a new scene so that it can be populated by the imported file.
        fbx_object->lScene = FbxScene::Create(lSdkManager,"myScene");

        // Import the contents of the file into the scene.
        lImporter->Import(fbx_object->lScene);

        // The file is imported; so get rid of the importer.
        lImporter->Destroy();
        fbx_object->userData().set2("version", vec3i(major, minor, revision));
        usedPath = lFilename;
        _inner_fbx_object = fbx_object;

        set_output("fbx_object", std::move(fbx_object));
    }
};

ZENDEFNODE(ReadFBXFile, {
    {
        {"readpath", "path"},
    },
    {
        "fbx_object",
    },
    {},
    {"FBX"},
});

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

template<typename T>
void getAttr(T* arr, std::string name, std::shared_ptr<PrimitiveObject> prim) {
    if (arr->GetMappingMode() == FbxLayerElement::EMappingMode::eByControlPoint) {
        zeno::log_info("{}, eByControlPoint", name);
        auto &attr = prim->verts.add_attr<vec3f>(name);
        for (auto i = 0; i < prim->verts.size(); i++) {
            int pIndex = i;
            if (arr->GetReferenceMode() == FbxLayerElement::EReferenceMode::eIndexToDirect) {
                pIndex = arr->GetIndexArray().GetAt(i);
            }
            auto x = arr->GetDirectArray().GetAt(pIndex)[0];
            auto y = arr->GetDirectArray().GetAt(pIndex)[1];
            auto z = arr->GetDirectArray().GetAt(pIndex)[2];
            attr[i] = vec3f(x, y, z);
        }
    }
    else if (arr->GetMappingMode() == FbxLayerElement::EMappingMode::eByPolygonVertex) {
        zeno::log_info("{}, eByPolygonVertex", name);
        auto &attr = prim->loops.add_attr<vec3f>(name);
        for (auto i = 0; i < prim->loops.size(); i++) {
            int pIndex = i;
            if (arr->GetReferenceMode() == FbxLayerElement::EReferenceMode::eIndexToDirect) {
                pIndex = arr->GetIndexArray().GetAt(i);
            }
            auto x = arr->GetDirectArray().GetAt(pIndex)[0];
            auto y = arr->GetDirectArray().GetAt(pIndex)[1];
            auto z = arr->GetDirectArray().GetAt(pIndex)[2];
            attr[i] = vec3f(x, y, z);
        }
    }
}

std::shared_ptr<PrimitiveObject> GetMesh(FbxNode* pNode) {
    FbxMesh* pMesh = pNode->GetMesh();
    if (!pMesh) return nullptr;
    std::string nodeName = pNode->GetName();
    auto prim = std::make_shared<PrimitiveObject>();
    prim->userData().set2("RootName", nodeName);

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

        FbxSkin* pSkin = (FbxSkin*)pMesh->GetDeformer(0, FbxDeformer::eSkin);
        std::vector<std::string> bone_names;
        // Iterate over each cluster (bone)
        std::vector<std::vector<std::pair<int, float>>> bone_weight(numVertices);
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
                bone_weight[indices[k]].emplace_back(j, weights[k]);
                    }
                }
        int maxnum_boneWeight = 0;
        for (auto i = 0; i < prim->verts.size(); i++) {
            maxnum_boneWeight = zeno::max(maxnum_boneWeight, bone_weight[i].size());
            }
        for (auto i = 0; i < maxnum_boneWeight; i++) {
            auto &bi = prim->verts.add_attr<int>(zeno::format("boneName_{}", i));
            std::fill(bi.begin(), bi.end(), -1);
            auto &bw = prim->verts.add_attr<float>(zeno::format("boneWeight_{}", i));
            std::fill(bw.begin(), bw.end(), -1.0);
        }
        for (auto i = 0; i < prim->verts.size(); i++) {
            for (auto j = 0; j < bone_weight[i].size(); j++) {
                prim->verts.attr<int>(format("boneName_{}", j))[i] = bone_weight[i][j].first;
                prim->verts.attr<float>(format("boneWeight_{}", j))[i] = bone_weight[i][j].second;
            }
        }
        ud.set2("maxnum_boneWeight", int(maxnum_boneWeight));
        ud.set2("boneName_count", int(bone_names.size()));
        for (auto i = 0; i < bone_names.size(); i++) {
            ud.set2(zeno::format("boneName_{}", i), bone_names[i]);
        }
    }
    if (pMesh->GetElementUVCount() > 0) {
        auto* arr = pMesh->GetElementUV(0);
        std::string name = "uv";
        if (arr->GetMappingMode() == FbxLayerElement::EMappingMode::eByControlPoint) {
            zeno::log_info("{}, eByControlPoint", name);
            auto &attr = prim->verts.add_attr<vec3f>(name);
            for (auto i = 0; i < prim->verts.size(); i++) {
                int pIndex = i;
                if (arr->GetReferenceMode() == FbxLayerElement::EReferenceMode::eIndexToDirect) {
                    pIndex = arr->GetIndexArray().GetAt(i);
                }
                auto x = arr->GetDirectArray().GetAt(pIndex)[0];
                auto y = arr->GetDirectArray().GetAt(pIndex)[1];
                attr[i] = vec3f(x, y, 0);
            }
        }
        else if (arr->GetMappingMode() == FbxLayerElement::EMappingMode::eByPolygonVertex) {
            zeno::log_info("{}, eByPolygonVertex", name);
            if (arr->GetReferenceMode() == FbxLayerElement::EReferenceMode::eDirect) {
                auto &uvs = prim->loops.add_attr<int>("uvs");
                std::iota(uvs.begin(), uvs.end(), 0);
                prim->uvs.resize(prim->loops.size());
            }
            else if (arr->GetReferenceMode() == FbxLayerElement::EReferenceMode::eIndexToDirect) {
                auto &uvs = prim->loops.add_attr<int>("uvs");
                for (auto i = 0; i < prim->loops.size(); i++) {
                    uvs[i] = arr->GetIndexArray().GetAt(i);
                }
                int count = arr->GetDirectArray().GetCount();
                prim->uvs.resize(count);
            }
            for (auto i = 0; i < prim->uvs.size(); i++) {
                auto x = arr->GetDirectArray().GetAt(i)[0];
                auto y = arr->GetDirectArray().GetAt(i)[1];
                prim->uvs[i] = vec2f(x, y);
            }
        }
    }
    if (pMesh->GetElementNormalCount() > 0) {
        getAttr(pMesh->GetElementNormal(0), "nrm", prim);
    }
    if (pMesh->GetElementTangentCount() > 0) {
        getAttr(pMesh->GetElementTangent(0), "tang", prim);
    }
    prim_set_abcpath(prim.get(), format("/ABC/{}", nodeName));
    return prim;
}

void TraverseNodesToGetNames(FbxNode* pNode, std::vector<std::string> &names) {
    if (!pNode) return;

    FbxMesh* mesh = pNode->GetMesh();
    if (mesh) {
        auto name = pNode->GetName();
        names.emplace_back(name);
    }

    for (int i = 0; i < pNode->GetChildCount(); i++) {
        TraverseNodesToGetNames(pNode->GetChild(i), names);
    }
}

void TraverseNodesToGetPrim(FbxNode* pNode, std::string target_name, std::shared_ptr<PrimitiveObject> &prim) {
    if (!pNode) return;

    FbxMesh* mesh = pNode->GetMesh();
    if (mesh) {
        auto name = pNode->GetName();
        if (target_name == name) {
            auto sub_prim = GetMesh(pNode);
            if (sub_prim) {
                prim = sub_prim;
        }
            return;
        }
    }

    for (int i = 0; i < pNode->GetChildCount(); i++) {
        TraverseNodesToGetPrim(pNode->GetChild(i), target_name, prim);
    }
}
void TraverseNodesToGetPrims(FbxNode* pNode, std::vector<std::shared_ptr<PrimitiveObject>> &prims) {
    if (!pNode) return;

    FbxMesh* mesh = pNode->GetMesh();
    if (mesh) {
        auto sub_prim = GetMesh(pNode);
        if (sub_prim) {
            prims.push_back(sub_prim);
        }
    }

    for (int i = 0; i < pNode->GetChildCount(); i++) {
        TraverseNodesToGetPrims(pNode->GetChild(i), prims);
    }
}

struct NewFBXImportSkin : INode {
    virtual void apply() override {
        auto fbx_object = get_input2<FBXObject>("fbx_object");
        auto lScene = fbx_object->lScene;

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();
        FbxNode* lRootNode = lScene->GetRootNode();
        std::vector<std::string> availableRootNames;
        if(lRootNode) {
            TraverseNodesToGetNames(lRootNode, availableRootNames);
            auto rootName = get_input2<std::string>("rootName");
            if (rootName.empty()) {
                std::vector<std::shared_ptr<PrimitiveObject>> prims;
                TraverseNodesToGetPrims(lRootNode, prims);

                std::map<std::string, int> nameMappingGlobal;

                std::vector<zeno::PrimitiveObject *> prims_ptr;
                int maxnum_boneWeight = 0;
                for (auto prim: prims) {
                    maxnum_boneWeight = zeno::max(maxnum_boneWeight, prim->userData().get2<int>("maxnum_boneWeight", 0));
                }
                for (auto prim: prims) {
                    prims_ptr.push_back(prim.get());
                    std::vector<int> nameMapping;
                    auto boneName_count = prim->userData().get2<int>("boneName_count", 0);
                    for (auto i = 0; i < boneName_count; i++) {
                        auto boneName = prim->userData().get2<std::string>(zeno::format("boneName_{}", i));
                        if (nameMappingGlobal.count(boneName) == 0) {
                            auto index = nameMappingGlobal.size();
                            nameMappingGlobal[boneName] = index;
            }
                        nameMapping.push_back(nameMappingGlobal[boneName]);
            }
                    prim->userData().del("boneName_count");
                    for (auto i = 0; i < boneName_count; i++) {
                        prim->userData().del(zeno::format("boneName_{}", i));
        }
                    for (auto j = 0; j < maxnum_boneWeight; j++) {
                        if (!prim->verts.attr_is<int>(format("boneName_{}", j))) {
                            auto &bi = prim->verts.add_attr<int>(zeno::format("boneName_{}", j));
                            std::fill(bi.begin(), bi.end(), -1);
                            auto &bw = prim->verts.add_attr<float>(zeno::format("boneWeight_{}", j));
                            std::fill(bw.begin(), bw.end(), -1.0);
                }
                        else {
                            auto &bi = prim->verts.attr<int>(zeno::format("boneName_{}", j));
                            for (auto &_bi: bi) {
                                if (_bi != -1) {
                                    _bi = nameMapping[_bi];
            }
        }
                        }
                    }
                }
                prim = primMergeWithFacesetMatid(prims_ptr);
                prim->userData().set2("boneName_count", int(nameMappingGlobal.size()));
                prim->userData().set2("maxnum_boneWeight", maxnum_boneWeight);
                for (auto [key, value]: nameMappingGlobal) {
                    prim->userData().set2(zeno::format("boneName_{}", value), key);
                }
            }
            else {
                TraverseNodesToGetPrim(lRootNode, rootName, prim);
            }
        }
        if (get_input2<bool>("ConvertUnits")) {
            for (auto & v: prim->verts) {
                v = v * 0.01;
            }
        }
        if (get_input2<bool>("CopyVectorsFromLoopsToVert")) {
            auto vectors_str = get_input2<std::string>("vectors");
            std::vector<std::string> vectors = zeno::split_str(vectors_str, ',');
            for (auto vector: vectors) {
                vector = zeno::trim_string(vector);
                if (vector.size() && prim->loops.attr_is<vec3f>(vector)) {
                    auto &nrm = prim->loops.attr<vec3f>(vector);
                    auto &vnrm = prim->verts.add_attr<vec3f>(vector);
                    for (auto i = 0; i < prim->loops.size(); i++) {
                        vnrm[prim->loops[i]] = nrm[i];
                    }
                }
            }
        }
        {
            auto &ud = prim->userData();
            ud.set2("AvailableRootName_count", int(availableRootNames.size()));
            for (int i = 0; i < availableRootNames.size(); i++) {
                ud.set2(format("AvailableRootName_{}", i), availableRootNames[i]);
            }
        }
        set_output("prim", prim);
    }
};

ZENDEFNODE(NewFBXImportSkin, {
    {
        {gParamType_Unknown, "fbx_object"},
        {gParamType_String, "rootName", ""},
        {gParamType_Bool, "ConvertUnits", "0"},
        {gParamType_String, "vectors", "nrm,"},
        {gParamType_Bool, "CopyVectorsFromLoopsToVert", "1"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"primitive"},
});

struct NewFBXImportSkeleton : INode {
    virtual void apply() override {
        auto fbx_object = get_input2<FBXObject>("fbx_object");
        auto lSdkManager = fbx_object->lSdkManager;
        auto lScene = fbx_object->lScene;

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();

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

        std::vector<std::string> bone_names;
        std::map<std::string, std::string> parent_mapping;
        std::vector<vec3f> poss;
        std::vector<vec3f> transform_r0;
        std::vector<vec3f> transform_r1;
        std::vector<vec3f> transform_r2;
        for (auto i = 0; i < pose_count; i++) {
            auto pose = lScene->GetPose(i);
            if (pose == nullptr || !pose->IsBindPose()) {
                continue;
            }
            for (int j = 1; j < pose->GetCount(); ++j) {
                std::string bone_name = pose->GetNode(j)->GetName();
                if (std::count(bone_names.begin(), bone_names.end(), bone_name)) {
                    continue;
                }

                FbxMatrix transformMatrix = pose->GetMatrix(j);
                auto t = transformMatrix.GetRow(3);
                poss.emplace_back(t[0], t[1], t[2]);

                auto r0 = transformMatrix.GetRow(0);
                auto r1 = transformMatrix.GetRow(1);
                auto r2 = transformMatrix.GetRow(2);
                transform_r0.emplace_back(r0[0], r0[1], r0[2]);
                transform_r1.emplace_back(r1[0], r1[1], r1[2]);
                transform_r2.emplace_back(r2[0], r2[1], r2[2]);

                bone_names.emplace_back(pose->GetNode(j)->GetName());
            }
            for (int j = 1; j < pose->GetCount(); ++j) {
                auto self_name = pose->GetNode(j)->GetName();
                auto parent = pose->GetNode(j)->GetParent();
                if (parent) {
                    auto parent_name = parent->GetName();
                    parent_mapping[self_name] = parent_name;
                }
            }
        }
            {
            prim->verts.resize(bone_names.size());
            prim->verts.values = poss;
            prim->verts.add_attr<vec3f>("transform_r0") = transform_r0;
            prim->verts.add_attr<vec3f>("transform_r1") = transform_r1;
            prim->verts.add_attr<vec3f>("transform_r2") = transform_r2;
            auto &boneNames = prim->verts.add_attr<int>("boneName");
            std::iota(boneNames.begin(), boneNames.end(), 0);

            std::vector<int> bone_connects;
            for (auto bone_name: bone_names) {
                if (parent_mapping.count(bone_name)) {
                    auto parent_name = parent_mapping[bone_name];
                    if (std::count(bone_names.begin(), bone_names.end(), parent_name)) {
                        auto self_index = std::find(bone_names.begin(), bone_names.end(), bone_name) - bone_names.begin();
                        auto parent_index = std::find(bone_names.begin(), bone_names.end(), parent_name) - bone_names.begin();
                        bone_connects.push_back(parent_index);
                        bone_connects.push_back(self_index);
                    }
                }
            }
                prim->loops.values = bone_connects;
                prim->polys.resize(bone_connects.size() / 2);
                for (auto j = 0; j < bone_connects.size() / 2; j++) {
                    prim->polys[j] = {j * 2, 2};
                }

            prim->userData().set2("boneName_count", int(bone_names.size()));
            for (auto i = 0; i < bone_names.size(); i++) {
                prim->userData().set2(zeno::format("boneName_{}", i), bone_names[i]);
            }
        }

        if (get_input2<bool>("ConvertUnits")) {
            for (auto & v: prim->verts) {
                v = v * 0.01;
            }
            auto &transform_r0 = prim->verts.add_attr<vec3f>("transform_r0");
            auto &transform_r1 = prim->verts.add_attr<vec3f>("transform_r1");
            auto &transform_r2 = prim->verts.add_attr<vec3f>("transform_r2");
            for (auto i = 0; i < prim->verts.size(); i++) {
                transform_r0[i] *= 0.01;
                transform_r1[i] *= 0.01;
                transform_r2[i] *= 0.01;
        }
        }
        set_output("prim", prim);
    }
};

ZENDEFNODE(NewFBXImportSkeleton, {
    {
        {gParamType_Unknown, "fbx_object"},
        {gParamType_Bool, "ConvertUnits", "0"},
    },
    {
        {gParamType_Primitive, "prim"},
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

        auto fbx_object = get_input2<FBXObject>("fbx_object");
        auto lSdkManager = fbx_object->lSdkManager;
        auto lScene = fbx_object->lScene;

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();
        auto &ud = prim->userData();

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
            zeno::log_info("FBX: Can not find default clip name, use first");
            stack_index = 0;
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
                zeno::log_info("boneName: {}", bone_names[i]);
            }
        }

        if (get_input2<bool>("ConvertUnits")) {
            for (auto & v: prim->verts) {
                v = v * 0.01;
            }
            auto &transform_r0 = prim->verts.add_attr<vec3f>("transform_r0");
            auto &transform_r1 = prim->verts.add_attr<vec3f>("transform_r1");
            auto &transform_r2 = prim->verts.add_attr<vec3f>("transform_r2");
            for (auto i = 0; i < prim->verts.size(); i++) {
                transform_r0[i] *= 0.01;
                transform_r1[i] *= 0.01;
                transform_r2[i] *= 0.01;
        }
        }
        set_output("prim", prim);
    }
};

ZENDEFNODE(NewFBXImportAnimation, {
    {
        "fbx_object",
        {gParamType_String, "clipName", ""},
        {gParamType_Float, "frameid"},
        {gParamType_Float, "fps", "25"},
        {gParamType_Bool, "ConvertUnits", "0"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"primitive"},
});

struct NewFBXImportCamera : INode {
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
        auto fbx_object = get_input2<FBXObject>("fbx_object");
        auto lSdkManager = fbx_object->lSdkManager;
        auto lScene = fbx_object->lScene;

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();
        auto &ud = prim->userData();

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


        FbxAnimStack* animStack = lScene->GetSrcObject<FbxAnimStack>(stack_index);
        ud.set2("clipinfo.name", std::string(animStack->GetName()));


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
            for (int j = 0; j < node_count; ++j) {
                auto pNode = lScene->GetNode(j);
                FbxAMatrix lGlobalPosition = pNode->EvaluateGlobalTransform(curTime);
                FbxMatrix transformMatrix;
                memcpy(&transformMatrix, &lGlobalPosition, sizeof(FbxMatrix));
                auto t = transformMatrix.GetRow(3);
                auto pos = vec3f(t[0], t[1], t[2]);

                auto r0 = transformMatrix.GetRow(0);
                auto r1 = transformMatrix.GetRow(1);
                auto r2 = transformMatrix.GetRow(2);
                auto view = vec3f(r0[0], r0[1], r0[2]);
                auto up = vec3f(r1[0], r1[1], r1[2]);
                auto right = vec3f(r2[0], r2[1], r2[2]);
                FbxCamera* pCamera = pNode->GetCamera();
                if (pCamera) {
                    set_output2("pos", pos);
                    set_output2("right", right);
                    set_output2("up", up);
                    set_output2("view", view);

                    float focal_length = pCamera->FocalLength.Get();
                    set_output2("focal_length", focal_length);
                    float m_ha = pCamera->GetApertureWidth() * 25.4; // inch -> mm
                    float m_va = pCamera->GetApertureHeight() * 25.4; // inch -> mm
                    set_output2("horizontalAperture", m_ha);
                    set_output2("verticalAperture", m_va);
                    auto m_nx = get_input2<float>("nx");
                    auto m_ny = get_input2<float>("ny");
                    float c_aspect = m_ha/m_va;
                    float u_aspect = m_nx/m_ny;
                    float fov_y = glm::degrees(2.0f * std::atan(m_va/(u_aspect/c_aspect) / (2.0f * focal_length)));
                    set_output2("fov_y", fov_y);
                    float _near = pCamera->NearPlane.Get();
                    float _far = pCamera->FarPlane.Get();
                    set_output2("near", _near);
                    set_output2("far", _far);
                }
            }
        }
    }
};

ZENDEFNODE(NewFBXImportCamera, {
    {
        {gParamType_Unknown, "fbx_object"},
        {gParamType_String, "clipName", ""},
        {gParamType_Float, "frameid"},
        {gParamType_Float, "fps", "25"},
        {gParamType_Bool, "ConvertUnits", "1"},
        {gParamType_Int, "nx", "1920"},
        {gParamType_Int, "ny", "1080"},
    },
    {
        {gParamType_Vec3f,"pos"},
        {gParamType_Vec3f,"up"},
        {gParamType_Vec3f,"view"},
        {gParamType_Vec3f,"right"},
        {gParamType_Float,"fov_y"},
        {gParamType_Float,"focal_length"},
        {gParamType_Float,"horizontalAperture"},
        {gParamType_Float,"verticalAperture"},
        {gParamType_Float,"near"},
        {gParamType_Float,"far"},
    },
    {},
    {"primitive"},
});
}
#endif
namespace zeno {
struct RigPoseItemObject : PrimitiveObject {
    std::string boneName;
    vec3f translate = {0, 0, 0};
    vec3f rotate = {0, 0, 0};
};
struct NewFBXRigPoseItem : INode {
    virtual void apply() override {
        auto item = std::make_shared<RigPoseItemObject>();
        item->boneName  = get_input2<std::string>("boneName");
        item->translate = get_input2<vec3f>("translate");
        item->rotate    = get_input2<vec3f>("rotate");
        set_output2("poseItem", std::move(item));
        }
};

ZENDEFNODE(NewFBXRigPoseItem, {
    {
        {gParamType_String, "boneName", ""},
        {gParamType_Vec3f, "translate", "0, 0, 0"},
        {gParamType_Vec3f, "rotate", "0, 0, 0"},
    },
    {
        {gParamType_Unknown, "poseItem"},
    },
    {},
    {"FBXSDK"},
});
static std::vector<glm::mat4> getBoneMatrix(PrimitiveObject *prim) {
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
static std::vector<glm::mat4> getInvertedBoneMatrix(PrimitiveObject *prim) {
        std::vector<glm::mat4> inv_matrixs;
        auto matrixs = getBoneMatrix(prim);
        for (auto i = 0; i < matrixs.size(); i++) {
            auto m = matrixs[i];
            auto inv_m = glm::inverse(m);
            inv_matrixs.push_back(inv_m);
        }
        return inv_matrixs;
}
static vec3f transform_pos(glm::mat4 &transform, vec3f pos) {
        auto p = transform * glm::vec4(pos[0], pos[1], pos[2], 1);
        return {p.x, p.y, p.z};
}
static vec3f transform_nrm(glm::mat4 &transform, vec3f pos) {
        auto p = glm::transpose(glm::inverse(transform)) * glm::vec4(pos[0], pos[1], pos[2], 0);
        return {p.x, p.y, p.z};
}

static std::map<std::string, int> getBoneNameMapping(PrimitiveObject *prim) {
    auto boneName_count = prim->userData().get2<int>("boneName_count");
    std::map<std::string, int> boneNames;
    for (auto i = 0; i < boneName_count; i++) {
        auto boneName = prim->userData().get2<std::string>(format("boneName_{}", i));
        boneNames[boneName] = i;
    }
    return boneNames;
}
static std::vector<std::string> getBoneNames(PrimitiveObject *prim) {
    auto boneName_count = prim->userData().get2<int>("boneName_count");
    std::vector<std::string> boneNames;
    boneNames.reserve(boneName_count);
    for (auto i = 0; i < boneName_count; i++) {
        auto boneName = prim->userData().get2<std::string>(format("boneName_{}", i));
        boneNames.emplace_back(boneName);
    }
    return boneNames;
}
static std::vector<int> TopologicalSorting(std::map<int, int> bone_connects, zeno::PrimitiveObject* skeleton) {
    std::vector<int> ordering;
    std::set<int> ordering_set;
    while (bone_connects.size()) {
        std::set<int> need_to_remove;
        for (auto [s, p]: bone_connects) {
            if (bone_connects.count(p) == 0) {
                if (ordering_set.count(p) == 0) {
                    ordering.emplace_back(p);
                    ordering_set.insert(p);
                }
                need_to_remove.insert(s);
            }
        }
        for (auto index: need_to_remove) {
            bone_connects.erase(index);
        }
    }
    for (auto i = 0; i < skeleton->verts.size(); i++) {
        if (ordering_set.count(i) == 0) {
            ordering.push_back(i);
        }
    }
    if (false) { // debug
        for (auto i = 0; i < ordering.size(); i++) {
            auto bi = ordering[i];
            auto bone_name = skeleton->userData().get2<std::string>(format("boneName_{}", bi));
            zeno::log_info("{}: {}: {}", i, bi, bone_name);
        }
    }
    return ordering;
}
struct NewFBXRigPose : INode {
    virtual void apply() override {
        auto skeleton = std::dynamic_pointer_cast<PrimitiveObject>(get_input<PrimitiveObject>("skeleton")->clone());
        auto nodelist = get_input<zeno::ListObject>("Transformations")->getRaw<RigPoseItemObject>();
        std::map<int, RigPoseItemObject*> Transformations;
        {
            auto boneNameMapping = getBoneNameMapping(skeleton.get());
            for (auto n: nodelist) {
                if (boneNameMapping.count(n->boneName)) {
                    Transformations[boneNameMapping[n->boneName]] = n;
                }
                else {
                    zeno::log_warn("{} missing", n->boneName);
                }
            }
        }

        auto WorldSpace = get_input2<bool>("WorldSpace");
        std::map<int, int> bone_connects;
        for (auto i = 0; i < skeleton->polys.size(); i++) {
            bone_connects[skeleton->loops[i * 2 + 1]] = skeleton->loops[i * 2];
        }

        auto ordering = TopologicalSorting(bone_connects, skeleton.get());
        auto &verts = skeleton->verts;
        auto &transform_r0 = skeleton->verts.add_attr<vec3f>("transform_r0");
        auto &transform_r1 = skeleton->verts.add_attr<vec3f>("transform_r1");
        auto &transform_r2 = skeleton->verts.add_attr<vec3f>("transform_r2");
        auto transforms    = getBoneMatrix(skeleton.get());
        auto transformsInv = getInvertedBoneMatrix(skeleton.get());
        std::map<int, glm::mat4> cache;
        for (auto bi: ordering) {
            glm::mat4 transform = glm::mat4(1.0f);
            if (Transformations.count(bi)) {
                auto trans = Transformations[bi];
                glm::mat4 matTrans = glm::translate(vec_to_other<glm::vec3>(trans->translate));
                glm::mat4 matRotx  = glm::rotate( (float)(trans->rotate[0] * M_PI / 180), glm::vec3(1,0,0) );
                glm::mat4 matRoty  = glm::rotate( (float)(trans->rotate[1] * M_PI / 180), glm::vec3(0,1,0) );
                glm::mat4 matRotz  = glm::rotate( (float)(trans->rotate[2] * M_PI / 180), glm::vec3(0,0,1) );
                transform = matTrans*matRoty*matRotx*matRotz;
                transform = transforms[bi] * transform * transformsInv[bi];
            }
            if (bone_connects.count(bi) && WorldSpace == false) {
                transform = cache[bone_connects[bi]] * transform;
            }
            cache[bi] = transform;
            verts[bi]        = transform_pos(transform, verts[bi]);
            transform_r0[bi] = transform_nrm(transform, transform_r0[bi]);
            transform_r1[bi] = transform_nrm(transform, transform_r1[bi]);
            transform_r2[bi] = transform_nrm(transform, transform_r2[bi]);
        }

        set_output2("skeleton", std::move(skeleton));
    }
};

ZENDEFNODE(NewFBXRigPose, {
    {
        {gParamType_Primitive, "skeleton"},
        {gParamType_Bool, "WorldSpace", "0"},
        {gParamType_List, "Transformations"},
    },
    {
        {gParamType_Primitive, "skeleton"},
    },
    {},
    {"FBXSDK"},
});
struct NewFBXBoneDeform : INode {
    std::vector<int> getBoneMapping(std::vector<std::string> &old, std::vector<std::string> &_new) {
        std::vector<int> mapping;
        mapping.reserve(old.size());
        for (auto i = 0; i < old.size(); i++) {
            auto index = std::find(_new.begin(), _new.end(), old[i]) - _new.begin();
            if (index == _new.size()) {
                index = -1;
                zeno::log_info("connot find bone: {}, {}", i, old[i]);
            }
            mapping.push_back(index);
        }
        return mapping;
    }
    virtual void apply() override {
        auto usingDualQuaternion = get_input2<std::string>("SkinningMethod") == "DualQuaternion";
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
        std::vector<DualQuaternion> dqs;
        dqs.reserve(geometryToDeformBoneNames.size());
        for (auto i = 0; i < geometryToDeformBoneNames.size(); i++) {
            glm::mat4 res_inv_matrix = glm::mat4(1);
            glm::mat4 deform_matrix = glm::mat4(1);
            if (restPointTransformsBoneMapping[i] >= 0 && deformPointTransformsBoneMapping[i] >= 0) {
                res_inv_matrix = restPointTransformsInv[restPointTransformsBoneMapping[i]];
                deform_matrix = deformPointTransforms[deformPointTransformsBoneMapping[i]];
            }
            auto matrix = deform_matrix * res_inv_matrix;
            matrixs.push_back(matrix);
            dqs.push_back(mat4ToDualQuat2(matrix));
        }

        auto prim = std::dynamic_pointer_cast<PrimitiveObject>(geometryToDeform->clone());

        int maxnum_boneWeight = prim->userData().get2<int>("maxnum_boneWeight");
        std::vector<std::vector<int>*> bi;
        std::vector<std::vector<float>*> bw;
        for (auto i = 0; i < maxnum_boneWeight; i++) {
            bi.push_back(&prim->verts.add_attr<int>(format("boneName_{}", i)));
            bw.push_back(&prim->verts.add_attr<float>(format("boneWeight_{}", i)));
        }
        size_t vert_count = prim->verts.size();
        #pragma omp parallel for
        for (auto i = 0; i < vert_count; i++) {
            auto opos = prim->verts[i];
            vec3f pos = {};
            DualQuaternion dq_acc({0, 0, 0, 0}, {0, 0, 0, 0});
            float w = 0;
            for (auto j = 0; j < maxnum_boneWeight; j++) {
                auto index = bi[j]->operator[](i);
                if (index < 0) {
                    continue;
                }
                auto weight = bw[j]->operator[](i);
                if (usingDualQuaternion) {
                    dq_acc = dq_acc + dqs[index] * weight;
            }
                else {
                    pos += transform_pos(matrixs[index], opos) * weight;
                }
                w += weight;
            }
            if (w > 0) {
                if (usingDualQuaternion) {
                    dq_acc = normalized(dq_acc);
                    prim->verts[i] = transformPoint2(dq_acc, opos);
                }
                else {
            prim->verts[i] = pos / w;
        }
            }
        }
        auto vectors_str = get_input2<std::string>("vectors");
        std::vector<std::string> vectors = zeno::split_str(vectors_str, ',');
        for (auto vector: vectors) {
            vector = zeno::trim_string(vector);
            if (vector.size()) {
                if (prim->verts.attr_is<vec3f>(vector)) {
                    auto &nrms = prim->verts.attr<vec3f>(vector);
                    #pragma omp parallel for
            for (auto i = 0; i < vert_count; i++) {
                glm::mat4 matrix(0);
                        DualQuaternion dq_acc({0, 0, 0, 0}, {0, 0, 0, 0});
                float w = 0;
                        for (auto j = 0; j < maxnum_boneWeight; j++) {
                            auto index = bi[j]->operator[](i);
                            if (index < 0) {
                        continue;
                    }
                            auto weight = bw[j]->operator[](i);
                            if (usingDualQuaternion) {
                                dq_acc = dq_acc + dqs[index] * weight;
                }
                            else {
                                matrix += matrixs[index] * weight;
                            }
                            w += weight;
                        }
                        if (w > 0) {
                            if (usingDualQuaternion) {
                                dq_acc = normalized(dq_acc);
                                nrms[i] = transformVector(dq_acc, nrms[i]);
                            }
                            else {
                matrix = matrix / w;
                                nrms[i] = transform_nrm(matrix, nrms[i]);
            }
                            nrms[i] = zeno::normalize(nrms[i]);
        }
                    }
                }
                if (prim->loops.attr_is<vec3f>(vector)) {
                    auto &nrms = prim->loops.attr<vec3f>(vector);
                    #pragma omp parallel for
                    for (auto i = 0; i < prim->loops.size(); i++) {
                        auto vi = prim->loops[i];
                        glm::mat4 matrix(0);
                        DualQuaternion dq_acc({0, 0, 0, 0}, {0, 0, 0, 0});
                        float w = 0;
                        for (auto j = 0; j < maxnum_boneWeight; j++) {
                            auto index = bi[j]->operator[](vi);
                            if (index < 0) {
                                continue;
                            }
                            auto weight = bw[j]->operator[](vi);
                            if (usingDualQuaternion) {
                                dq_acc = dq_acc + dqs[index] * weight;
                            }
                            else {
                                matrix += matrixs[index] * weight;
                            }
                            w += weight;
                        }
                        if (w > 0) {
                            if (usingDualQuaternion) {
                                dq_acc = normalized(dq_acc);
                                nrms[i] = transformVector(dq_acc, nrms[i]);
                            }
                            else {
                                matrix = matrix / w;
                                nrms[i] = transform_nrm(matrix, nrms[i]);
                            }
                            nrms[i] = zeno::normalize(nrms[i]);
                        }
                    }
                }
            }
        }

        set_output("prim", prim);
    }
};

ZENDEFNODE(NewFBXBoneDeform, {
    {
        {gParamType_Primitive, "GeometryToDeform"},
        {gParamType_Primitive, "RestPointTransforms"},
        {gParamType_Primitive, "DeformPointTransforms"},
        {"enum Linear DualQuaternion", "SkinningMethod", "Linear"},
        {gParamType_String, "vectors", "nrm,"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"primitive"},
});

struct NewFBXExtractKeyframe : INode {
    std::map<std::string, std::string> get_parent_name(PrimitiveObject *prim) {
        std::vector<std::string> bone_names = getBoneNames(prim);
        std::map<std::string, std::string> res;
        for (auto i = 0; i < prim->polys.size(); i++) {
            auto p = prim->loops[i * 2 + 0];
            auto s = prim->loops[i * 2 + 1];
            res[bone_names[s]] = bone_names[p];
        }
        return res;
    }
    virtual void apply() override {
        auto restPointTransformsPrim = get_input2<PrimitiveObject>("RestPointTransforms");
        auto restBoneNameMapping = getBoneNameMapping(restPointTransformsPrim.get());
        auto restPointTransforms = getBoneMatrix(restPointTransformsPrim.get());
        auto restPointTransformsInv = getInvertedBoneMatrix(restPointTransformsPrim.get());
        auto deformPointTransformsPrim = get_input2<PrimitiveObject>("DeformPointTransforms");
        auto deformBoneNameMapping = getBoneNameMapping(deformPointTransformsPrim.get());
        auto deformPointTransforms = getBoneMatrix(deformPointTransformsPrim.get());
        auto deformPointTransformsInv = getInvertedBoneMatrix(deformPointTransformsPrim.get());
        std::vector<std::string> keyframe_boneName;
        std::vector<glm::mat4> keyframe_bone_matrix;
        auto parent_names = get_parent_name(deformPointTransformsPrim.get());
        auto boneName_count = deformPointTransformsPrim->userData().get2<int>("boneName_count");
        for (auto i = 0; i < boneName_count; i++) {
            auto boneName = deformPointTransformsPrim->userData().get2<std::string>(format("boneName_{}", i));
            if (restBoneNameMapping.count(boneName) == 0) {
                continue;
            }
            keyframe_boneName.emplace_back(boneName);
            glm::mat4 parent_matrix = glm::mat4(1);
            if (parent_names.count(boneName)) {
                int pi = deformBoneNameMapping[parent_names[boneName]];
                if (restBoneNameMapping.count(parent_names[boneName])) {
                    auto rpi = restBoneNameMapping[parent_names[boneName]];
                    parent_matrix = restPointTransforms[rpi] * deformPointTransformsInv[pi];
                }
            }
            glm::mat4 restPointTransformInv = restPointTransformsInv[restBoneNameMapping[boneName]];
            glm::mat4 deformPointTransform = deformPointTransforms[i];
            auto keyframeTransform = restPointTransformInv * parent_matrix * deformPointTransform;
            keyframe_bone_matrix.emplace_back(keyframeTransform);
        }

        auto keyframe = std::make_shared<zeno::PrimitiveObject>();
        keyframe->userData().set2("boneName_count", int(keyframe_boneName.size()));
        for (auto i = 0; i < keyframe_boneName.size(); i++) {
            keyframe->userData().set2(format("boneName_{}", i), keyframe_boneName[i]);
        }
        keyframe->verts.resize(keyframe_boneName.size());
        auto &verts = keyframe->verts;
        auto &transform_r0 = keyframe->verts.add_attr<vec3f>("transform_r0");
        auto &transform_r1 = keyframe->verts.add_attr<vec3f>("transform_r1");
        auto &transform_r2 = keyframe->verts.add_attr<vec3f>("transform_r2");
        for (auto i = 0; i < keyframe->verts.size(); i++) {
            auto matrix = keyframe_bone_matrix[i];
            transform_r0[i] = {matrix[0][0], matrix[0][1], matrix[0][2]};
            transform_r1[i] = {matrix[1][0], matrix[1][1], matrix[1][2]};
            transform_r2[i] = {matrix[2][0], matrix[2][1], matrix[2][2]};
            verts[i]        = {matrix[3][0], matrix[3][1], matrix[3][2]};
        }
        auto &boneNames = keyframe->verts.add_attr<int>("boneName");
        std::iota(boneNames.begin(), boneNames.end(), 0);
        set_output2("keyframe", keyframe);
    }
};

ZENDEFNODE(NewFBXExtractKeyframe, {
    {
        {gParamType_Primitive, "RestPointTransforms"},
        {gParamType_Primitive, "DeformPointTransforms"},
    },
    {
        {gParamType_Primitive, "keyframe"},
    },
    {},
    {"primitive"},
});


struct NewFBXGenerateAnimation : INode {
    virtual void apply() override {
        auto keyframe = get_input2<PrimitiveObject>("keyframe");
        std::map<std::string, glm::mat4> Transformations;
        {
            auto keyframe_matrix = getBoneMatrix(keyframe.get());
            auto boneNames = getBoneNames(keyframe.get());
            for (auto i = 0; i < boneNames.size(); i++) {
                Transformations[boneNames[i]] = keyframe_matrix[i];
            }
        }

        auto skeleton = std::dynamic_pointer_cast<PrimitiveObject>(get_input<PrimitiveObject>("skeleton")->clone());
        std::map<int, int> bone_connects;
        for (auto i = 0; i < skeleton->polys.size(); i++) {
            bone_connects[skeleton->loops[i * 2 + 1]] = skeleton->loops[i * 2];
        }
        auto ordering = TopologicalSorting(bone_connects, skeleton.get());
        auto &verts = skeleton->verts;
        auto &transform_r0 = skeleton->verts.add_attr<vec3f>("transform_r0");
        auto &transform_r1 = skeleton->verts.add_attr<vec3f>("transform_r1");
        auto &transform_r2 = skeleton->verts.add_attr<vec3f>("transform_r2");
        auto transforms    = getBoneMatrix(skeleton.get());
        auto transformsInv = getInvertedBoneMatrix(skeleton.get());
        auto boneNames = getBoneNames(skeleton.get());
        std::map<int, glm::mat4> cache;
        for (auto bi: ordering) {
            glm::mat4 transform = glm::mat4(1.0f);
            if (Transformations.count(boneNames[bi])) {
                auto trans = Transformations[boneNames[bi]];
                transform = transforms[bi] * trans * transformsInv[bi];
            }
            if (bone_connects.count(bi)) {
                transform = cache[bone_connects[bi]] * transform;
            }
            cache[bi] = transform;
            verts[bi]        = transform_pos(transform, verts[bi]);
            transform_r0[bi] = transform_nrm(transform, transform_r0[bi]);
            transform_r1[bi] = transform_nrm(transform, transform_r1[bi]);
            transform_r2[bi] = transform_nrm(transform, transform_r2[bi]);
        }

        set_output2("DeformPointTransforms", skeleton);
    }
};

ZENDEFNODE(NewFBXGenerateAnimation, {
    {
        {gParamType_Primitive, "skeleton"},
        {gParamType_Primitive, "keyframe"},
    },
    {
        {gParamType_Primitive, "DeformPointTransforms"},
    },
    {},
    {"primitive"},
});


struct NormalView : INode {
    virtual void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        auto &nrms = prim->verts.attr<vec3f>("nrm");
        auto scale = get_input2<float>("scale");
        auto normals = std::make_shared<zeno::PrimitiveObject>();
        normals->verts.resize(prim->verts.size() * 2);
        for (auto i = 0; i < prim->verts.size(); i++) {
            normals->verts[i] = prim->verts[i];
            normals->verts[i + prim->size()] = prim->verts[i] + nrms[i] * scale;
        }
        normals->lines.resize(prim->verts.size());
        for (auto i = 0; i < prim->verts.size(); i++) {
            normals->lines[i] = vec2i(i, i + prim->verts.size());
        }
        set_output("normals", normals);
    }
};

ZENDEFNODE(NormalView, {
    {
        {gParamType_Primitive, "prim"},
        {gParamType_Float, "scale", "0.01"},
    },
    {
        {gParamType_Primitive, "normals"},
    },
    {},
    {"debug"},
});

struct BoneTransformView : INode {
    virtual void apply() override {
        auto bones = get_input2<PrimitiveObject>("bones");
        auto view = std::make_shared<zeno::PrimitiveObject>();
        auto scale = get_input2<float>("scale");
        auto index = get_input2<int>("index");
        view->verts.resize(bones->verts.size() * 6);
        auto &transform_r0 = bones->verts.attr<vec3f>("transform_r0");
        auto &transform_r1 = bones->verts.attr<vec3f>("transform_r1");
        auto &transform_r2 = bones->verts.attr<vec3f>("transform_r2");
        auto &clr = view->verts.add_attr<vec3f>("clr");
        for (auto i = 0; i < bones->verts.size(); i++) {
            view->verts[i * 6 + 0] = bones->verts[i];
            view->verts[i * 6 + 1] = bones->verts[i] + transform_r0[i] * scale;
            view->verts[i * 6 + 2] = bones->verts[i];
            view->verts[i * 6 + 3] = bones->verts[i] + transform_r1[i] * scale;
            view->verts[i * 6 + 4] = bones->verts[i];
            view->verts[i * 6 + 5] = bones->verts[i] + transform_r2[i] * scale;
            clr[i * 6 + 0] = {0.8, 0.2, 0.2};
            clr[i * 6 + 1] = {0.8, 0.2, 0.2};
            clr[i * 6 + 2] = {0.2, 0.8, 0.2};
            clr[i * 6 + 3] = {0.2, 0.8, 0.2};
            clr[i * 6 + 4] = {0.2, 0.2, 0.8};
            clr[i * 6 + 5] = {0.2, 0.2, 0.8};
        }
        view->loops.resize(view->verts.size());
        std::iota(view->loops.begin(), view->loops.end(), 0);
        view->polys.resize(bones->verts.size() * 3);
        for (auto i = 0; i < bones->verts.size() * 3; i++) {
            view->polys[i] = {i * 2, 2};
        }
        set_output("view", view);
    }
};

ZENDEFNODE(BoneTransformView, {
    {
        {gParamType_Primitive, "bones"},
        {gParamType_Float, "scale", "0.1"},
        {gParamType_Int, "index", "-1"},
    },
    {
        {gParamType_Primitive, "view"},
    },
    {},
    {"debug"},
});

struct PrimAttrFlat : INode {
    virtual void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        auto params = get_input2<std::string>("params");
        std::vector<std::string> params_ = zeno::split_str(params, ',');
        std::vector<float> values;
        for (auto i = 0; i < prim->size(); i++) {
            for (const auto& param: params_) {
                auto value = prim->attr<vec3f>(param);
                values.push_back(value[i][0]);
                values.push_back(value[i][1]);
                values.push_back(value[i][2]);
            }
        }

        auto output = std::make_shared<zeno::PrimitiveObject>();
        output->resize(values.size());
        auto &value = output->add_attr<float>("value");
        for (auto i = 0; i < values.size(); i++) {
            value[i] = values[i];
        }

        set_output("output", output);
    }
};

ZENDEFNODE(PrimAttrFlat, {
    {
        {gParamType_Primitive, "prim"},
        {"string", "params", "transform_r0,transform_r1,transform_r2"},
    },
    {
        {gParamType_Primitive, "output"},
    },
    {},
    {"debug"},
});

}