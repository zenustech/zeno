#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <vector>
#include <numeric>

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>

#include "zeno/utils/log.h"
#include <zeno/types/UserData.h>
#include "zeno/types/PrimitiveObject.h"
#include <zeno/types/PrimitiveUtils.h>
#include "zeno/utils/scope_exit.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <tinygltf/json.hpp>
using Json = nlohmann::json;

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

bool GetMesh(FbxNode* pNode, std::shared_ptr<PrimitiveObject> prim, std::string name) {
    FbxMesh* pMesh = pNode->GetMesh();
    if (!pMesh) return false;
    std::string nodeName = pNode->GetName();
    if (name.size() > 0 && nodeName != name) {
        return false;
    }
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
            zeno::log_info("arr->GetReferenceMode(): {}", arr->GetReferenceMode());
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
    if (pMesh->GetElementMaterialCount() > 0) {
        auto &faceset = prim->polys.add_attr<int>("faceset");
        for (auto i = 0; i < numPolygons; ++i) {
            faceset[i] = pMesh->GetElementMaterial()->GetIndexArray().GetAt(i);
        }
        auto mat_count = pNode->GetMaterialCount();
        ud.set2("faceset_count", mat_count);
        Json mat_json;
        for (auto i = 0; i < mat_count; i++) {
            FbxSurfaceMaterial* material = pNode->GetMaterial(i);
            ud.set2(format("faceset_{}", i), material->GetName());
            Json json;
            json["mat_name"] = material->GetName();
            {
                FbxProperty diff_property = material->FindProperty(FbxSurfaceMaterial::sDiffuse);
                if (diff_property.IsValid()) {
                    int textureCount = diff_property.GetSrcObjectCount<FbxTexture>();
                    for (int i = 0; i < textureCount; ++i) {
                        FbxFileTexture* texture = FbxCast<FbxFileTexture>(diff_property.GetSrcObject<FbxTexture>(i));
                        if (texture) {
                            json["diffuse_tex"] = texture->GetFileName();
                        }
                    }
                }
                FbxProperty normal_property = material->FindProperty(FbxSurfaceMaterial::sNormalMap);
                if (normal_property.IsValid()) {
                    int textureCount = normal_property.GetSrcObjectCount<FbxTexture>();
                    for (int i = 0; i < textureCount; ++i) {
                        FbxFileTexture* texture = FbxCast<FbxFileTexture>(normal_property.GetSrcObject<FbxTexture>(i));
                        if (texture) {
                            json["normal_map_tex"] = texture->GetFileName();
                        }
                    }
                }
            }
            mat_json.push_back(json);
        }
        ud.set2("material", mat_json.dump());
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
            set_output("prim", std::shared_ptr<PrimitiveObject>());
            return;
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
        std::vector<std::string> availableRootNames;
        if(lRootNode) {
            for(int i = 0; i < lRootNode->GetChildCount(); i++) {
                auto pNode = lRootNode->GetChild(i);
                FbxMesh* pMesh = pNode->GetMesh();
                if (pMesh) {
                    std::string meshName = pNode->GetName();
                    availableRootNames.emplace_back(meshName);
                }
            }
            ud.set2("AvailableRootName_count", int(availableRootNames.size()));
            for (int i = 0; i < availableRootNames.size(); i++) {
                ud.set2(format("AvailableRootName_{}", i), availableRootNames[i]);
            }
        }
        auto rootName = get_input2<std::string>("rootName");
        if(lRootNode) {
            for(int i = 0; i < lRootNode->GetChildCount(); i++) {
                auto pNode = lRootNode->GetChild(i);
                if (GetMesh(pNode, prim, rootName)) {
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
        {"string", "rootName", ""},
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

struct AnimationMix : INode {
    virtual void apply() override {
        auto anim_from = get_input2<PrimitiveObject>("anim_from");
        auto anim_to = get_input2<PrimitiveObject>("anim_to");
        auto mix_value = get_input2<float>("mix_value");
        if (anim_from->userData().get2<int>("boneName_count") != anim_to->userData().get2<int>("boneName_count")) {
            throw zeno::makeError("anim_from anim_to not matching");
        }
        auto anim = std::dynamic_pointer_cast<PrimitiveObject>(anim_from->clone());
        auto &verts_from = anim_from->verts;
        auto &transform_r0_from = anim_from->verts.add_attr<vec3f>("transform_r0");
        auto &transform_r1_from = anim_from->verts.add_attr<vec3f>("transform_r1");
        auto &transform_r2_from = anim_from->verts.add_attr<vec3f>("transform_r2");
        auto &verts_to = anim_to->verts;
        auto &transform_r0_to = anim_to->verts.add_attr<vec3f>("transform_r0");
        auto &transform_r1_to = anim_to->verts.add_attr<vec3f>("transform_r1");
        auto &transform_r2_to = anim_to->verts.add_attr<vec3f>("transform_r2");
        auto &verts = anim->verts;
        auto &transform_r0 = anim->verts.add_attr<vec3f>("transform_r0");
        auto &transform_r1 = anim->verts.add_attr<vec3f>("transform_r1");
        auto &transform_r2 = anim->verts.add_attr<vec3f>("transform_r2");
        for (auto i = 0; i < anim->verts.size(); i++) {
            verts[i] = mix(verts_from[i], verts_to[i], mix_value);
            if (get_input2<bool>("decompose")) {
                glm::mat4 matrix_from;
                matrix_from[0] = {transform_r0_from[i][0], transform_r0_from[i][1], transform_r0_from[i][2], 0};
                matrix_from[1] = {transform_r1_from[i][0], transform_r1_from[i][1], transform_r1_from[i][2], 0};
                matrix_from[2] = {transform_r2_from[i][0], transform_r2_from[i][1], transform_r2_from[i][2], 0};
                matrix_from[3] = {verts_from[i][0], verts_from[i][1], verts_from[i][2], 1};
                glm::vec3 from_Scale;
                glm::quat from_Orientation;
                glm::vec3 from_Translation;
                glm::vec3 from_Skew;
                glm::vec4 from_Perspective;
                glm::decompose(matrix_from, from_Scale, from_Orientation, from_Translation, from_Skew, from_Perspective);
                from_Orientation = glm::conjugate(from_Orientation);

                glm::mat4 matrix_to;
                matrix_to[0] = {transform_r0_to[i][0], transform_r0_to[i][1], transform_r0_to[i][2], 0};
                matrix_to[1] = {transform_r1_to[i][0], transform_r1_to[i][1], transform_r1_to[i][2], 0};
                matrix_to[2] = {transform_r2_to[i][0], transform_r2_to[i][1], transform_r2_to[i][2], 0};
                matrix_to[3] = {verts_to[i][0], verts_to[i][1], verts_to[i][2], 1};
                glm::vec3 to_Scale;
                glm::quat to_Orientation;
                glm::vec3 to_Translation;
                glm::vec3 to_Skew;
                glm::vec4 to_Perspective;
                glm::decompose(matrix_from, to_Scale, to_Orientation, to_Translation, to_Skew, to_Perspective);
                to_Orientation = glm::conjugate(to_Orientation);
                glm::vec3 mixed_Scale = glm::mix(from_Scale, to_Scale, mix_value);
                glm::quat mixed_Orientation = glm::slerp(from_Orientation, to_Orientation, mix_value);
                glm::vec3 mixed_Translation = glm::mix(from_Translation, to_Translation, mix_value);

                glm::mat4 transformMatrix = glm::mat4(1.0f);
                transformMatrix = glm::scale(transformMatrix, mixed_Scale);
                transformMatrix = glm::mat4_cast(mixed_Orientation) * transformMatrix;
                transformMatrix = glm::translate(transformMatrix, mixed_Translation);
                auto r0 = transformMatrix[0];
                auto r1 = transformMatrix[1];
                auto r2 = transformMatrix[2];

                transform_r0[i] = vec3f(r0[0], r0[1], r0[2]);
                transform_r1[i] = vec3f(r1[0], r1[1], r1[2]);
                transform_r2[i] = vec3f(r2[0], r2[1], r2[2]);
            }
            else {
                transform_r0[i] = mix(transform_r0_from[i], transform_r0_to[i], mix_value);
                transform_r1[i] = mix(transform_r1_from[i], transform_r1_to[i], mix_value);
                transform_r2[i] = mix(transform_r2_from[i], transform_r2_to[i], mix_value);
            }
        }
        set_output("anim", anim);
    }
};

ZENDEFNODE(AnimationMix, {
    {
        {"prim", "anim_from"},
        {"prim", "anim_to"},
        {"float", "mix_value", "0.5"},
        {"bool", "decompose", "true"},
    },
    {
        "anim",
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
        {"readpath", "path"},
        {"string", "clipName", ""},
        {"frameid"},
        {"float", "fps", "25"},
        {"bool", "ConvertUnits", "1"},
        {"int", "nx", "1920"},
        {"int", "ny", "1080"},
    },
    {
        "pos",
        "up",
        "view",
        "right",
        "fov_y",
        "focal_length",
        "horizontalAperture",
        "verticalAperture",
        "near",
        "far",
    },
    {},
    {"primitive"},
});
// Create a cube mesh.
FbxNode* CreateSkin(FbxScene* pScene, char* pName, std::shared_ptr<PrimitiveObject> prim) {
    FbxMesh* lMesh = FbxMesh::Create(pScene, pName);

    // Create control points.
    lMesh->InitControlPoints(prim->verts.size());
    FbxVector4* lControlPoints = lMesh->GetControlPoints();
    for (auto i = 0; i < prim->verts.size(); i++) {
        lControlPoints[i] = FbxVector4(prim->verts[i][0], prim->verts[i][1], prim->verts[i][2]);
    }

    if (prim->verts.has_attr("nrm")) {
        FbxGeometryElementNormal* lGeometryElementNormal= lMesh->CreateElementNormal();
        lGeometryElementNormal->SetMappingMode(FbxGeometryElement::eByControlPoint);
        lGeometryElementNormal->SetReferenceMode(FbxGeometryElement::eDirect);
        auto &nrm = prim->verts.attr<vec3f>("nrm");
        for (auto i = 0; i < prim->verts.size(); i++) {
            lGeometryElementNormal->GetDirectArray().Add(FbxVector4(nrm[i][0], nrm[i][1], nrm[i][2]));
        }
    }

    // Create UV for Diffuse channel.
    if (prim->loops.has_attr("uvs")) {
        FbxGeometryElementUV* lUVDiffuseElement = lMesh->CreateElementUV( "DiffuseUV");
        lUVDiffuseElement->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
        lUVDiffuseElement->SetReferenceMode(FbxGeometryElement::eIndexToDirect);
        for (auto i = 0; i < prim->uvs.size(); i++) {
            lUVDiffuseElement->GetDirectArray().Add(FbxVector2(prim->uvs[i][0], prim->uvs[i][1]));
        }

        lUVDiffuseElement->GetIndexArray().SetCount(prim->loops.size());
        auto &uvs = prim->loops.attr<int>("uvs");
        for (auto i = 0; i < uvs.size(); i++) {
            lUVDiffuseElement->GetIndexArray().SetAt(i, uvs[i]);
        }
    }

    // Create polygons. Assign texture and texture UV indices.
    for (auto i = 0; i < prim->polys.size(); i++) {
        lMesh->BeginPolygon(-1, -1, -1, false);
        auto poly = prim->polys[i];
        for (auto j = 0; j < poly[1]; j++) {
            lMesh->AddPolygon(prim->loops[poly[0] + j]);
        }
        lMesh->EndPolygon();
    }

    // create a FbxNode
    FbxNode* lNode = FbxNode::Create(pScene,pName);

    // set the node attribute
    lNode->SetNodeAttribute(lMesh);

    // set the shading mode to view texture
    lNode->SetShadingMode(FbxNode::eTextureShading);

    // rescale the cube
//    lNode->LclScaling.Set(FbxVector4(0.3, 0.3, 0.3));

    // return the FbxNode
    return lNode;
}

void createSkeleton(FbxScene* scene, std::shared_ptr<PrimitiveObject> prim) {
    std::vector<FbxNode*> boneNodes;
    boneNodes.reserve(prim->verts.size() + 1);
    {
        FbxSkeleton* SkeletonAttribute = FbxSkeleton::Create(scene, "RootNode");
        SkeletonAttribute->SetSkeletonType(FbxSkeleton::eRoot);
        FbxNode* BoneNode = FbxNode::Create(scene, "RootNode");
        BoneNode->SetNodeAttribute(SkeletonAttribute);
        boneNodes.push_back(BoneNode);
    }
    auto &transform_r0 = prim->verts.add_attr<vec3f>("transform_r0");
    auto &transform_r1 = prim->verts.add_attr<vec3f>("transform_r1");
    auto &transform_r2 = prim->verts.add_attr<vec3f>("transform_r2");
    for (auto i = 0; i < prim->verts.size(); i++) {
        auto boneName = prim->userData().get2<std::string>(format("boneName_{}", i));
        FbxSkeleton* SkeletonAttribute = FbxSkeleton::Create(scene, boneName.c_str());
        SkeletonAttribute->SetSkeletonType(FbxSkeleton::eLimbNode);
        FbxNode* BoneNode = FbxNode::Create(scene, boneName.c_str());
        BoneNode->SetNodeAttribute(SkeletonAttribute);
        auto pos = prim->verts[i];
        glm::mat4 matrix;
        matrix[0] = {transform_r0[i][0], transform_r0[i][1], transform_r0[i][2], 0};
        matrix[1] = {transform_r1[i][0], transform_r1[i][1], transform_r1[i][2], 0};
        matrix[2] = {transform_r2[i][0], transform_r2[i][1], transform_r2[i][2], 0};
        matrix[3] = {pos[0], pos[1], pos[2], 1};
        glm::vec3 Scale;
        glm::quat Orientation;
        glm::vec3 Translation;
        glm::vec3 Skew;
        glm::vec4 Perspective;
        glm::decompose(matrix, Scale, Orientation, Translation, Skew, Perspective);
        glm::vec3 rot = glm::eulerAngles(Orientation);

        BoneNode->LclTranslation.Set(FbxDouble3(pos[0], pos[1], pos[2]));
        BoneNode->LclRotation.Set(FbxDouble3(rot[0], rot[1], rot[2]));
        BoneNode->LclScaling.Set(FbxDouble3(Scale[0], Scale[1], Scale[2]));
        boneNodes.push_back(BoneNode);
    }
    std::set<int> has_parent;
    for (auto i = 0; i < prim->loops.size() / 2; i++) {
        auto p = prim->loops[i * 2 + 0] + 1;
        auto c = prim->loops[i * 2 + 1] + 1;
        boneNodes[p]->AddChild(boneNodes[c]);
        has_parent.insert(c);
    }
    for (auto i = 0; i < prim->verts.size(); i++) {
        if (has_parent.count(i+1) == 0) {
            boneNodes[0]->AddChild(boneNodes[i+1]);
        }
    }
}
struct NewFBXWrite : INode {
    virtual void apply() override {
        FbxManager* manager = FbxManager::Create();
        FbxIOSettings* ios = FbxIOSettings::Create(manager, IOSROOT);
        manager->SetIOSettings(ios);

        // 创建一个场景
        FbxScene* scene = FbxScene::Create(manager, "MyScene");
        {
            auto skin = get_input2<PrimitiveObject>("skin");
            if (skin->tris.size()) {
                primPolygonate(skin.get(), true);
            }
            if (get_input2<bool>("ConvertUnits")) {
                for (auto i = 0; i < skin->verts.size(); i++) {
                    skin->verts[i] *= 100.0f;
                }
            }
            FbxNode* lSkin = CreateSkin(scene, "Mesh", skin);
            scene->GetRootNode()->AddChild(lSkin);
        }

        // 导出场景到FBX文件
        int fileFormat = manager->GetIOPluginRegistry()->FindWriterIDByDescription("FBX ascii");
        FbxExporter* exporter = FbxExporter::Create(manager, "");
        exporter->Initialize(get_input2<std::string>("path").c_str(), fileFormat, manager->GetIOSettings());
        exporter->Export(scene);
        exporter->Destroy();

        // 清理
        scene->Destroy();
        manager->Destroy();
    }
};

//ZENDEFNODE(NewFBXWrite, {
//    {
//        "skin",
//        "skeleton",
//        {"readpath", "path", "test0428.fbx"},
//        {"bool", "ConvertUnits", "1"},
//    },
//    {
//    },
//    {},
//    {"FBX"},
//});
}
#endif
namespace zeno {
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
    vec3f transform_nrm(glm::mat4 &transform, vec3f pos) {
        auto p = glm::transpose(glm::inverse(transform)) * glm::vec4(pos[0], pos[1], pos[2], 0);
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
        if (prim->verts.attr_is<vec3f>("nrm")) {
            auto &nrms = prim->verts.attr<vec3f>("nrm");
            for (auto i = 0; i < vert_count; i++) {
                glm::mat4 matrix(0);
                float w = 0;
                for (auto j = 0; j < 4; j++) {
                    if (bi[i][j] < 0) {
                        continue;
                    }
                    matrix += matrixs[bi[i][j]] * bw[i][j];
                    w += bw[i][j];
                }
                matrix = matrix / w;
                auto nrm = transform_nrm(matrix, nrms[i]);
                nrms[i] = zeno::normalize(nrm );
            }
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

float sqr(float v) {
    return v * v;
}
struct TwoBoneIK : INode {
    virtual void apply() override {
        auto root = get_input2<vec3f>("root");
        auto joint = get_input2<vec3f>("joint");
        auto end = get_input2<vec3f>("end");
        auto effector = get_input2<vec3f>("effector");
        vec3f jointTarget = has_input2<vec3f>("jointTarget")? get_input2<vec3f>("jointTarget") : joint;
        vec3f output_joint = {};
        vec3f output_end = {};

        auto root_to_effect = effector - root;
        auto root_to_end = end - root;
        auto root_to_jointTarget = jointTarget - root;

        auto upper_limb_length = zeno::length(root - joint);
        auto lower_limb_length = zeno::length(joint - end);
        auto desired_length = zeno::length(root_to_effect);
        if (desired_length < abs(upper_limb_length - lower_limb_length)) {
            output_end = root + normalize(root_to_effect) * abs(upper_limb_length - lower_limb_length);
            output_joint = root + normalize(root_to_effect) * upper_limb_length;
        }
        else if (desired_length > upper_limb_length + lower_limb_length) {
            output_end = root + normalize(root_to_effect) * (upper_limb_length + lower_limb_length);
            output_joint = root + normalize(root_to_effect) * upper_limb_length;
        }
        else {
            output_end = effector;

            vec3f to_right = normalize(cross(root_to_effect, root_to_jointTarget));
            if (!has_input<vec3f>("jointTarget")) {
                to_right = normalize(cross(root_to_end, root_to_jointTarget));
            }

            vec3f to_pole = normalize(cross(to_right, root_to_effect));
            float cos_theta = (sqr(upper_limb_length) + sqr(desired_length) - sqr(lower_limb_length)) / (2.0f * upper_limb_length * desired_length);
            float sin_theta = sqrt(1 - sqr(cos_theta));
            output_joint = root + normalize(root_to_effect) * cos_theta + to_pole * sin_theta;
        }
        auto prim = std::make_shared<PrimitiveObject>();
        prim->verts.resize(3);
        prim->verts[0] = root;
        prim->verts[1] = output_joint;
        prim->verts[2] = output_end;
        prim->lines.emplace_back(0, 1);
        prim->lines.emplace_back(1, 2);

        set_output2("prim", prim);
    }
};

ZENDEFNODE(TwoBoneIK, {
    {
        {"vec3f", "root", "0, 2, 0"},
        {"vec3f", "joint", "0, 1, 0.01"},
        {"vec3f", "end", "0, 0, 0"},
        {"vec3f", "effector", "0, 0.2, 0.2"},
        { "jointTarget" },
    },
    {
        "prim",
//        "joint",
//        "end",
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
        "prim",
        {"float", "scale", "0.01"},
    },
    {
        "normals",
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
        "bones",
        {"float", "scale", "0.1"},
        {"int", "index", "-1"},
    },
    {
        "view",
    },
    {},
    {"debug"},
});
}