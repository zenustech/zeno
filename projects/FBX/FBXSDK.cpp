#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <numeric>
#include <filesystem>

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>

#include "zeno/utils/log.h"
#include "zeno/utils/bit_operations.h"
#include <zeno/types/UserData.h>
#include "zeno/types/PrimitiveObject.h"
#include "zeno/utils/scope_exit.h"
#include "zeno/funcs/PrimitiveUtils.h"
#include "zeno/utils/string.h"
#include "zeno/utils/arrayindex.h"
#include "zeno/utils/variantswitch.h"
#include "zeno/utils/eulerangle.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include "DualQuaternion.h"
#include "zeno/extra/TempNode.h"
#include "magic_enum.hpp"
#include <tinygltf/json.hpp>
using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

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
        FbxRootNodeUtility::RemoveAllFbxRoots(fbx_object->lScene);

        // The file is imported; so get rid of the importer.
        lImporter->Destroy();
        fbx_object->userData().set2("version", vec3i(major, minor, revision));
        usedPath = lFilename;
        _inner_fbx_object = fbx_object;
        fbx_object->userData().set2("file_path", usedPath);

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
    {"FBXSDK"},
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

static std::shared_ptr<PrimitiveObject> GetMesh(FbxNode* pNode, bool output_tex_even_missing) {
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
    auto &faceset = prim->polys.add_attr<int>("faceset");
    std::fill(faceset.begin(), faceset.end(), -1);
    int mat_count = 0;
    if (pMesh->GetElementMaterialCount() > 0) {
        for (auto i = 0; i < numPolygons; ++i) {
            faceset[i] = pMesh->GetElementMaterial()->GetIndexArray().GetAt(i);
        }
        mat_count = pNode->GetMaterialCount();
        for (auto i = 0; i < mat_count; i++) {
            FbxSurfaceMaterial* material = pNode->GetMaterial(i);
            ud.set2(format("faceset_{}", i), material->GetName());
        }
    }
    ud.set2("faceset_count", mat_count);
    prim_set_abcpath(prim.get(), format("/ABC/{}", nodeName));
    if (mat_count > 0) {
        for (auto i = 0; i < mat_count; i++) {
            FbxSurfaceMaterial* material = pNode->GetMaterial(i);
            ud.set2(format("faceset_{}", i), material->GetName());
            Json json;
            std::string mat_name = material->GetName();
            {
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sEmissive);
                    if (output_tex_even_missing) {
                        json["emissive_tex"] = "";
                    }
                    if (property.IsValid()) {
                        FbxDouble3 value = property.Get<FbxDouble3>();
                        json["emissive_value"] = {value[0], value[1], value[2]};
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["emissive_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sAmbient);
                    if (output_tex_even_missing) {
                        json["ambient_tex"] = "";
                    }
                    if (property.IsValid()) {
                        FbxDouble3 value = property.Get<FbxDouble3>();
                        json["ambient_value"] = {value[0], value[1], value[2]};
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["ambient_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sDiffuse);
                    if (output_tex_even_missing) {
                        json["diffuse_tex"] = "";
                    }
                    if (property.IsValid()) {
                        FbxDouble3 value = property.Get<FbxDouble3>();
                        json["diffuse_value"] = {value[0], value[1], value[2]};
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["diffuse_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sSpecular);
                    if (output_tex_even_missing) {
                        json["specular_tex"] = "";
                    }
                    if (property.IsValid()) {
                        FbxDouble3 value = property.Get<FbxDouble3>();
                        json["specular_value"] = {value[0], value[1], value[2]};
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["specular_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sShininess);
                    if (output_tex_even_missing) {
                        json["shininess_tex"] = "";
                    }
                    if (property.IsValid()) {
                        double value = property.Get<double>();
                        json["shininess_value"] = value;
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["shininess_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sBump);
                    if (output_tex_even_missing) {
                        json["bump_tex"] = "";
                    }
                    if (property.IsValid()) {
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["bump_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sNormalMap);
                    if (output_tex_even_missing) {
                        json["normal_map_tex"] = "";
                    }
                    if (property.IsValid()) {
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["normal_map_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sTransparentColor);
                    if (output_tex_even_missing) {
                        json["transparent_color_tex"] = "";
                    }
                    if (property.IsValid()) {
                        FbxDouble3 value = property.Get<FbxDouble3>();
                        json["transparent_color_value"] = {value[0], value[1], value[2]};
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["transparent_color_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sTransparencyFactor);
                    if (output_tex_even_missing) {
                        json["opacity_tex"] = "";
                    }
                    if (property.IsValid()) {
                        double value = property.Get<double>();
                        json["opacity_value"] = value;
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["opacity_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sReflection);
                    if (output_tex_even_missing) {
                        json["reflection_tex"] = "";
                    }
                    if (property.IsValid()) {
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["reflection_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sDisplacementColor);
                    if (output_tex_even_missing) {
                        json["displacement_color_tex"] = "";
                    }
                    if (property.IsValid()) {
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["displacement_color_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
                {
                    FbxProperty property = material->FindProperty(FbxSurfaceMaterial::sVectorDisplacementColor);
                    if (output_tex_even_missing) {
                        json["vector_displacement_color_tex"] = "";
                    }
                    if (property.IsValid()) {
                        int textureCount = property.GetSrcObjectCount<FbxTexture>();
                        for (int i = 0; i < textureCount; ++i) {
                            FbxFileTexture* texture = FbxCast<FbxFileTexture>(property.GetSrcObject<FbxTexture>(i));
                            if (texture) {
                                json["vector_displacement_color_tex"] = texture->GetFileName();
                            }
                        }
                    }
                }
            }
            ud.set2(mat_name, json.dump());
        }
    }
    return prim;
}

static std::shared_ptr<PrimitiveObject> GetSkeleton(FbxNode* pNode) {
    FbxMesh* pMesh = pNode->GetMesh();
    if (!pMesh) return nullptr;
    std::vector<std::string> bone_names;
    std::vector<vec3f> poss;
    std::vector<vec3f> transform_r0;
    std::vector<vec3f> transform_r1;
    std::vector<vec3f> transform_r2;
    std::map<std::string, std::string> parent_mapping;
    if (pMesh->GetDeformerCount(FbxDeformer::eSkin)) {
        FbxSkin* pSkin = (FbxSkin*)pMesh->GetDeformer(0, FbxDeformer::eSkin);
        // Iterate over each cluster (bone)
        for (int j = 0; j < pSkin->GetClusterCount(); ++j) {
            FbxCluster* pCluster = pSkin->GetCluster(j);

            FbxNode* pBoneNode = pCluster->GetLink();
            if (!pBoneNode) continue;
            FbxAMatrix transformLinkMatrix;
            pCluster->GetTransformLinkMatrix(transformLinkMatrix);

            // The transformation of the mesh at binding time
            FbxAMatrix transformMatrix;
            pCluster->GetTransformMatrix(transformMatrix);

            // Inverse bind matrix.
            FbxAMatrix bindMatrix_ = transformMatrix.Inverse() * transformLinkMatrix;
            auto bindMatrix = bit_cast<FbxMatrix>(bindMatrix_);
            auto t = bindMatrix.GetRow(3);
            poss.emplace_back(t[0], t[1], t[2]);

            auto r0 = bindMatrix.GetRow(0);
            auto r1 = bindMatrix.GetRow(1);
            auto r2 = bindMatrix.GetRow(2);
            transform_r0.emplace_back(r0[0], r0[1], r0[2]);
            transform_r1.emplace_back(r1[0], r1[1], r1[2]);
            transform_r2.emplace_back(r2[0], r2[1], r2[2]);
            std::string boneName = pBoneNode->GetName();
            bone_names.emplace_back(boneName);
            auto pParentNode = pBoneNode->GetParent();
            if (pParentNode) {
                std::string parentName = pParentNode->GetName();
                parent_mapping[boneName] = parentName;
            }
        }
    }
    std::string nodeName = pNode->GetName();
    auto prim = std::make_shared<PrimitiveObject>();
    prim->userData().set2("RootName", nodeName);
    prim->verts.resize(bone_names.size());
    prim->verts.values = poss;
    prim->verts.add_attr<vec3f>("transform_r0") = transform_r0;
    prim->verts.add_attr<vec3f>("transform_r1") = transform_r1;
    prim->verts.add_attr<vec3f>("transform_r2") = transform_r2;
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
    auto &boneNames = prim->verts.add_attr<int>("boneName");
    std::iota(boneNames.begin(), boneNames.end(), 0);
    prim->userData().set2("boneName_count", int(bone_names.size()));
    for (auto i = 0; i < bone_names.size(); i++) {
        prim->userData().set2(zeno::format("boneName_{}", i), bone_names[i]);
    }
    return prim;
}

static void TraverseNodesToGetNames(FbxNode* pNode, std::vector<std::string> &names) {
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

static void TraverseNodesToGetPrim(FbxNode* pNode, std::string target_name, std::shared_ptr<PrimitiveObject> &prim, bool output_tex_even_missing) {
    if (!pNode) return;

    FbxMesh* mesh = pNode->GetMesh();
    if (mesh) {
        auto name = pNode->GetName();
        if (target_name == name) {
            auto sub_prim = GetMesh(pNode, output_tex_even_missing);
            if (sub_prim) {
                prim = sub_prim;
            }
            return;
        }
    }

    for (int i = 0; i < pNode->GetChildCount(); i++) {
        TraverseNodesToGetPrim(pNode->GetChild(i), target_name, prim, output_tex_even_missing);
    }
}
static void TraverseNodesToGetPrims(FbxNode* pNode, std::vector<std::shared_ptr<PrimitiveObject>> &prims, bool output_tex_even_missing) {
    if (!pNode) return;

    FbxMesh* mesh = pNode->GetMesh();
    if (mesh) {
        auto sub_prim = GetMesh(pNode, output_tex_even_missing);
        if (sub_prim) {
            prims.push_back(sub_prim);
        }
    }

    for (int i = 0; i < pNode->GetChildCount(); i++) {
        TraverseNodesToGetPrims(pNode->GetChild(i), prims, output_tex_even_missing);
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
        bool output_tex_even_missing = get_input2<bool>("OutputTexEvenMissing");
        if(lRootNode) {
            TraverseNodesToGetNames(lRootNode, availableRootNames);
            auto rootName = get_input2<std::string>("rootName");
            if (rootName.empty()) {
                std::vector<std::shared_ptr<PrimitiveObject>> prims;
                TraverseNodesToGetPrims(lRootNode, prims, output_tex_even_missing);

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
                TraverseNodesToGetPrim(lRootNode, rootName, prim, output_tex_even_missing);
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
            auto file_path = fbx_object->userData().get2<std::string>("file_path");
            ud.set2("file_path", file_path);
        }
        if (get_input2<bool>("CopyFacesetToMatid")) {
            prim_copy_faceset_to_matid(prim.get());
        }
        set_output("prim", prim);
    }
};

ZENDEFNODE(NewFBXImportSkin, {
    {
        "fbx_object",
        {"string", "rootName", ""},
        {"bool", "ConvertUnits", "0"},
        {"string", "vectors", "nrm,"},
        {"bool", "CopyVectorsFromLoopsToVert", "1"},
        {"bool", "CopyFacesetToMatid", "1"},
        {"bool", "OutputTexEvenMissing", "0"},
    },
    {
        "prim",
    },
    {},
    {"FBXSDK"},
});

struct NewFBXResolveTexPath : INode {
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
        {
            auto orig_path = fs::u8path(origPath);
            std::error_code ec;
            if (std::filesystem::exists(orig_path, ec)) {
                oPath = origPath;
            }
        }
        std::vector<std::string> paths;
        StringSplitReverse(origPath, '/', paths);
        formPath(paths);
        for(int i=0; i<paths.size(); i++)
        {
            auto filename = HintPath + '/' + paths[i];
            auto cur_path = fs::u8path(filename);
            std::error_code ec;
            if(std::filesystem::exists(cur_path, ec))
            {
                oPath = filename;
                return true;
            }
        }
        return false;
    }
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        std::string hint_directory = get_input2<std::string>("HintDirectory");

        auto &ud = prim->userData();
        auto file_path = prim->userData().get2<std::string>("file_path");
        fs::path path = fs::u8path(file_path);
        auto file_path_directory = path.parent_path().u8string();
        if (hint_directory.empty()) {
            hint_directory = file_path_directory;
        }
        auto faceset_count = ud.get2<int>("faceset_count", 0);
        for (auto i = 0; i < faceset_count; i++) {
            auto mat_name = ud.get2<std::string>(format("faceset_{}", i));
            auto content = ud.get2<std::string>(mat_name);
            Json mat_json = Json::parse(content);
            std::vector<std::string> keys;
            for (auto &[key, _]: mat_json.items()) {
                if (zeno::ends_with(key, "_tex")) {
                    keys.push_back(key);
                }
            }
            for (auto &key: keys) {
                std::string tex_path_str = mat_json[key];
                if (key == "diffuse_tex") {

                }
                tex_path_str = zeno::replace_all(tex_path_str, "\\", "/");

                std::string oPath;
                findFile(hint_directory, tex_path_str, oPath);
                mat_json[key] = oPath;
            }
            ud.set2<std::string>(mat_name, mat_json.dump());
        }

        set_output2("prim", prim);
    }
};

ZENDEFNODE(NewFBXResolveTexPath, {
    {
        "prim",
       {"string", "HintDirectory"},
    },
    {
        "prim",
    },
    {},
    {"FBXSDK"},
});

static int GetSkeletonFromBindPose(FbxManager* lSdkManager, FbxScene* lScene, std::shared_ptr<PrimitiveObject>& prim) {
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
    return pose_count;
}

static void TraverseNodesToGetSkeleton(FbxNode* pNode, std::vector<std::string> &bone_names, std::vector<FbxMatrix> &transforms, std::map<std::string, std::string> &parent_mapping) {
    if (!pNode) return;

    FbxMesh* pMesh = pNode->GetMesh();
    if (pMesh && pMesh->GetDeformerCount(FbxDeformer::eSkin)) {
        FbxSkin* pSkin = (FbxSkin*)pMesh->GetDeformer(0, FbxDeformer::eSkin);
        // Iterate over each cluster (bone)
        for (int j = 0; j < pSkin->GetClusterCount(); ++j) {
            FbxCluster* pCluster = pSkin->GetCluster(j);

            FbxNode* pBoneNode = pCluster->GetLink();
            if (!pBoneNode) continue;
            std::string boneName = pBoneNode->GetName();
            if (std::count(bone_names.begin(), bone_names.end(), boneName)) {
                continue;
            }
            bone_names.emplace_back(boneName);
            FbxAMatrix transformLinkMatrix;
            pCluster->GetTransformLinkMatrix(transformLinkMatrix);

            // The transformation of the mesh at binding time
            FbxAMatrix transformMatrix;
            pCluster->GetTransformMatrix(transformMatrix);

            // Inverse bind matrix.
            FbxAMatrix bindMatrix_ = transformMatrix.Inverse() * transformLinkMatrix;
            auto bindMatrix = bit_cast<FbxMatrix>(bindMatrix_);
            transforms.emplace_back(bindMatrix);

            auto pParentNode = pBoneNode->GetParent();
            if (pParentNode) {
                std::string parentName = pParentNode->GetName();
                parent_mapping[boneName] = parentName;
            }
        }
    }

    for (int i = 0; i < pNode->GetChildCount(); i++) {
        TraverseNodesToGetSkeleton(pNode->GetChild(i), bone_names, transforms, parent_mapping);
    }
}
std::shared_ptr<PrimitiveObject> GetSkeletonFromMesh(FbxScene* lScene) {
    auto prim = std::make_shared<PrimitiveObject>();

    FbxNode* lRootNode = lScene->GetRootNode();
    if (lRootNode) {
        std::vector<std::string> bone_names;
        std::vector<FbxMatrix> transforms;
        std::map<std::string, std::string> parent_mapping;
        TraverseNodesToGetSkeleton(lRootNode, bone_names, transforms, parent_mapping);
        std::vector<vec3f> poss;
        std::vector<vec3f> transform_r0;
        std::vector<vec3f> transform_r1;
        std::vector<vec3f> transform_r2;
        for (auto i = 0; i < bone_names.size(); i++) {
            auto bone_name = bone_names[i];
            auto bindMatrix = transforms[i];
            auto t = bindMatrix.GetRow(3);
            poss.emplace_back(t[0], t[1], t[2]);

            auto r0 = bindMatrix.GetRow(0);
            auto r1 = bindMatrix.GetRow(1);
            auto r2 = bindMatrix.GetRow(2);
            transform_r0.emplace_back(r0[0], r0[1], r0[2]);
            transform_r1.emplace_back(r1[0], r1[1], r1[2]);
            transform_r2.emplace_back(r2[0], r2[1], r2[2]);
        }
        prim->verts.resize(bone_names.size());
        prim->verts.values = poss;
        prim->verts.add_attr<vec3f>("transform_r0") = transform_r0;
        prim->verts.add_attr<vec3f>("transform_r1") = transform_r1;
        prim->verts.add_attr<vec3f>("transform_r2") = transform_r2;
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
        auto &boneNames = prim->verts.add_attr<int>("boneName");
        std::iota(boneNames.begin(), boneNames.end(), 0);
        prim->userData().set2("boneName_count", int(bone_names.size()));
        for (auto i = 0; i < bone_names.size(); i++) {
            prim->userData().set2(zeno::format("boneName_{}", i), bone_names[i]);
        }
    }
    return prim;
}
struct NewFBXImportSkeleton : INode {
    virtual void apply() override {
        auto fbx_object = get_input2<FBXObject>("fbx_object");
        auto lSdkManager = fbx_object->lSdkManager;
        auto lScene = fbx_object->lScene;

        // Print the nodes of the scene and their attributes recursively.
        // Note that we are not printing the root node because it should
        // not contain any attributes.
        auto prim = std::make_shared<PrimitiveObject>();

        auto pose_count = GetSkeletonFromBindPose(lSdkManager, lScene, prim);
        if (pose_count == 0 || get_input2<bool>("ForceFromMesh")) {
            prim = GetSkeletonFromMesh(lScene);
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
        "fbx_object",
        {"bool", "ConvertUnits", "0"},
        {"bool", "ForceFromMesh", "0"},
    },
    {
        "prim",
    },
    {},
    {"FBXSDK"},
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
        {"string", "clipName", ""},
        {"frameid"},
        {"float", "fps", "25"},
        {"bool", "ConvertUnits", "0"},
    },
    {
        "prim",
    },
    {},
    {"FBXSDK"},
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
        "fbx_object",
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
    {"FBXSDK"},
});
}
#endif
namespace zeno {
struct RigPoseItemObject : PrimitiveObject {
    std::string boneName;
    bool use_quat = false;
    vec3f translate = {0, 0, 0};
    vec3f rotate = {0, 0, 0};
    vec4f quat = {1,0,0,0};
};
struct NewFBXRigPoseItem : INode {
    virtual void apply() override {
        auto item = std::make_shared<RigPoseItemObject>();
        item->boneName  = get_input2<std::string>("boneName");
        item->translate = get_input2<vec3f>("translate");
        item->rotate    = get_input2<vec3f>("rotate");
        item->use_quat  = get_input2<bool>("UseQuaternion");
        item->quat = get_input2<vec4f>("quaternion");
        set_output2("poseItem", std::move(item));
    }
};

ZENDEFNODE(NewFBXRigPoseItem, {
    {
        {"string", "boneName", ""},
        {"bool", "UseQuaternion", "0"},
        {"vec3f", "translate", "0, 0, 0"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"vec4f", "quaternion", "1,0,0,0"},
    },
    {
        "poseItem",
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
struct BoneSetAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("skeleton");
        auto value = get_input<NumericObject>("value");
        auto attr = get_input2<std::string>("attr");
        auto type = get_input2<std::string>("type");
        auto boneName = get_input2<std::string>("boneName");
        auto boneNameMapping = getBoneNameMapping(prim.get());
        auto index = boneNameMapping[boneName];

        std::visit(
            [&](auto ty) {
              using T = decltype(ty);

              auto val = value->get<T>();
              auto &attr_arr = prim->add_attr<T>(attr);
              if (index < attr_arr.size()) {
                  attr_arr[index] = val;
              }
            },
            enum_variant<std::variant<float, vec2f, vec3f, vec4f, int, vec2i, vec3i, vec4i>>(
                array_index({"float", "vec2f", "vec3f", "vec4f", "int", "vec2i", "vec3i", "vec4i"}, type)));

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(BoneSetAttr,
{ /* inputs: */ {
    "skeleton",
    {"string", "boneName", ""},
    {"int", "value", "0"},
    {"string", "attr", ""},
    {"enum float vec2f vec3f vec4f int vec2i vec3i vec4i", "type", "int"},
}, /* outputs: */ {
   "prim",
}, /* params: */ {
}, /* category: */ {
   "FBXSDK",
}});
struct BoneGetAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("skeleton");
        auto attr = get_input2<std::string>("attr");
        auto type = get_input2<std::string>("type");
        auto boneName = get_input2<std::string>("boneName");
        auto boneNameMapping = getBoneNameMapping(prim.get());
        auto index = boneNameMapping[boneName];

        auto value = std::make_shared<NumericObject>();

        std::visit(
            [&](auto ty) {
              using T = decltype(ty);
              auto &attr_arr = prim->attr<T>(attr);
              if (index < attr_arr.size()) {
                  value->set<T>(attr_arr[index]);
              }
            },
            enum_variant<std::variant<float, vec2f, vec3f, vec4f, int, vec2i, vec3i, vec4i>>(
                array_index({"float", "vec2f", "vec3f", "vec4f", "int", "vec2i", "vec3i", "vec4i"}, type)));

        set_output("value", std::move(value));
    }
};
ZENDEFNODE(BoneGetAttr,
{ /* inputs: */ {
    "skeleton",
    {"string", "boneName", ""},
    {"string", "attr", ""},
    {"enum float vec2f vec3f vec4f int vec2i vec3i vec4i", "type", "int"},
}, /* outputs: */ {
   "value",
}, /* params: */ {
}, /* category: */ {
   "FBXSDK",
}});
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
                glm::mat4 matRotx  = glm::rotate(glm::radians(trans->rotate[0]), glm::vec3(1,0,0) );
                glm::mat4 matRoty  = glm::rotate(glm::radians(trans->rotate[1]), glm::vec3(0,1,0) );
                glm::mat4 matRotz  = glm::rotate(glm::radians(trans->rotate[2]), glm::vec3(0,0,1) );
                glm::mat4 matRot = matRoty*matRotx*matRotz;
                if(trans->use_quat==true)
                {
                  glm::quat rot(trans->quat[3], trans->quat[0], trans->quat[1], trans->quat[2]);
                  matRot = glm::toMat4(rot);
                }
                transform = matTrans*matRot;
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
        "skeleton",
        {"bool", "WorldSpace", "0"},
        {"list", "Transformations"},
    },
    {
        "skeleton",
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
        "GeometryToDeform",
        "RestPointTransforms",
        "DeformPointTransforms",
        {"enum Linear DualQuaternion", "SkinningMethod", "Linear"},
        {"string", "vectors", "nrm,"},
    },
    {
        "prim",
    },
    {},
    {"FBXSDK"},
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
        "RestPointTransforms",
        "DeformPointTransforms",
    },
    {
        "keyframe",
    },
    {},
    {"FBXSDK"},
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
        "skeleton",
        "keyframe",
    },
    {
        "DeformPointTransforms",
    },
    {},
    {"FBXSDK"},
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
        "prim",
        {"string", "params", "transform_r0,transform_r1,transform_r2"},
    },
    {
        "output",
    },
    {},
    {"debug"},
});

struct IKChainsItemObject : PrimitiveObject {
    std::string RootName;
    std::string MidName;
    std::string TipName;
    bool MatchByName = true;
    std::string TwistName;
    std::string GoalName;
    float Blend = 1;
    bool OrientTip = true;
};
struct IKChainsItem : INode {
    virtual void apply() override {
        auto item = std::make_shared<IKChainsItemObject>();
        item->RootName  = get_input2<std::string>("RootName");
        item->MidName  = get_input2<std::string>("MidName");
        item->TipName  = get_input2<std::string>("TipName");
        item->MatchByName  = get_input2<bool>("MatchByName");
        item->TwistName  = get_input2<std::string>("TwistName");
        item->OrientTip  = get_input2<bool>("OrientTip");

        set_output2("poseItem", std::move(item));
    }
};

ZENDEFNODE(IKChainsItem, {
    {
        {"string", "RootName", ""},
        {"string", "MidName", ""},
        {"string", "TipName", ""},
        {"bool", "MatchByName", "1"},
        {"string", "TwistName", ""},
        {"string", "GoalName", ""},
        {"bool", "OrientTip", "0"},
    },
    {
        "poseItem",
    },
    {},
    {"deprecated"},
});

float sqr(float v) {
    return v * v;
}
// return: mid, tip
std::pair<vec3f, vec3f> twoBoneIK(
    vec3f root
    , vec3f joint
    , vec3f end
    , vec3f jointTarget
    , vec3f effector
) {
        vec3f output_joint = {};
        vec3f output_end = {};

        auto root_to_effect = effector - root;
        auto root_to_jointTarget = jointTarget - root;

        auto upper_limb_length = zeno::length(root - joint);
        auto lower_limb_length = zeno::length(joint - end);
        auto desired_length = zeno::length(root_to_effect);
        if (desired_length < abs(upper_limb_length - lower_limb_length)) {
            zeno::log_info("A");
            output_joint = root + normalize(root_to_effect) * abs(upper_limb_length - lower_limb_length);
            output_end = root + normalize(root_to_effect) * upper_limb_length;
        }
        else if (desired_length > upper_limb_length + lower_limb_length) {
            zeno::log_info("B");

            output_joint = root + normalize(root_to_effect) * upper_limb_length;
            output_end = root + normalize(root_to_effect) * (upper_limb_length + lower_limb_length);
        }
        else {
            zeno::log_info("C");

            vec3f to_pole = normalize(cross(cross(root_to_effect, root_to_jointTarget), root_to_effect));
            float cos_theta = (sqr(upper_limb_length) + sqr(desired_length) - sqr(lower_limb_length)) / (2.0f * upper_limb_length * desired_length);
            float sin_theta = sqrt(1 - sqr(cos_theta));
            output_joint = root + (normalize(root_to_effect) * cos_theta + to_pole * sin_theta) * upper_limb_length;
            output_end = effector;
        }

    return {output_joint, output_end};
}

struct IKChains : INode {
    virtual void apply() override {
        auto skeleton = get_input2<PrimitiveObject>("Skeleton");
        auto ikDrivers = get_input2<PrimitiveObject>("IK Drivers");
        auto items = get_input<zeno::ListObject>("items")->getRaw<IKChainsItemObject>();
        auto skeletonBoneNameMapping = getBoneNameMapping(skeleton.get());
        auto ikDriversBoneNameMapping = getBoneNameMapping(ikDrivers.get());
        std::map<int, int> bone_connects;
        for (auto i = 0; i < skeleton->polys.size(); i++) {
            bone_connects[skeleton->loops[i * 2 + 1]] = skeleton->loops[i * 2];
        }
        auto ordering = TopologicalSorting(bone_connects, skeleton.get());

        auto &verts = skeleton->verts;
        auto &transform_r0 = skeleton->verts.attr<vec3f>("transform_r0");
        auto &transform_r1 = skeleton->verts.attr<vec3f>("transform_r1");
        auto &transform_r2 = skeleton->verts.attr<vec3f>("transform_r2");

        for (auto item: items) {
            std::string TwistName = item->MatchByName? item->MidName: item->TwistName;
            std::string GoalName = item->MatchByName? item->TipName: item->GoalName;
            auto root_index = skeletonBoneNameMapping[item->RootName];
            vec3f root = skeleton->verts[root_index];
            auto joint_index = skeletonBoneNameMapping[item->MidName];
            vec3f joint = skeleton->verts[joint_index];
            auto end_index = skeletonBoneNameMapping[item->TipName];
            vec3f end = skeleton->verts[end_index];
            vec3f jointTarget = ikDrivers->verts[ikDriversBoneNameMapping[TwistName]];
            vec3f effector = ikDrivers->verts[ikDriversBoneNameMapping[GoalName]];
            auto [midPos, tipPos] = twoBoneIK(root, joint, end, jointTarget, effector);
            auto parent = glm::rotation(bit_cast<glm::vec3>(normalize(joint - root)), bit_cast<glm::vec3>(normalize(midPos - root)));
            auto from_ = parent * bit_cast<glm::vec3>(normalize(end - joint));
            auto child = glm::rotation(from_, bit_cast<glm::vec3>(normalize(tipPos - midPos)));
            bool start = false;
            std::map<int, glm::mat4> cache;
            for (auto bi: ordering) {
                if (bi == root_index) {
                    start = true;
                }
                if (start) {
                    glm::mat4 transform = glm::mat4(1.0f);
                    if (bi == root_index) {
                        transform = glm::translate(bit_cast<glm::vec3>(verts[bi])) * glm::toMat4(parent) * glm::translate(-bit_cast<glm::vec3>(verts[bi]));
                    }
                    else if (bi == joint_index) {
                        transform = glm::translate(bit_cast<glm::vec3>(verts[bi])) * glm::toMat4(child) * glm::translate(-bit_cast<glm::vec3>(verts[bi]));
                    }
                    if (bone_connects.count(bi) && cache.count(bone_connects[bi])) {
                        transform = cache[bone_connects[bi]] * transform;
                    }
                    if (bi == end_index && item->OrientTip) {
                        auto target_pos = transform_pos(transform, verts[bi]);
                        transform = glm::translate(bit_cast<glm::vec3>(target_pos - verts[bi]));
                    }
                    cache[bi] = transform;
                    verts[bi]        = transform_pos(transform, verts[bi]);
                    transform_r0[bi] = transform_nrm(transform, transform_r0[bi]);
                    transform_r1[bi] = transform_nrm(transform, transform_r1[bi]);
                    transform_r2[bi] = transform_nrm(transform, transform_r2[bi]);
                }
            }
        }

        set_output("Skeleton", skeleton);
    }
};

ZENDEFNODE(IKChains, {
    {
        "Skeleton",
        "IK Drivers",
        {"list", "items"},
    },
    {
        "Skeleton",
    },
    {},
    {"deprecated"},
});

float length(std::vector<float> &b)
{
    float l = 0;
    for(int i=0;i<b.size();i++)
    {
        l += b[i] * b[i];
    }
    return sqrt(l);
}
//
void GaussSeidelSolve(std::vector<std::vector<float>> &A, std::vector<float> &b, std::vector<float>&x,
                      int max_iter, float tol)
    {
        int iter=0;
        float b_nrm = 0;
        for(int i=0;i<b.size();i++)
        {
            b_nrm = max(b_nrm, abs(b[i]));
        }
        if(b_nrm<=0.00001)
            return;
        while(iter<max_iter)
        {
//            std::cout<<"solving"<<std::endl;
            float e_max = 0;
            for(int i=0;i<x.size();i++)
            {
                float e = b[i];
                for(int j=0;j<A[i].size();j++)
                {
                    e -= A[i][j] * x[j];
                }
                e_max = max(e_max, abs(e));
                x[i] += e / A[i][i];
            }

            if(e_max/b_nrm<tol) {
//                std::cout<<"iter:"<<iter<<", err:"<<e_max/b_nrm<<std::endl;
                break;
            }
            iter++;
        }
    }
vec3f getJointPos(int id, PrimitiveObject * skel_ptr) {
    return skel_ptr->verts[id];
}
void computeJointJacobian(std::vector<int> &index,
                          std::vector<vec3f> &J,
                          std::vector<vec3f> &r,
                          PrimitiveObject * skel_ptr,
                          vec3f e_curr
                          )
{
    J.resize(index.size() * 3);
    for(int i=0;i<index.size();i++)
    {
        int id = index[i];
        auto p_i = getJointPos(id, skel_ptr);
        for (auto j = 0; j < 3; j++) {
            auto r_i = r[id * 3 + j];
            auto dedtheta_i = cross(r_i, e_curr - p_i);
            J[i * 3 + j] = dedtheta_i;
        }
    }
}
void computeJTJ(std::vector<vec3f> &J, std::vector<std::vector<float>> &JTJ, float alpha)
{
    JTJ.resize(J.size());
    for(int i=0;i<J.size();i++)
    {
        JTJ[i].resize(J.size());
    }
    for(int i=0;i<J.size();i++)
    {
        for(int j=0;j<J.size();j++)
        {
            JTJ[i][j] = dot(J[i], J[j]);
        }
        JTJ[i][i] += alpha;
    }
    for(int i=0;i<JTJ.size();i++)
    {
        float row_sum = 0;
        for(int j=0;j<JTJ[i].size();j++)
        {
            if(j!=i)
                row_sum += abs(JTJ[i][j]);
        }
        if(abs(JTJ[i][i])<row_sum)
            JTJ[i][i] = glm::sign(JTJ[i][i]) * row_sum;
    }
}
std::shared_ptr<PrimitiveObject> FK(
    std::vector<float> theta
    , std::shared_ptr<PrimitiveObject> skel_ptr
) {
        std::vector<glm::mat4> Transformations;
        for (auto i = 0; i < skel_ptr->verts.size(); i++) {
            auto mx = glm::rotate(theta[i * 3 + 0], glm::vec3(1, 0, 0));
            auto my = glm::rotate(theta[i * 3 + 1], glm::vec3(0, 1, 0));
            auto mz = glm::rotate(theta[i * 3 + 2], glm::vec3(0, 0, 1));
            auto Transformation = mx * my * mz;
            Transformations.push_back(Transformation);
        }

        auto skeleton = std::dynamic_pointer_cast<PrimitiveObject>(skel_ptr->clone());
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
            auto trans = Transformations[bi];
            transform = transforms[bi] * trans * transformsInv[bi];
            if (bone_connects.count(bi)) {
                transform = cache[bone_connects[bi]] * transform;
            }
            cache[bi] = transform;
            verts[bi]        = transform_pos(transform, verts[bi]);
            transform_r0[bi] = transform_nrm(transform, transform_r0[bi]);
            transform_r1[bi] = transform_nrm(transform, transform_r1[bi]);
            transform_r2[bi] = transform_nrm(transform, transform_r2[bi]);
        }
    return skeleton;
}
void solveJointUpdate(int id,
                      vec3f tarPos,
                      std::shared_ptr<PrimitiveObject> skel_ptr,
                      std::vector<int> &index,
                      std::vector<float> &dtheta,
                      std::vector<float> &theta,
                      float &dist,
                      float scale)
{
    dtheta.resize(theta.size());
    dtheta.assign(dtheta.size(), 0);
//    zeno::log_error("{} FK.....", id);
    std::shared_ptr<PrimitiveObject> skeleton = FK(theta, skel_ptr);
//    zeno::log_error("{} FK----------", id);
    vec3f e_curr = getJointPos(id, skeleton.get());
    vec3f de = tarPos - e_curr;
    dist = glm::length(bit_cast<glm::vec3>(de));
    if(dist / scale <0.0001)
        return;
    std::vector<vec3f> r;
    {
        auto &transform_r0 = skeleton->verts.attr<vec3f>("transform_r0");
        auto &transform_r1 = skeleton->verts.attr<vec3f>("transform_r1");
        auto &transform_r2 = skeleton->verts.attr<vec3f>("transform_r2");
        for (auto i = 0; i < skeleton->verts.size(); i++) {
            r.push_back(zeno::normalize(transform_r0[i]));
            r.push_back(zeno::normalize(transform_r1[i]));
            r.push_back(zeno::normalize(transform_r2[i]));
        }
    }
    std::vector<vec3f> J;
    computeJointJacobian(index, J, r, skel_ptr.get(), e_curr);
    if (0) {
        // log
        auto boneNames = getBoneNames(skel_ptr.get());
        for (auto i = 0; i < index.size(); i++) {
            auto idx = index[i];
            std::cout << boneNames[idx] << " : ";
            std::cout << J[i * 3 + 0][0] << ", " << J[i * 3 + 0][1] << ", " << J[i * 3 + 0][2] << "; ";
            std::cout << J[i * 3 + 1][0] << ", " << J[i * 3 + 1][1] << ", " << J[i * 3 + 1][2] << "; ";
            std::cout << J[i * 3 + 2][0] << ", " << J[i * 3 + 2][1] << ", " << J[i * 3 + 2][2] << "; ";
        }
    }

//    zeno::log_error("computeJointJacobian");
    std::vector<std::vector<float>> JTJ;
    computeJTJ(J, JTJ, 0.001);
    for (auto i = 0; i < JTJ.size(); i++) {
//        std::cout << JTJ[i][i] << ' ';
    }
//    zeno::log_error("computeJTJ");
    auto b = std::vector<float>(index.size() * 3);
    for(int i=0;i<b.size();i++)
    {
        b[i] = dot(J[i], de);
    }
    std::vector<float> x(index.size() * 3);
    GaussSeidelSolve(JTJ, b, x, 100, 0.00001);
//    zeno::log_error("GaussSeidelSolve");
    for (auto i = 0; i < x.size(); i++) {
//        std::cout << x[i] << ' ';
    }
    for(int i=0;i<index.size();i++)
    {
        for (auto j = 0; j < 3; j++) {
            dtheta[index[i] * 3 + j] = x[i * 3 +j];
        }
    }
//    zeno::log_error("return..........");
}

std::vector<int> getIds(int endId, int depth, PrimitiveObject* skeletonPtr) {
    std::map<int, int> connects;
    auto count = skeletonPtr->loops.size() / 2;
    for (auto i = 0; i < count; i++) {
        auto parentId = skeletonPtr->loops[2 * i + 0];
        auto childId = skeletonPtr->loops[2 * i + 1];
        connects[childId] = parentId;
    }
    std::vector<int> result = {endId};
    auto cur_id = endId;
    for (auto i = 0; i < depth; i++) {
        if (connects.count(cur_id) == 0) {
            break;
        }
        cur_id = connects[cur_id];
        result.push_back(cur_id);
    }

    return result;
}

float computeError(int id, std::shared_ptr<PrimitiveObject> skeletion, vec3f targetPos
                 , std::vector<float> & theta) {
    auto curPose = FK(theta, skeletion);
    auto curJointPos = getJointPos(id, curPose.get());
    return zeno::distance(curJointPos, targetPos);
}

float proposeTheta(std::vector<int> &ids, std::shared_ptr<PrimitiveObject> skeletion, std::vector<vec3f> &targetPoss
                  , std::vector<float> & new_theta, std::vector<float> & theta, std::vector<float> & dtheta, std::vector<float> & total_theta
                  , std::vector<vec2f> &limit, float alpha, std::vector<float> w) {

    for (int i = 0; i < theta.size(); ++i) {
        new_theta[i] = theta[i] + alpha * dtheta[i] / (w[i] > 0 ? w[i] : 1);
        auto tmp_theta = clamp(total_theta[i] + new_theta[i], limit[i][0], limit[i][1]);
        new_theta[i] = tmp_theta - total_theta[i];
    }
    float e = 0;
    for (auto i = 0; i < ids.size(); i++) {
        e += computeError(ids[i], skeletion, targetPoss[i], new_theta);
    }
    return e;
}


void line_search(std::vector<int> &ids, std::shared_ptr<PrimitiveObject> skeletion, std::vector<vec3f> &targetPoss
                 , std::vector<float> & theta, std::vector<float> & dtheta, std::vector<float> & total_theta
                  , std::vector<vec2f> &limit, float damp, std::vector<float> &w, float prev_err) {
    std::vector<float> new_theta = theta;
    float alpha = 1;

    float e;
    e = proposeTheta(ids, skeletion, targetPoss, new_theta, theta, dtheta, total_theta, limit, alpha, w);

    if (e < prev_err) {
        theta = new_theta;
        return;
    }
    while (alpha > 1e-7) {
        alpha *= damp;
        e = proposeTheta(ids, skeletion, targetPoss, new_theta, theta, dtheta, total_theta, limit, alpha, w);
        if (e < prev_err) {
            theta = new_theta;
            return;
        }
    }
}

void SolveIKConstrained(std::shared_ptr<PrimitiveObject> skeletonPtr,
                        std::vector<float> & theta,
                        std::vector<float> & total_theta,
                        std::vector<vec2f> & theta_constraints,
                        std::vector<vec3f> &targets,
                        std::vector<int> endEffectorIDs,
                        std::vector<int> depths,
                        int iter_max
                        )
{
    std::vector<std::vector<float>> dtheta;
    dtheta.resize(endEffectorIDs.size());
    int iter = 0;
    std::vector<float> old_theta;
    old_theta = theta;
    float prev_err = INFINITY;
    float scale = 1;
    {
        for (int i = 0; i < endEffectorIDs.size(); i++) {
            auto endId = endEffectorIDs[i];
            auto depth = depths[i];
            std::vector<int> index = getIds(endId, depth, skeletonPtr.get());
            auto tarPos = targets[i];
            float e_i;
            solveJointUpdate(endId, tarPos, skeletonPtr, index, dtheta[i], theta, e_i, 1);
            scale = max(scale, e_i);
        }
    }
    while(iter<iter_max) {
//        zeno::log_error("iter: {}", iter);
        iter++;
        float err = 0;
        for (int i = 0; i < endEffectorIDs.size(); i++) {
//            zeno::log_error("i: {}", i);
            auto endId = endEffectorIDs[i];
            auto depth = depths[i];
            std::vector<int> index = getIds(endId, depth, skeletonPtr.get());
            auto tarPos = targets[i];
            float e_i;
            solveJointUpdate(endId, tarPos, skeletonPtr, index, dtheta[i], theta, e_i, scale);
            err += e_i;
        }
        prev_err = err;
//        std::cout<<"current err:"<<err<<std::endl;
        if (err / scale < 0.0001)
            break;
        std::vector<float> w;
        w.resize(theta.size());
        w.assign(w.size(), 0);
        std::vector<float> total_dtheta;
        total_dtheta.resize(theta.size());
        total_dtheta.assign(total_dtheta.size(), 0);

        for(int j=0;j<endEffectorIDs.size();j++)
            for (int i = 0; i < theta.size(); i++) {
                total_dtheta[i] += dtheta[j][i];
                w[i] += abs(dtheta[j][i]) > 0 ? 1 : 0;
            }

        float damp = 0.5;
        line_search(endEffectorIDs, skeletonPtr, targets , theta, total_dtheta, total_theta
                  , theta_constraints, damp, w, prev_err);
        if (0) {
            // log
            auto boneNames = getBoneNames(skeletonPtr.get());
            for (auto i = 0; i < skeletonPtr->verts.size(); i++) {
                std::cout << boneNames[i] << " : ";
                std::cout << total_dtheta[i * 3 + 0] << ", ";
                std::cout << total_dtheta[i * 3 + 1] << ", ";
                std::cout << total_dtheta[i * 3 + 2] << ", " << std::endl;
            }
        }

//        float max_dtheta = 0;
//        for(int i=0;i<theta.size();i++)
//        {
//            max_dtheta = max(abs(old_theta[i] - theta[i]), max_dtheta);
//        }
//        if(max_dtheta<0.0001) break;
//
//        old_theta = theta;
    }
}

struct IkChainsItemObject : PrimitiveObject {
    int depth;
    std::string endEffectorName;
    vec3f targetPos;
};

struct IkChainsItem : INode {
    virtual void apply() override {
        auto item = std::make_shared<IkChainsItemObject>();
        item->depth = get_input2<int>("depth");
        item->endEffectorName = get_input2<std::string>("endEffectorName");
        item->targetPos = get_input2<vec3f>("targetPos");

        set_output2("IkChain", std::move(item));
    }
};

ZENDEFNODE(IkChainsItem, {
    {
        {"string", "endEffectorName", ""},
        {"int", "depth", "2"},
        {"vec3f", "targetPos", ""},
    },
    {
        "IkChain",
    },
    {},
    {"FBXSDK"},
});

struct JointLimitObject : PrimitiveObject {
    std::string boneName;
    vec3i enableLimit;
    vec2f xLimit;
    vec2f yLimit;
    vec2f zLimit;
};

struct JointLimitItem : INode {
    virtual void apply() override {
        auto item = std::make_shared<JointLimitObject>();
        item->boneName = get_input2<std::string>("boneName");
        item->enableLimit = {
            get_input2<int>("enableXLimit"),
            get_input2<int>("enableYLimit"),
            get_input2<int>("enableZLimit"),
        };
        item->xLimit = get_input2<vec2f>("xLimit");
        item->yLimit = get_input2<vec2f>("yLimit");
        item->zLimit = get_input2<vec2f>("zLimit");

        set_output2("JointLimit", std::move(item));
    }
};

ZENDEFNODE(JointLimitItem, {
    {
        {"string", "boneName", ""},
        {"bool", "enableXLimit", "0"},
        {"vec2f", "xLimit", "0,0"},
        {"bool", "enableYLimit", "0"},
        {"vec2f", "yLimit", "0,0"},
        {"bool", "enableZLimit", "0"},
        {"vec2f", "zLimit", "0,0"},
    },
    {
        "JointLimit",
    },
    {},
    {"FBXSDK"},
});

struct IkSolver : INode {
    void apply() override {
        auto skeleton = get_input2<PrimitiveObject>("Skeleton");
        auto boneNameMapping = getBoneNameMapping(skeleton.get());
        int iter_max = get_input2<int>("iterCount");
        auto &enableXYZLimit = skeleton->add_attr<vec3i>("enableXYZLimit");
        auto &xLimit = skeleton->add_attr<vec2f>("xLimit");
        auto &yLimit = skeleton->add_attr<vec2f>("yLimit");
        auto &zLimit = skeleton->add_attr<vec2f>("zLimit");
        if (has_input("jointLimits")) {
            auto items = get_input<zeno::ListObject>("jointLimits")->getRaw<JointLimitObject>();
            for (auto &item: items) {
                if (boneNameMapping.count(item->boneName)) {
                    auto index = boneNameMapping[item->boneName];
                    enableXYZLimit[index] = item->enableLimit;
                    xLimit[index] = item->xLimit;
                    yLimit[index] = item->yLimit;
                    zLimit[index] = item->zLimit;
                }
                else {
                    zeno::log_warn("joint limit: joint {} missing", item->boneName);
                }
            }
        }
        std::vector<float> theta;
        std::vector<vec2f> theta_constraints;
        {
            theta.resize(skeleton->verts.size() * 3);
            for (auto i = 0; i < skeleton->verts.size(); i++) {
                theta_constraints.push_back(enableXYZLimit[i][0]? xLimit[i]: vec2f(-INFINITY, INFINITY));
                theta_constraints.push_back(enableXYZLimit[i][1]? yLimit[i]: vec2f(-INFINITY, INFINITY));
                theta_constraints.push_back(enableXYZLimit[i][2]? zLimit[i]: vec2f(-INFINITY, INFINITY));
            }
        }
        std::vector<vec3f> targets;
        std::vector<int> endEffectorIDs;
        std::vector<int> depths;
        {
            auto items = get_input<zeno::ListObject>("IkChains")->getRaw<IkChainsItemObject>();
            for (auto &item: items) {
                if (boneNameMapping.count(item->endEffectorName) == 0) {
                    log_warn("Not find ik endEffector: {}", item->endEffectorName);
                    continue;
                }
                endEffectorIDs.push_back(boneNameMapping[item->endEffectorName]);
                depths.push_back(item->depth);
                targets.push_back(item->targetPos);
            }
        }

        auto &total_theta_3 = skeleton->verts.attr<vec3f>("TotalTheta");
        std::vector<float> total_theta(skeleton->verts.size() * 3);
        for (auto i = 0; i < skeleton->verts.size(); i++) {
            total_theta[i*3 + 0] = total_theta_3[i][0];
            total_theta[i*3 + 1] = total_theta_3[i][1];
            total_theta[i*3 + 2] = total_theta_3[i][2];
        }
        SolveIKConstrained(
            skeleton,
            theta,
            total_theta,
            theta_constraints,
            targets,
            endEffectorIDs,
            depths,
            iter_max
            );
        std::shared_ptr<PrimitiveObject> out_skeleton = FK(theta, skeleton);
        {
            auto &total_theta = out_skeleton->verts.attr<vec3f>("TotalTheta");
            for (auto i = 0; i < out_skeleton->verts.size(); i++) {
                total_theta[i][0] += theta[i * 3 + 0];
                total_theta[i][1] += theta[i * 3 + 1];
                total_theta[i][2] += theta[i * 3 + 2];
            }
        }
        set_output2("Skeleton", out_skeleton);
    }
};
ZENDEFNODE(IkSolver, {
    {
        "Skeleton",
        {"int", "iterCount", "50"},
        {"list", "IkChains"},
        {"list", "jointLimits"},
    },
    {
        "Skeleton",
    },
    {},
    {"FBXSDK"},
});

struct IkJointConstraints : INode {
    void apply() override {
        auto skeleton = get_input2<PrimitiveObject>("Skeleton");
        auto boneNameMapping = getBoneNameMapping(skeleton.get());
        auto rest_skeleton = get_input2<PrimitiveObject>("RestSkeleton");
        std::vector<vec3i> enableXYZLimit(skeleton->verts.size());
        std::vector<vec2f> xLimit(skeleton->verts.size());
        std::vector<vec2f> yLimit(skeleton->verts.size());
        std::vector<vec2f> zLimit(skeleton->verts.size());
        if (has_input("jointLimits")) {
            auto items = get_input<zeno::ListObject>("jointLimits")->getRaw<JointLimitObject>();
            for (auto &item: items) {
                if (boneNameMapping.count(item->boneName)) {
                    auto index = boneNameMapping[item->boneName];
                    enableXYZLimit[index] = item->enableLimit;
                    xLimit[index] = item->xLimit;
                    yLimit[index] = item->yLimit;
                    zLimit[index] = item->zLimit;
                }
            }
        }
        auto &total_theta_3 = skeleton->verts.attr<vec3f>("TotalTheta");
        for (auto i = 0; i < skeleton->verts.size(); i++) {
            if (enableXYZLimit[i][0]) {
                total_theta_3[i][0] = clamp(total_theta_3[i][0], xLimit[i][0], xLimit[i][1]);
            }
            if (enableXYZLimit[i][1]) {
                total_theta_3[i][1] = clamp(total_theta_3[i][1], yLimit[i][0], yLimit[i][1]);
            }
            if (enableXYZLimit[i][2]) {
                total_theta_3[i][2] = clamp(total_theta_3[i][2], zLimit[i][0], zLimit[i][1]);
            }
        }

        std::vector<float> total_theta(skeleton->verts.size() * 3);
        for (auto i = 0; i < skeleton->verts.size(); i++) {
            total_theta.push_back(total_theta_3[i][0]);
            total_theta.push_back(total_theta_3[i][1]);
            total_theta.push_back(total_theta_3[i][2]);

        }
        std::shared_ptr<PrimitiveObject> out_skeleton = FK(total_theta, skeleton);
        set_output2("Skeleton", out_skeleton);
    }
};
ZENDEFNODE(IkJointConstraints, {
    {
        "Skeleton",
        "RestSkeleton",
        {"list", "jointLimits"},
    },
    {
        "Skeleton",
    },
    {},
    {"FBXSDK"},
});

struct PrimBindOneBone : INode {
    void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        prim->userData().set2("boneName_count", 1);
        prim->userData().set2("boneName_0", get_input2<std::string>("boneName"));
        auto &boneName_0 = prim->add_attr<int>("boneName_0");
        std::fill(boneName_0.begin(), boneName_0.end(), 0);
        auto &boneWeight_0 = prim->add_attr<float>("boneWeight_0");
        std::fill(boneWeight_0.begin(), boneWeight_0.end(), 1.0f);
        set_output2("prim", prim);
    }
};
ZENDEFNODE(PrimBindOneBone, {
    {
        "prim",
        {"string", "boneName", ""},
    },
    {
        "prim",
    },
    {},
    {"FBXSDK"},
});

struct PrimDeformByOneBone : INode {
    void apply() override {
        auto prim = get_input2<PrimitiveObject>("prim");
        auto skeleton = get_input2<PrimitiveObject>("skeleton");
        auto boneName = get_input2<std::string>("boneName");
        auto useCustomPivot = get_input2<bool>("useCustomPivot");
        auto pivot = get_input2<vec3f>("pivot");
        if (useCustomPivot == false) {
            auto outs = zeno::TempNodeSimpleCaller("PrimReduction")
                    .set2("prim", prim)
                    .set2("attrName", "pos")
                    .set2("op", "avg")
                    .call();
            pivot = outs.get2<zeno::vec3f>("result");
        }
        auto eularAngleXYZ = bit_cast<glm::vec3>(get_input2<vec3f>("rotation"));
        glm::mat4 matRotate = EulerAngle::rotate(EulerAngle::RotationOrder::YXZ, EulerAngle::Measure::Degree, eularAngleXYZ);
        glm::mat4 matTrans = glm::translate(-bit_cast<glm::vec3>(pivot));
        auto nameMapping = getBoneNameMapping(skeleton.get());
        auto boneMatrix = getBoneMatrix(skeleton.get());
        glm::mat4 transform(1);
        if (nameMapping.count(boneName)) {
            auto mat = boneMatrix[nameMapping[boneName]];
            if (get_input2<bool>("inheritRotation")) {
                mat[0] = glm::normalize(mat[0]);
                mat[1] = glm::normalize(mat[1]);
                mat[2] = glm::normalize(mat[2]);
            }
            else {
                mat[0] = {1, 0, 0, 0};
                mat[1] = {0, 1, 0, 0};
                mat[2] = {0, 0, 1, 0};
            }
            transform = mat * matTrans * matRotate;
            auto vert_count = prim->verts.size();
            #pragma omp parallel for
            for (auto i = 0; i < vert_count; i++) {
                prim->verts[i] = transform_pos(transform, prim->verts[i]);
            }
            if (prim->verts.attr_is<vec3f>("nrm")) {
                auto &nrms = prim->verts.attr<vec3f>("nrm");
                #pragma omp parallel for
                for (auto i = 0; i < vert_count; i++) {
                    nrms[i] = transform_nrm(transform, nrms[i]);
                }
            }
        }
        set_output2("prim", prim);
    }
};
ZENDEFNODE(PrimDeformByOneBone, {
    {
        "prim",
        "skeleton",
        {"string", "boneName", ""},
        {"bool", "useCustomPivot", "0"},
        {"bool", "inheritRotation", "0"},
        {"vec3f", "pivot", "0, 0, 0"},
        {"vec3f", "rotation", "0, 0, 0"},
    },
    {
        "prim",
    },
    {},
    {"FBXSDK"},
});



}