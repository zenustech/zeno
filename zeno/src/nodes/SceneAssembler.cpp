//
// Created by zh on 2025/4/14.
//
#include <zeno/extra/GlobalComm.h>
#include <zeno/types/DummyObject.h>
#include <zeno/core/Graph.h>
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/ListObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/fileio.h"
#include "zeno/utils/log.h"
#include "zeno/utils/string.h"
#include "zeno/utils/scope_exit.h"
#include <deque>
#include <string>
#include <tinygltf/json.hpp>
#include <zeno/zeno.h>
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <zeno/utils/eulerangle.h>
#include "zeno/extra/SceneAssmbler.h"

namespace zeno {
#if 0
static std::string get_parent_path(const std::string& path) {
    if (path.empty() || path == "/") {
        return path;  // 根目录的父目录是它自己
    }

    // 移除末尾的斜杠（如果有）
    std::string normalized = path;
    if (normalized.back() == '/') {
        normalized.pop_back();
    }

    // 查找最后一个斜杠
    auto last_slash = normalized.find_last_of('/');

    if (last_slash == std::string::npos) {
        return "/";  // 没有斜杠，返回根目录
    }

    if (last_slash == 0) {
        return "/";  // 已经是根目录的直接子目录
    }

    return normalized.substr(0, last_slash);
}
std::shared_ptr<zeno::ListObject> scene_tree_to_structure(SceneObject* sceneSource) {
    std::unordered_map<std::string, glm::mat4> global_matrices;
    std::unordered_map<std::string, std::vector<std::shared_ptr<PrimitiveObject>>> temp_matrices;
    global_matrices["/ABC"] = glm::mat4(1);
    for (const auto &path: sceneSource->matrices) {
        if (path == "/ABC") {
            continue;
        }
        auto parent_path = get_parent_path(path);
        if (global_matrices.count(parent_path) == 0) {
            zeno::log_error("in {}, parent_matrix {} is missing", path, parent_path);
        }
        auto parent_matrix = global_matrices.at(parent_path);
        if (sceneSource->matrices.count(path) == 0) {
            zeno::log_error("in {}, local_matrix {} is missing", path, parent_path);
        }
        auto local_matrix = sceneSource->matrices.at(path);
        if (sceneSource->node_visibility.count(path)) {
            if (sceneSource->node_visibility.at(path) == 0) {
                local_matrix = glm::mat4(0);
            }
        }
        auto global_matrix = parent_matrix * local_matrix;
        auto r0 = global_matrix[0];
        auto r1 = global_matrix[1];
        auto r2 = global_matrix[2];
        auto t  = global_matrix[3];
        global_matrices[path] = global_matrix;
        auto prim = std::make_shared<PrimitiveObject>();
        prim->verts.resize(1);
        prim->verts[0] = {t[0], t[1], t[2]};
        prim->verts.add_attr<vec3f>("r0")[0] = {r0[0], r0[1], r0[2]};
        prim->verts.add_attr<vec3f>("r1")[0] = {r1[0], r1[1], r1[2]};
        prim->verts.add_attr<vec3f>("r2")[0] = {r2[0], r2[1], r2[2]};
        auto zeno_scene_name = path;
        if (sceneSource->instance_source_paths.count(zeno_scene_name)) {
            zeno_scene_name = sceneSource->instance_source_paths[zeno_scene_name];
        }
        temp_matrices[zeno_scene_name].push_back(prim);
    }
    auto scene = std::make_shared<zeno::ListObject>();

    auto dict = std::make_shared<PrimitiveObject>();

    Json dict_json;
    int index = 0;
    for (auto &[zeno_scene_name, prims]: temp_matrices) {
        std::vector<PrimitiveObject*> prims_raws;
        for (auto &prim: prims) {
            prims_raws.push_back(prim.get());
        }
        auto prim = primMerge(prims_raws);
        prim->userData().set2("ResourceType", "Matrixes");
        prim->userData().set2("stamp-change", "TotalChange");
        prim->userData().set2("ObjectName", zeno_scene_name+"_m");
        scene->arr.push_back(prim);
        dict_json[zeno_scene_name+"_m"] = index;
        index += 1;
    }
    dict->userData().set2("json", dict_json.dump());
    auto prim_list_size = sceneSource->prim_list.size();
    dict->userData().set2("prim_list_size", int(prim_list_size));

    scene->arr.insert(scene->arr.begin(), dict);
    {
        auto scene_descriptor = std::make_shared<PrimitiveObject>();
        auto &ud = scene_descriptor->userData();
        ud.set2("ResourceType", std::string("SceneDescriptor"));
        Json json;
        json["BasicRenderInstances"] = Json();
        json["DynamicRenderGroups"]["Objects"] = Json();
        json["StaticRenderGroups"]["Objects"] = Json();
        for (const auto &[path, prim]: sceneSource->prim_list) {
            json["BasicRenderInstances"][path]["Geom"] = path;
            json["StaticRenderGroups"]["Objects"][path] = Json::array({path+"_m"});
        }

        ud.set2("Scene", std::string(json.dump()));
        scene->arr.push_back(scene_descriptor);
    }

    // prim list
    std::vector<std::string> prim_list_name;
    {
        std::vector<std::shared_ptr<PrimitiveObject>> prim_list;
        for (auto &[abc_path, p]: sceneSource->prim_list) {
            p->userData().set2("ObjectName", abc_path);
            prim_list.push_back(p);
            prim_list_name.push_back(abc_path);
        }
        scene->arr.insert(scene->arr.begin() + 1, prim_list.begin(), prim_list.end());
    }
    {
        Json json;
        for (auto i = 0; i < prim_list_name.size(); i++) {
            auto prim_name = prim_list_name[i];
            json[prim_name] = i;
        }
        scene->arr[0]->userData().set2("prim_list_name", json.dump());
    }
    // scene tree prim
    {
        auto scene_tree = std::make_shared<JsonObject>();
        scene_tree->json = sceneSource->to_json();
        scene->arr.push_back(scene_tree);
    }
    return scene;
}
#endif
static void get_local_matrix_map(
    Json &json
    , std::string parent_path
    , std::shared_ptr<SceneObject> scene
) {
    SceneTreeNode stn;
    std::string node_name = json["node_name"];
    std::string node_path = parent_path + '/' + node_name;
    Json r0 = json["r0"];
    Json r1 = json["r1"];
    Json r2 = json["r2"];
    Json t = json["t"];
    glm::mat4 mat;
    mat[0] = {float(r0[0]), float(r0[1]), float(r0[2]), 0.0f};
    mat[1] = {float(r1[0]), float(r1[1]), float(r1[2]), 0.0f};
    mat[2] = {float(r2[0]), float(r2[1]), float(r2[2]), 0.0f};
    mat[3] = {float(t[0]),  float(t[1]),  float(t[2]),  1.0f};
    scene->node_to_matrix[node_path+"_m"] = mat;
    stn.matrix = node_path+"_m";
    stn.visibility = json["visibility"] == 0? 0 : 1;
    if (json.contains("mesh")) {
        stn.meshes.push_back(node_path + "/" + std::string(json["mesh"]));
    }

    for (auto i = 0; i < json["children_name"].size(); i++) {
        std::string child_name = json["children_name"][i];
        std::string child_path = node_path + '/' + child_name;
        if (json[child_name].contains("instance_source_path")) {
            stn.children.push_back(json[child_name]["instance_source_path"]);
        }
        else {
            stn.children.push_back(child_path);
        }
    }
    scene->scene_tree[node_path] = stn;

    for (auto i = 0; i < json["children_name"].size(); i++) {
        std::string child_name = json["children_name"][i];
        if (!json[child_name].contains("instance_source_path")) {
            get_local_matrix_map(json[child_name], node_path, scene);
        }
    }
}

struct FormSceneTree : zeno::INode {
    int inputObjType = 0;
    void apply() override {
        auto sceneTree = std::make_shared<SceneObject>();
        auto scene_json = get_input2<JsonObject>("scene_info");
        sceneTree->root_name = "/ABC";
        auto prim_list = get_input2<ListObject>("prim_list");
//        zeno::log_info("prim_list: {}", prim_list->arr.size());
        for (auto p: prim_list->arr) {
            auto abc_path = p->userData().get2<std::string>("abcpath_0");
            {
                auto session = &zeno::getSession();
                int currframe = session->globalState->frameid;
                int beginframe = session->globalComm->beginFrameNumber;
                std::string mode = get_input2<std::string>("stampMode");
                if (mode == "UnChanged") {
                    if (currframe != beginframe) {
                        p = session->globalComm->constructEmptyObj(inputObjType);
                    }
                    p->userData().set2("stamp-change", "UnChanged");
                } else if (mode == "TotalChange") {
                    p->userData().set2("stamp-change", "TotalChange");
                } else if (mode == "DataChange") {
                    p->userData().set2("stamp-change", "DataChange");
                } else if (mode == "ShapeChange") {
                    p->userData().set2("stamp-change", "TotalChange");//shapechange暂时全部按Totalchange处理
                }
                p->userData().set2("ResourceType", "Mesh");
            }
            auto prim = std::static_pointer_cast<PrimitiveObject>(p);
            prim->userData().set2("ObjectName", abc_path);
            sceneTree->prim_list[abc_path] = prim;
        }
        get_local_matrix_map(scene_json->json, "", sceneTree);
        auto scene = sceneTree->to_layer_structure();
        set_output2("scene", scene);
    }
};

ZENDEFNODE( FormSceneTree, {
    {
        "scene_info",
        "prim_list",
        {"enum UnChanged DataChange ShapeChange TotalChange", "stampMode", "UnChanged"},
    },
    {
        {"scene"},
    },
    {},
    {
        "Scene",
    },
});

static void scene_add_prefix(
    std::string path
    , glm::mat4 xform
    , std::shared_ptr<SceneObject> sceneObject
) {
    std::unordered_map<std::string, SceneTreeNode> scene_tree;
    for (const auto &[key, value]: sceneObject->scene_tree) {
        SceneTreeNode stn;
        stn.visibility = value.visibility;
        if (value.matrix.size()) {
            stn.matrix = path + value.matrix;
        }
        for (auto &child: value.children) {
            stn.children.push_back(path + child);
        }
        for (auto &mesh: value.meshes) {
            stn.meshes.push_back(path + mesh);
        }
        scene_tree[path + key] = stn;
    }
    std::unordered_map<std::string, std::shared_ptr<PrimitiveObject>> new_prim_list;
    for (auto& [key, value] : sceneObject->prim_list) {
        auto obj_name = value->userData().get2<std::string>("ObjectName");
        obj_name = path + obj_name;
        value->userData().set2("ObjectName", obj_name);
        new_prim_list[path + key] = value;
    }
    std::unordered_map<std::string, glm::mat4> new_node_to_matrix;
    for (const auto& [key, value] : sceneObject->node_to_matrix) {
        new_node_to_matrix[path + key] = value;
    }
    sceneObject->scene_tree = scene_tree;
    sceneObject->prim_list = new_prim_list;
    sceneObject->node_to_matrix = new_node_to_matrix;
    {
        std::string xform_name = path + "_m";
        sceneObject->node_to_matrix[xform_name] = xform;
    }
}
static std::vector<std::string> splitPath(const std::string& path) {
    std::vector<std::string> result;
    if (path.empty()) return result;

    std::istringstream iss(path);
    std::string segment;
    std::string currentPath;

    //  跳过第一个空段（因为路径以 '/' 开头）
    std::getline(iss, segment, '/');

    while (std::getline(iss, segment, '/')) {
        if (segment.empty()) continue; //  跳过连续的 '/'
        currentPath += "/" + segment;
        result.push_back(currentPath);
    }

    return result;
}
static void merge_scene(
    std::shared_ptr<SceneObject> main_object
    , std::shared_ptr<SceneObject> second_object
) {
    for (const auto& [key, stn] : second_object->scene_tree) {
        main_object->scene_tree[key] = stn;
    }
    for (const auto& [key, mat] : second_object->node_to_matrix) {
        main_object->node_to_matrix[key] = mat;
    }
    for (const auto& [key, prim] : second_object->prim_list) {
        main_object->prim_list[key] = prim;
    }
}
glm::mat4 get_xform_from_prim(std::shared_ptr<PrimitiveObject> prim) {
    auto pos = prim->verts[0];
    auto r0 = prim->verts.add_attr<vec3f>("r0")[0];
    auto r1 = prim->verts.add_attr<vec3f>("r1")[0];
    auto r2 = prim->verts.add_attr<vec3f>("r2")[0];
    glm::mat4 mat;
    mat[0] = {r0[0], r0[1], r0[2], 0};
    mat[1] = {r1[0], r1[1], r1[2], 0};
    mat[2] = {r2[0], r2[1], r2[2], 0};
    mat[3] = {pos[0], pos[1], pos[2], 1};
    return mat;
}

struct MergeScene : zeno::INode {
    void apply() override {
        auto main_scene = get_scene_tree_from_list(get_input2<ListObject>("main_scene"));
        auto second_scene = get_scene_tree_from_list(get_input2<ListObject>("second_scene"));
        auto namespace1 = get_input2<std::string>("namespace1");
        auto namespace2 = get_input2<std::string>("namespace2");
        auto insert_path = get_input2<std::string>("insert_path");

        auto append_path1 = (namespace1 == ""? "" : "/") + namespace1;
        if (append_path1.size()) {
            glm::mat4 xform1 = glm::mat4(1);
            if (has_input2<PrimitiveObject>("xform1")) {
                xform1 = get_xform_from_prim(get_input2<PrimitiveObject>("xform1"));
            }
            scene_add_prefix(append_path1, xform1, main_scene);
        }
        auto append_path2 = append_path1 + (insert_path == ""? "" : "/") + insert_path + (namespace2 == ""? "" : "/") + namespace2;
        glm::mat4 xform2 = glm::mat4(1);
        if (has_input2<PrimitiveObject>("xform2")) {
            xform2 = get_xform_from_prim(get_input2<PrimitiveObject>("xform2"));
        }
        scene_add_prefix(append_path2, xform2, second_scene);

        merge_scene(main_scene, second_scene);
        if (append_path1.size()) {
            auto abc_stn = SceneTreeNode();
            abc_stn.children.push_back(append_path1 + main_scene->root_name);
            main_scene->scene_tree[append_path1] = abc_stn;
            main_scene->root_name = append_path1;
        }
        {
            auto inner_parent = append_path1 + (insert_path == ""? "" : "/") + insert_path;
            main_scene->scene_tree.at(inner_parent).children.push_back( namespace2==""?append_path2+second_scene->root_name:append_path2);
        }
        auto scene = main_scene->to_layer_structure();
        set_output2("scene", scene);
    }
};

ZENDEFNODE( MergeScene, {
    {
        "main_scene",
        "second_scene",
        {"string", "namespace1", ""},
        {"string", "namespace2", "namespace2"},
        {"string", "insert_path", ""},
        {"xform1"},
        {"xform2"},
    },
    {
        {"scene"},
    },
    {},
    {
        "Scene",
    },
});


struct FlattenSceneTree : zeno::INode {
    void apply() override {
        auto scene_tree = get_scene_tree_from_list(get_input2<ListObject>("scene"));
        auto use_static = get_input2<bool>("use_static");

        auto scene = scene_tree->to_flatten_structure(use_static);
        set_output2("scene", scene);
    }
};

ZENDEFNODE( FlattenSceneTree, {
    {
        "scene",
        {"bool", "use_static", "1"},
    },
    {
        {"scene"}
    },
    {},
    {
        "Scene",
    },
});

struct SceneRootRename : zeno::INode {
    void apply() override {
        auto scene_tree = get_scene_tree_from_list(get_input2<ListObject>("scene"));
//        zeno::log_info("SceneRootRename input root_name {}", scene_tree->root_name);
        auto new_root_name = get_input2<std::string>("new_root_name");
        if (zeno::ends_with(new_root_name, "/")) {
            new_root_name.pop_back();
        }
        if (new_root_name.empty()) {
            new_root_name = scene_tree->root_name;
        }
        if (zeno::starts_with(new_root_name, "/") == false) {
            new_root_name = "/" + new_root_name;
        }
        std::optional<glm::mat4> root_xform = std::nullopt;
        if (has_input2<PrimitiveObject>("xform")) {
            root_xform = get_xform_from_prim(get_input2<PrimitiveObject>("xform"));
        }
        auto new_scene_tree = scene_tree->root_rename(new_root_name, root_xform);
//        zeno::log_info("SceneRootRename output root_name {}", new_scene_tree->root_name);

        auto scene = new_scene_tree->to_layer_structure();
        set_output2("scene", scene);
    }
};

ZENDEFNODE( SceneRootRename, {
    {
        "scene",
        {"string", "new_root_name", "new_scene"},
        "xform",
    },
    {
        {"scene"},
    },
    {},
    {
        "Scene",
    },
});

struct RenderScene : zeno::INode {
    std::shared_ptr<ListObject> m_static_scene = nullptr;
    void apply() override {
        auto scene = std::make_shared<ListObject>();
        Json scene_descriptor_json;
        if (has_input("static_scene")) {
            if (!m_static_scene) {
                auto static_scene_tree = get_scene_tree_from_list(get_input2<ListObject>("static_scene"));
                auto new_static_scene_tree = static_scene_tree->root_rename("SRG", std::nullopt);
                auto static_scene = get_input2<bool>("flatten_static_scene")? new_static_scene_tree->to_flatten_structure(true) : new_static_scene_tree->to_layer_structure(true);
                for (auto i = 1; i < static_scene->arr.size() - 2; i++) {
                    scene->arr.push_back(static_scene->arr[i]);
                }
                m_static_scene = static_scene;
            }
            auto scene_str = m_static_scene->arr[m_static_scene->arr.size() - 2]->userData().get2<std::string>("Scene");
            auto static_scene_descriptor = Json::parse(scene_str);
            scene_descriptor_json["StaticRenderGroups"] = static_scene_descriptor["StaticRenderGroups"];
            scene_descriptor_json["BasicRenderInstances"].update(static_scene_descriptor["BasicRenderInstances"]);
        }
        if (has_input("dynamic_scene")) {
            auto dynamic_scene_tree = get_scene_tree_from_list(get_input2<ListObject>("dynamic_scene"));
            auto new_dynamic_scene_tree = dynamic_scene_tree->root_rename("DRG", std::nullopt);
            auto dynamic_scene = get_input2<bool>("flatten_dynamic_scene")? new_dynamic_scene_tree->to_flatten_structure(false) : new_dynamic_scene_tree->to_layer_structure(false);
            for (auto i = 1; i < dynamic_scene->arr.size() - 2; i++) {
                scene->arr.push_back(dynamic_scene->arr[i]);
            }
            auto scene_str = dynamic_scene->arr[dynamic_scene->arr.size() - 2]->userData().get2<std::string>("Scene");
            auto dynamic_scene_descriptor = Json::parse(scene_str);
            scene_descriptor_json["DynamicRenderGroups"] = dynamic_scene_descriptor["DynamicRenderGroups"];
            scene_descriptor_json["BasicRenderInstances"].update(dynamic_scene_descriptor["BasicRenderInstances"]);
        }
        {
            auto scene_descriptor = std::make_shared<PrimitiveObject>();
            auto &ud = scene_descriptor->userData();
            ud.set2("ResourceType", std::string("SceneDescriptor"));
            ud.set2("Scene", std::string(scene_descriptor_json.dump()));
            scene->arr.push_back(scene_descriptor);
        }
        set_output2("scene", scene);
    }
};

ZENDEFNODE( RenderScene, {
    {
        "static_scene",
        {"bool", "flatten_static_scene", "1"},
        "dynamic_scene",
        {"bool", "flatten_dynamic_scene", "1"},
    },
    {
        {"scene"},
    },
    {},
    {
        "Scene",
    },
});


struct MakeXform : zeno::INode {
    void apply() override {
        auto translate = get_input2<vec3f>("translate");
        auto eulerXYZ = get_input2<vec3f>("eulerXYZ");
        auto scale = get_input2<vec3f>("scale");
        glm::mat4 matScale  = glm::scale( glm::vec3(scale[0], scale[1], scale[2] ));


        auto order = get_input2<std::string>("EulerRotationOrder:");
        auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

        auto measure = get_input2<std::string>("EulerAngleMeasure:");
        auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

        glm::vec3 eularAngleXYZ = glm::vec3(eulerXYZ[0], eulerXYZ[1], eulerXYZ[2]);
        glm::mat4 matRotate = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        auto mat = matRotate * matScale;

        auto xform = std::make_shared<PrimitiveObject>();
        xform->resize(1);
        xform->verts[0] = translate;
        xform->verts.add_attr<vec3f>("r0")[0] = {mat[0][0], mat[0][1], mat[0][2]};
        xform->verts.add_attr<vec3f>("r1")[0] = {mat[1][0], mat[1][1], mat[1][2]};
        xform->verts.add_attr<vec3f>("r2")[0] = {mat[2][0], mat[2][1], mat[2][2]};
        set_output2("xform", xform);
    }
};

ZENDEFNODE( MakeXform, {
    {
        {"vec3f", "translate", "0, 0, 0"},
        {"vec3f", "eulerXYZ", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
    },
    {
        {"xform"},
    },
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "ZYX"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"}
    },
    {
        "Scene",
    },
});

struct MarkSceneState : zeno::INode {
    void apply() override {
        auto scene_tree_list = get_input2<ListObject>("scene");
        std::shared_ptr<PrimitiveObject> json_ptr = std::dynamic_pointer_cast<PrimitiveObject>(scene_tree_list->arr.back());
        auto json = Json::parse(json_ptr->userData().get2<std::string>("json"));
        json["type"] = get_input2<std::string>("type");
        json["flattened"] = get_input2<bool>("flattened");
        json_ptr->userData().set2("json", json.dump());
        set_output2("scene", scene_tree_list);
    }
};

ZENDEFNODE( MarkSceneState, {
    {
        {"scene"},
        {"enum static dynamic", "type", "static"},
        {"bool", "flattened", "1"},
    },
    {
        {"scene"},
    },
    {
    },
    {
        "Scene",
    },
});

}
