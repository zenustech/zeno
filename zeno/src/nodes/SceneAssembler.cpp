//
// Created by zh on 2025/4/14.
//
#include "zeno/types/DictObject.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/ListObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/fileio.h"
#include "zeno/utils/log.h"
#include "zeno/utils/string.h"
#include "zeno/utils/scope_exit.h"
#include "zeno/funcs/PrimitiveUtils.h"
#include <string>
#include <tinygltf/json.hpp>
#include <zeno/zeno.h>
#include <glm/glm.hpp>

using Json = nlohmann::ordered_json;
namespace zeno {

template <typename Key, typename Value>
class IndexMap {
private:
    std::vector<Key> insertion_order;
    std::unordered_map<Key, std::pair<Value, size_t>> data_map;

public:
    void insert(const Key& key, const Value& value) {
        auto it = data_map.find(key);
        if (it == data_map.end()) {
            data_map[key] = {value, insertion_order.size()};
            insertion_order.push_back(key);
        }
    }

    Value& at(const Key& key) {
        return data_map.at(key).first;
    }

    Value& at_index(size_t index) {
        if (index >= insertion_order.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_map.at(insertion_order[index]).first;
    }

    size_t get_index(const Key& key) const {
        return data_map.at(key).second;
    }

    size_t size() const {
        return insertion_order.size();
    }

    size_t count(const Key& key) const {
        return data_map.count(key);
    }

    auto begin() { return insertion_order.begin(); }
    auto end() { return insertion_order.end(); }
};
struct JsonObject : IObjectClone<JsonObject> {
    Json json;
};

struct SceneObject : IObjectClone<SceneObject> {
    IndexMap<std::string, glm::mat4> matrices;
    std::unordered_map<std::string, std::string> instance_source_paths;
    std::unordered_map<std::string, std::shared_ptr<PrimitiveObject>> prim_list;
    std::vector<std::string> find_children(std::string path) {
        std::vector<std::string> children;
        for (const auto &[k, v]: prim_list) {
            if (starts_with(k, path)) {
                children.push_back(k);
            }
        }
        return children;
    }
    void modify(std::string &path, vec3f pos, vec3f r0, vec3f r1, vec3f r2) {
        glm::mat4 mat;
        mat[0] = {r0[0], r0[1], r0[2], 0};
        mat[1] = {r1[0], r1[1], r1[2], 0};
        mat[2] = {r2[0], r2[1], r2[2], 0};
        mat[3] = {pos[0], pos[1], pos[2], 1};
        matrices.at(path) = mat;
    }
};


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
    return scene;
}

static void get_local_matrix_map(
    Json &json
    , std::string parent_path
    , std::shared_ptr<SceneObject> scene
) {
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
    scene->matrices.insert(node_path, mat);
    for (auto i = 0; i < json["children_name"].size(); i++) {
        std::string child_name = json["children_name"][i];
        get_local_matrix_map(json[child_name], node_path, scene);
    }
    if (json.contains("instance_source_path")) {
        std::string instance_source_path = json["instance_source_path"];
        if (instance_source_path.size()) {
            scene->instance_source_paths[node_path] = "/ABC" + instance_source_path;
        }
    }
}

struct FormSceneTree : zeno::INode {
    void apply() override {
        auto sceneTree = std::make_shared<SceneObject>();
        auto scene_json = get_input2<JsonObject>("scene_info");
        get_local_matrix_map(scene_json->json, "", sceneTree);
        auto prim_list = get_input2<ListObject>("prim_list");
        for (auto &p: prim_list->arr) {
            auto &ud = p->userData();
            auto abc_path = ud.get2<std::string>("abcpath_0");
            sceneTree->prim_list[abc_path] = std::static_pointer_cast<PrimitiveObject>(p);
        }
        set_output2("sceneTree", sceneTree);
        auto scene = scene_tree_to_structure(sceneTree.get());
        set_output2("scene", scene);

    }
};

ZENDEFNODE( FormSceneTree, {
    {
        "scene_info",
        "prim_list",
    },
    {
        {"scene"},
        {"sceneTree"},
    },
    {},
    {
        "Scene",
    },
});
struct ModifySceneTree : zeno::INode {
    void apply() override {
        auto sceneTree = get_input2<SceneObject>("sceneSource");
        auto path = get_input2<std::string>("path");
        auto prim = get_input2<PrimitiveObject>("prim");
        auto pos = prim->verts[0];
        auto r0 = prim->verts.add_attr<vec3f>("r0")[0];
        auto r1 = prim->verts.add_attr<vec3f>("r1")[0];
        auto r2 = prim->verts.add_attr<vec3f>("r2")[0];
        sceneTree->modify(path, pos, r0, r1, r2);


        set_output2("sceneTree", sceneTree);
        auto scene = scene_tree_to_structure(sceneTree.get());
        set_output2("scene", scene);
    }
};

ZENDEFNODE( ModifySceneTree, {
    {
        "sceneSource",
        {"string", "path", ""},
        "prim",
    },
    {
        {"scene"},
        {"sceneTree"},
    },
    {},
    {
        "Scene",
    },
});

struct SceneData : zeno::INode {
    void apply() override {
        auto scene = get_input2<SceneObject>("scene");
        auto prim_list = std::make_shared<zeno::ListObject>();
        for (auto &[k, v]: scene->prim_list) {
            v->userData().set2("zeno_scene_name", k);
            prim_list->arr.push_back(v);
        }
        set_output2("prim_list", prim_list);
    }
};

ZENDEFNODE( SceneData, {
    {
        "scene",
    },
    {
        "prim_list",
    },
    {},
    {
        "Scene",
    },
});


static void target_second_path(
    std::string path
    , std::shared_ptr<SceneObject> sceneObject
) {
    if (path.empty() || path == "/") {
        return;
    }
    if (zeno::starts_with(path, "/")) {
        path = path.substr(1);
    }
    if (zeno::ends_with(path, "/")) {
        path = path.substr(0, path.size() - 1);
    }
    IndexMap<std::string, glm::mat4> new_matrices;
    std::unordered_map<std::string, std::shared_ptr<PrimitiveObject>> new_prim_list;
    std::unordered_map<std::string, std::string> instance_source_paths;
    for (const auto& key : sceneObject->matrices) {
        const auto& value = sceneObject->matrices.at(key);
        new_matrices.insert("/ABC/" + path + key.substr(4), value);
    }
    for (const auto& [key, value] : sceneObject->prim_list) {
        new_prim_list["/ABC/" + path + key.substr(4)] = value;
    }
    for (const auto& [key, value] : sceneObject->instance_source_paths) {
        instance_source_paths["/ABC/" + path + key.substr(4)] = "/ABC/" + path + value.substr(4);
    }
    sceneObject->matrices = new_matrices;
    sceneObject->prim_list = new_prim_list;
    sceneObject->instance_source_paths = instance_source_paths;
}

static std::unordered_map<std::string, std::string> resolve_same_path(
    const std::shared_ptr<SceneObject> &main_object
    , std::shared_ptr<SceneObject> second_object
) {
    std::unordered_map<std::string, std::string> rename_mapping;
    rename_mapping["/ABC"] = "/ABC";
    for (const auto& key : second_object->matrices) {
        if (key == "/ABC") {
            continue;
        }
//        zeno::log_info("matrices key: {}", key);
        auto parent_path = get_parent_path(key);
//        zeno::log_info("get_parent_path: {} -> {}", key, parent_path);
        auto node_name = key.substr(parent_path.size());
        auto new_parent_path = rename_mapping[parent_path];
        auto new_path = new_parent_path + node_name;
        auto target_path = new_path;
        int index = 1;
        while (main_object->matrices.count(target_path)) {
            target_path = zeno::format("{}.{}", new_path, index);
            index += 1;
        }
        rename_mapping[key] = target_path;
    }
    return rename_mapping;
}
static std::vector<std::string> splitPath(const std::string& path) {
    std::vector<std::string> result;
    if (path.empty()) return result;

    std::istringstream iss(path);
    std::string segment;
    std::string currentPath;

    // 跳过第一个空段（因为路径以 '/' 开头）
    std::getline(iss, segment, '/');

    while (std::getline(iss, segment, '/')) {
        if (segment.empty()) continue; // 跳过连续的 '/'
        currentPath += "/" + segment;
        result.push_back(currentPath);
    }

    return result;
}
static void merge_scene(
    std::shared_ptr<SceneObject> main_object
    , std::shared_ptr<SceneObject> second_object
    , std::unordered_map<std::string, std::string> &rename_map
    , std::string path
) {
    if (path.size() && path != "/") {
        if (zeno::starts_with(path, "/")) {
            path = path.substr(1);
        }
        if (zeno::ends_with(path, "/")) {
            path = path.substr(0, path.size() - 1);
        }
        path = "/ABC/"+path;
        auto paths = splitPath(path);
        for (auto &_path: paths) {
            if (main_object->matrices.count(_path) == 0) {
                main_object->matrices.insert(_path, glm::mat4(1));
            }
        }
    }
    for (const auto& key : second_object->matrices) {
        auto new_path = rename_map[key];
        auto matrix = second_object->matrices.at(key);
        main_object->matrices.insert(new_path, matrix);
    }
    for (const auto& [key, prim] : second_object->prim_list) {
        auto new_path = rename_map[key];
        main_object->prim_list[new_path] = prim;
    }
    for (const auto& [key, value] : second_object->instance_source_paths) {
        auto new_key = rename_map[key];
        auto new_value = rename_map[value];
        main_object->instance_source_paths[new_key] = new_value;
    }
}

struct MergeScene : zeno::INode {
    void apply() override {
        auto main_scene = get_input2<SceneObject>("main_scene");
        auto second_scene = get_input2<SceneObject>("second_scene");
        auto path = get_input2<std::string>("path");
        target_second_path(path, second_scene);
        auto rename_map = resolve_same_path(main_scene, second_scene);
        merge_scene(main_scene, second_scene, rename_map, path);
        set_output2("sceneTree", main_scene);
        auto scene = scene_tree_to_structure(main_scene.get());
        set_output2("scene", scene);

    }
};

ZENDEFNODE( MergeScene, {
    {
        "main_scene",
        "second_scene",
        {"string", "path", ""}
    },
    {
        {"scene"},
        {"sceneTree"},
    },
    {},
    {
        "Scene",
    },
});

struct SceneStructure : zeno::INode {
    void apply() override {
        auto sceneSource = get_input2<SceneObject>("sceneSource");
        auto scene = scene_tree_to_structure(sceneSource.get());
        set_output2("scene", scene);
    }
};

ZENDEFNODE( SceneStructure, {
    {
        "sceneSource",
    },
    {
        "scene",
    },
    {},
    {
        "Scene",
    },
});

}
