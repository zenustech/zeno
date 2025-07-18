//
// Created by zh on 2025/7/2.
//

#ifndef ZENO_SCENEASSEMBLER_H
#define ZENO_SCENEASSEMBLER_H
#include <tinygltf/json.hpp>
#include <glm/glm.hpp>
#include <deque>

#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/ListObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/log.h"

using Json = nlohmann::json;
namespace zeno {
struct JsonObject : IObjectClone<JsonObject> {
    Json json;
};

struct SceneTreeNode {
    std::vector <std::string> meshes;
    std::string matrix;
    std::vector <std::string> children;
    int visibility = 1;
};

struct SceneObject : IObjectClone<SceneObject> {
    std::unordered_map <std::string, SceneTreeNode> scene_tree;
    std::unordered_map <std::string, glm::mat4> node_to_matrix;
    std::unordered_map <std::string, std::shared_ptr<PrimitiveObject>> prim_list;
    std::string root_name;
    std::string type = "static";
    bool flattened = true;

    std::string
    get_new_root_name(const std::string &root_name, const std::string &new_root_name, const std::string &path) {
        return new_root_name + path.substr(root_name.size());
    }

    std::shared_ptr <SceneObject> root_rename(std::string new_root_name, std::optional <glm::mat4> root_xform) {
        auto new_scene_obj = std::make_shared<SceneObject>();
        new_scene_obj->type = this->type;
        new_scene_obj->flattened = this->flattened;

        for (auto const &[path, stn]: scene_tree) {
            auto new_key = get_new_root_name(root_name, new_root_name, path);
//            zeno::log_info("path_rename {} -> {}", path, new_key);
            SceneTreeNode nstn;
            nstn.visibility = stn.visibility;
            if (stn.matrix.size()) {
                nstn.matrix = get_new_root_name(root_name, new_root_name, stn.matrix);
            }
            for (auto &mesh: stn.meshes) {
                nstn.meshes.push_back(get_new_root_name(root_name, new_root_name, mesh));
            }
            for (auto &child: stn.children) {
                nstn.children.push_back(get_new_root_name(root_name, new_root_name, child));
            }
            new_scene_obj->scene_tree[new_key] = nstn;
        }

        for (auto &[k, v]: node_to_matrix) {
            auto new_key = get_new_root_name(root_name, new_root_name, k);
            new_scene_obj->node_to_matrix[new_key] = v;
        }
        for (auto &[k, p]: prim_list) {
            auto new_key = get_new_root_name(root_name, new_root_name, k);
            auto new_prim = std::static_pointer_cast<PrimitiveObject>(p->clone());
            new_prim->userData().set2("ObjectName", new_key);
            new_scene_obj->prim_list[new_key] = new_prim;
        }
        new_scene_obj->root_name = new_root_name;
        std::string xform_name = new_root_name + "_m";
        if (root_xform.has_value()) {
            new_scene_obj->node_to_matrix[xform_name] = root_xform.value();
        } else {
            if (new_scene_obj->node_to_matrix.count(xform_name) == 0) {
                new_scene_obj->node_to_matrix[xform_name] = glm::mat4(1);
            }
        }
        return new_scene_obj;
    }

    std::string to_json() {
        Json json;
        json["root_name"] = root_name;
        json["type"] = type;
        json["flattened"] = flattened;
        {
            Json part;
            for (auto &[path, stn]: scene_tree) {
                Json node;
                node["meshes"] = Json::array();
                for (auto &mesh: stn.meshes) {
                    node["meshes"].push_back(mesh);
                }
                node["children"] = Json::array();
                for (auto &child: stn.children) {
                    node["children"].push_back(child);
                }
                node["matrix"] = stn.matrix;
                node["visibility"] = stn.visibility;
                part[path] = node;
            }
            json["scene_tree"] = part;
        }
        {
            Json mat_json;
            for (auto const &[path, mat]: node_to_matrix) {
                Json matrix = Json::array();
                for (auto i = 0; i < 4; i++) {
                    for (auto j = 0; j < 3; j++) {
                        matrix.push_back(mat[i][j]);
                    }
                }
                mat_json[path] = matrix;
            }
            json["node_to_matrix"] = mat_json;
        }
        return json.dump();
    }

    void from_json(std::string const &json_str) {
        Json json = Json::parse(json_str);
        from_json(json);
    }
    void from_json(Json const &json) {
        root_name = json["root_name"];
        type = json["type"];
        flattened = json["flattened"];
        {
            node_to_matrix.clear();
            Json const &mat_json = json["node_to_matrix"];
            for (auto &[path, mat_json]: mat_json.items()) {
                auto matrix = glm::mat4(1);
                for (auto i = 0; i < 4; i++) {
                    for (auto j = 0; j < 3; j++) {
                        int index = i * 3 + j;
                        matrix[i][j] = float(mat_json[index]);
                    }
                }
                node_to_matrix[path] = matrix;
            }
        }
        {
            scene_tree.clear();
            Json const &part = json["scene_tree"];
            for (auto &[path, jstn]: part.items()) {
                SceneTreeNode stn;
                stn.matrix = jstn["matrix"];
                stn.visibility = jstn["visibility"];
                for (auto &child: jstn["children"]) {
                    stn.children.push_back(child);
                }
                for (auto &mesh: jstn["meshes"]) {
                    stn.meshes.push_back(mesh);
                }
                scene_tree[path] = stn;
            }
        }
    }

    std::shared_ptr <zeno::ListObject> to_layer_structure(bool use_static) {
        auto scene = std::make_shared<zeno::ListObject>();
        auto dict = std::make_shared<PrimitiveObject>();
        scene->arr.push_back(dict);
        {
            for (auto &[abc_path, p]: prim_list) {
                scene->arr.push_back(p);
            }
            int prim_count = prim_list.size();
            dict->userData().set2("prim_count", prim_count);
        }
        {
            int matrix_count = 0;
            for (auto &[path, stn]: scene_tree) {
                auto matrix = glm::mat4(1);
                if (stn.visibility) {
                    if (stn.matrix.size() && node_to_matrix.count(stn.matrix)) {
                        matrix = node_to_matrix[stn.matrix];
                    } else {
                        continue;
                    }
                } else {
                    matrix = glm::mat4(0);
                }
                auto r0 = matrix[0];
                auto r1 = matrix[1];
                auto r2 = matrix[2];
                auto t = matrix[3];
                auto prim = std::make_shared<PrimitiveObject>();
                prim->verts.resize(4);
                prim->verts[0][0] = r0[0];
                prim->verts[0][1] = r1[0];
                prim->verts[0][2] = r2[0];
                prim->verts[1][0] = t[0];
                prim->verts[1][1] = r0[1];
                prim->verts[1][2] = r1[1];
                prim->verts[2][0] = r2[1];
                prim->verts[2][1] = t[1];
                prim->verts[2][2] = r0[2];
                prim->verts[3][0] = r1[2];
                prim->verts[3][1] = r2[2];
                prim->verts[3][2] = t[2];

                prim->userData().set2("ResourceType", "Matrixes");
                if (use_static) {
                    prim->userData().set2("stamp-change", "UnChanged");
                } else {
                    prim->userData().set2("stamp-change", "TotalChange");
                }
                std::string object_name = path + "_m";
                if (stn.matrix.size()) {
                    object_name = path + "_m";
                }
                prim->userData().set2("ObjectName", object_name);
                scene->arr.push_back(prim);
                matrix_count += 1;
            }
            dict->userData().set2("matrix_count", matrix_count);
        }
        {
            auto scene_descriptor = std::make_shared<PrimitiveObject>();
            auto &ud = scene_descriptor->userData();
            ud.set2("ResourceType", std::string("SceneDescriptor"));
            Json json;
            json["type"] = use_static ? "static" : "dynamic";
            Json BasicRenderInstances = Json();
            for (const auto &[path, prim]: prim_list) {
                BasicRenderInstances[path]["Geom"] = path;
                BasicRenderInstances[path]["Material"] = "Default";
            }
            json["BasicRenderInstances"] = BasicRenderInstances;

            Json RenderGroups = Json();
            for (auto &[path, stn]: scene_tree) {
                Json render_group = Json();
                for (auto &child: stn.children) {
                    render_group[child] = Json::array({path + "_m"});
                }
                for (auto &child: stn.meshes) {
                    render_group[child] = Json::array({path + "_m"});
                }
                RenderGroups[path] = render_group;
            }
            if (use_static) {
                json["StaticRenderGroups"] = RenderGroups;
            } else {
                json["DynamicRenderGroups"] = RenderGroups;
            }
            ud.set2("Scene", std::string(json.dump()));
            scene->arr.push_back(scene_descriptor);
        }
        {
            auto st = std::make_shared<PrimitiveObject>();
            st->userData().set2("json", to_json());
            st->userData().set2("ResourceType", std::string("SceneTree"));
            scene->arr.push_back(st);
        }
        return scene;
    }

    std::vector<std::shared_ptr<IObject>> to_flatten_structure_matrix(std::unordered_map<std::string, glm::mat4> &modified_xfroms) {
        std::vector<std::shared_ptr<IObject>> scene;
        std::unordered_map <std::string, std::vector<glm::mat4>> tmp_matrix_xforms;
        std::deque <std::pair<std::string, glm::mat4>> worker;
        worker.emplace_back(root_name, glm::mat4(1));
        while (worker.size()) {
            auto [path, parent_global_matrix] = worker.front();
            if (scene_tree.count(path) == 0) {
                zeno::log_error("path: {} not found, size: {}", path, path.size());
            }
            auto stn = scene_tree.at(path);
            worker.pop_front();

            auto local_mat = glm::mat4(1);
            if (stn.visibility) {
                if (modified_xfroms.count(path)) {
                    local_mat = modified_xfroms[path];
                }
                else {
                    if (stn.matrix.size()) {
                        local_mat = node_to_matrix[stn.matrix];
                    }
                }
            } else {
                local_mat = glm::mat4(0);
            }
            auto global_matrix = parent_global_matrix * local_mat;
            for (auto &mesh: stn.meshes) {
                tmp_matrix_xforms[mesh].push_back(global_matrix);
            }
            for (auto &child: stn.children) {
                worker.emplace_back(child, global_matrix);
                if (child.empty()) {
                    zeno::log_info("path child empty: {}", path);
                }
            }
        }
        for (auto &[mesh_name, mats]: tmp_matrix_xforms) {
            auto matrix = std::make_shared<PrimitiveObject>();
            matrix->resize(mats.size() * 4);
            for (auto i = 0; i < mats.size(); i++) {
                auto &mat = mats[i];
                matrix->verts[i * 4 + 0][0] = mat[0][0];
                matrix->verts[i * 4 + 0][1] = mat[1][0];
                matrix->verts[i * 4 + 0][2] = mat[2][0];
                matrix->verts[i * 4 + 1][0] = mat[3][0];
                matrix->verts[i * 4 + 1][1] = mat[0][1];
                matrix->verts[i * 4 + 1][2] = mat[1][1];
                matrix->verts[i * 4 + 2][0] = mat[2][1];
                matrix->verts[i * 4 + 2][1] = mat[3][1];
                matrix->verts[i * 4 + 2][2] = mat[0][2];
                matrix->verts[i * 4 + 3][0] = mat[1][2];
                matrix->verts[i * 4 + 3][1] = mat[2][2];
                matrix->verts[i * 4 + 3][2] = mat[3][2];
            }
            matrix->userData().set2("ResourceType", "Matrixes");
            matrix->userData().set2("stamp-change", "TotalChange");
            std::string object_name = mesh_name + "_m";
            matrix->userData().set2("ObjectName", object_name);
            matrix->userData().set2("objRunType", "matrix");
            scene.push_back(matrix);
        }
        return scene;
    }

    std::shared_ptr <zeno::ListObject> to_flatten_structure(bool use_static) {
//        zeno::log_info("to_flatten_structure root_name: {}", root_name);
        auto scene = std::make_shared<zeno::ListObject>();
        auto dict = std::make_shared<PrimitiveObject>();
        scene->arr.push_back(dict);
        {
            for (auto &[abc_path, p]: prim_list) {
                scene->arr.push_back(p);
            }
            dict->userData().set2("prim_count", int(prim_list.size()));
        }
        {
            std::unordered_map <std::string, std::vector<glm::mat4>> tmp_matrix_xforms;
            std::deque <std::pair<std::string, glm::mat4>> worker;
            worker.emplace_back(root_name, glm::mat4(1));
            while (worker.size()) {
                auto [path, parent_global_matrix] = worker.front();
                if (scene_tree.count(path) == 0) {
                    zeno::log_error("path: {} not found, size: {}", path, path.size());
                }
                auto stn = scene_tree.at(path);
                worker.pop_front();

                auto local_mat = glm::mat4(1);
                if (stn.visibility) {
                    if (stn.matrix.size()) {
                        local_mat = node_to_matrix[stn.matrix];
                    }
                } else {
                    local_mat = glm::mat4(0);
                }
                auto global_matrix = parent_global_matrix * local_mat;
                for (auto &mesh: stn.meshes) {
                    tmp_matrix_xforms[mesh].push_back(global_matrix);
                }
                for (auto &child: stn.children) {
                    worker.emplace_back(child, global_matrix);
                    if (child.empty()) {
                        zeno::log_info("path child empty: {}", path);
                    }
                }
            }
            for (auto &[mesh_name, mats]: tmp_matrix_xforms) {
                auto matrix = std::make_shared<PrimitiveObject>();
                matrix->resize(mats.size() * 4);
                for (auto i = 0; i < mats.size(); i++) {
                    auto &mat = mats[i];
                    matrix->verts[i * 4 + 0][0] = mat[0][0];
                    matrix->verts[i * 4 + 0][1] = mat[1][0];
                    matrix->verts[i * 4 + 0][2] = mat[2][0];
                    matrix->verts[i * 4 + 1][0] = mat[3][0];
                    matrix->verts[i * 4 + 1][1] = mat[0][1];
                    matrix->verts[i * 4 + 1][2] = mat[1][1];
                    matrix->verts[i * 4 + 2][0] = mat[2][1];
                    matrix->verts[i * 4 + 2][1] = mat[3][1];
                    matrix->verts[i * 4 + 2][2] = mat[0][2];
                    matrix->verts[i * 4 + 3][0] = mat[1][2];
                    matrix->verts[i * 4 + 3][1] = mat[2][2];
                    matrix->verts[i * 4 + 3][2] = mat[3][2];
                }
                matrix->userData().set2("ResourceType", "Matrixes");
                if (use_static) {
                    matrix->userData().set2("stamp-change", "UnChanged");
                } else {
                    matrix->userData().set2("stamp-change", "TotalChange");
                }
                std::string object_name = mesh_name + "_m";
                matrix->userData().set2("ObjectName", object_name);
                matrix->userData().set2("objRunType", "matrix");
                scene->arr.push_back(matrix);
            }
            dict->userData().set2("matrix_count", int(tmp_matrix_xforms.size()));
        }
        {
            auto scene_descriptor = std::make_shared<PrimitiveObject>();
            auto &ud = scene_descriptor->userData();
            ud.set2("ResourceType", std::string("SceneDescriptor"));
            Json json;
            json["type"] = use_static ? "static" : "dynamic";
            json["BasicRenderInstances"] = Json();
            for (const auto &[path, prim]: prim_list) {
                json["BasicRenderInstances"][path]["Geom"] = path;
                json["BasicRenderInstances"][path]["Material"] = "Default";
                if (use_static) {
                    json["StaticRenderGroups"]["StaticObjects"][path] = Json::array({path + "_m"});
                } else {
                    json["DynamicRenderGroups"]["DynamicObjects"][path] = Json::array({path + "_m"});
                }
            }
            ud.set2("Scene", std::string(json.dump()));
            scene->arr.push_back(scene_descriptor);
        }
        {
            auto st = std::make_shared<PrimitiveObject>();
            st->userData().set2("json", to_json());
            st->userData().set2("ResourceType", std::string("SceneTree"));
            scene->arr.push_back(st);
        }
        return scene;
    }

    std::shared_ptr <zeno::ListObject> to_structure() {
        if (flattened) {
            return to_flatten_structure(type == "static");
        } else {
            return to_layer_structure(type == "static");
        }
    }

    std::shared_ptr <zeno::ListObject> to_list() {
        auto scene = std::make_shared<zeno::ListObject>();
        for (auto &[abc_path, p]: prim_list) {
            scene->arr.push_back(p);
        }

        auto st = std::make_shared<PrimitiveObject>();
        st->userData().set2("json", to_json());
        st->userData().set2("ResourceType", std::string("SceneTree"));
        scene->arr.push_back(st);
        return scene;
    }
};

static std::shared_ptr <SceneObject> get_scene_tree_from_list2(std::shared_ptr <ListObject> list_obj) {
    auto scene_tree = std::make_shared<SceneObject>();
    for (auto i = 0; i < list_obj->arr.size(); i++) {
        auto &ud = list_obj->arr[i]->userData();
        auto resource_type = ud.get2("ResourceType", std::string("None"));
        if (resource_type == "SceneTree") {
            scene_tree->from_json(ud.get2<std::string>("json"));
        }
        else if (resource_type == "Mesh") {
            auto prim = std::static_pointer_cast<PrimitiveObject>(list_obj->arr[i]);
            auto object_name = ud.get2<std::string>("ObjectName");
            scene_tree->prim_list[object_name] = prim;
        }
    }
    return scene_tree;
}
}
#endif //ZENO_SCENEASSEMBLER_H
