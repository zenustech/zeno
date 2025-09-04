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
#include "zeno/utils/bit_operations.h"

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
    std::unordered_map <std::string, std::vector<glm::mat4>> node_to_matrix;
    std::unordered_map <std::string, std::vector<int>> node_to_id;
    std::unordered_map <std::string, std::shared_ptr<PrimitiveObject>> prim_list;
    std::string root_name;
    std::string type = "static";
    std::string matrixMode = "TotalChange";

    // return value is in world space
    std::optional<std::pair<glm::vec3, glm::vec3>> get_node_bbox(
            const std::string& node_name
            , const std::vector<glm::mat4> &parent_mat
            , const std::unordered_map<std::string, std::pair<glm::vec3, glm::vec3>> &mesh_bbox
            , const std::unordered_map<std::string, std::vector<glm::mat4>> &node_2_matrix
    ) {
        if (scene_tree.count(node_name) == 0) {
            return std::nullopt;
        }
        SceneTreeNode stn = scene_tree.at(node_name);
        std::vector<glm::mat4> local_mat = {glm::mat4(1)};
        if (node_2_matrix.count(stn.matrix)) {
            local_mat = node_2_matrix.at(stn.matrix);
        }
        std::vector<glm::mat4> global_mat;
        for (const auto &pm: parent_mat) {
            for (const auto &lm: local_mat) {
                global_mat.emplace_back(pm * lm);
            }
        }
        std::vector<glm::vec3> bbox_points;
        for (const auto &mesh: stn.meshes) {
            if (mesh_bbox.count(mesh) == 0) {
                continue;
            }
            {
                auto [bmin_OS, bmax_OS] = mesh_bbox.at(mesh);
                for (const auto &gm: global_mat) {
                    auto bmin_WS = glm::vec3(gm * glm::vec4(bmin_OS, 1));
                    auto bmax_WS = glm::vec3(gm * glm::vec4(bmax_OS, 1));
                    bbox_points.emplace_back(bmin_WS);
                    bbox_points.emplace_back(bmax_WS);
                }
            }
        }
        for (const auto &child: stn.children) {
            auto bboxWS = get_node_bbox(child, global_mat, mesh_bbox, node_2_matrix);
            if (bboxWS.has_value()) {
                bbox_points.emplace_back(bboxWS->first);
                bbox_points.emplace_back(bboxWS->second);
            }
        }

        if (bbox_points.empty()) {
            return std::nullopt;
        }
        else {
            // Initialize min and max with first point
            glm::vec3 bmin = bbox_points[0];
            glm::vec3 bmax = bbox_points[0];

            // Iterate through all points to find min and max
            for (const auto& point : bbox_points) {
                bmin.x = std::min(bmin.x, point.x);
                bmin.y = std::min(bmin.y, point.y);
                bmin.z = std::min(bmin.z, point.z);

                bmax.x = std::max(bmax.x, point.x);
                bmax.y = std::max(bmax.y, point.y);
                bmax.z = std::max(bmax.z, point.z);
            }
            return std::pair{bmin, bmax};
        }
    }
    std::optional<std::pair<glm::vec3, glm::vec3>> get_node_bbox(
            const std::vector<std::string> &links
            , const std::unordered_map<std::string, std::pair<glm::vec3, glm::vec3>> &mesh_bbox
            , const std::unordered_map<std::string, std::vector<glm::mat4>> &node_2_matrix
    ) {
        if (links.empty()) {
            return std::nullopt;
        }
        std::vector<glm::mat4> parent_mat = {glm::mat4(1)};
        for (auto i = 0; i < links.size() - 1; i++) {
            const auto& node_name = links[i];
            if (scene_tree.count(node_name) == 0) {
                return std::nullopt;
            }
            SceneTreeNode stn = scene_tree.at(node_name);
            std::vector<glm::mat4> local_mat = {glm::mat4(1)};
            if (node_2_matrix.count(stn.matrix)) {
                local_mat = node_2_matrix.at(stn.matrix);
            }
            std::vector<glm::mat4> global_mat;
            for (const auto &pm: parent_mat) {
                for (const auto &lm: local_mat) {
                    global_mat.emplace_back(pm * lm);
                }
            }
            parent_mat = global_mat;
        }
        return get_node_bbox(links.back(), parent_mat, mesh_bbox, node_2_matrix);
    }

    std::string
    get_new_root_name(const std::string &root_name, const std::string &new_root_name, const std::string &path) {
        return new_root_name + path.substr(root_name.size());
    }

    std::shared_ptr <SceneObject> root_rename(std::string new_root_name, std::vector<glm::mat4> root_xform) {
        auto new_scene_obj = std::make_shared<SceneObject>();
        new_scene_obj->type = this->type;

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
        if (root_xform.size()) {
            new_scene_obj->node_to_matrix[xform_name] = root_xform;
        } else {
            if (new_scene_obj->node_to_matrix.count(xform_name) == 0) {
                new_scene_obj->node_to_matrix[xform_name] = {glm::mat4(1)};
            }
        }
        return new_scene_obj;
    }

    std::string to_json() {
        Json json;
        json["root_name"] = root_name;
        json["type"] = type;
        json["matrixMode"] = matrixMode;
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
            for (auto const &[path, mats]: node_to_matrix) {
                for (auto const& mat: mats) {
                    Json matrix = Json::array();
                    for (auto i = 0; i < 4; i++) {
                        for (auto j = 0; j < 3; j++) {
                            matrix.push_back(mat[i][j]);
                        }
                    }
                    mat_json[path].push_back(matrix);
                }
            }
            json["node_to_matrix"] = mat_json;
        }
        {
            Json id_json;
            for (auto const &[path, ids]: node_to_id) {
                for (auto const& id: ids) {
                    id_json[path].push_back(id);
                }
            }
            json["node_to_id"] = id_json;
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
        matrixMode = json["matrixMode"];
        {
            node_to_matrix.clear();
            Json const &mat_json = json["node_to_matrix"];
            for (auto &[path, mats_json]: mat_json.items()) {
                auto count = mats_json.size();
                for (auto idx = 0; idx < count; idx++) {
                    auto mat_json = mats_json[idx];
                    auto matrix = glm::mat4(1);
                    for (auto i = 0; i < 4; i++) {
                        for (auto j = 0; j < 3; j++) {
                            int index = i * 3 + j;
                            matrix[i][j] = float(mat_json[index]);
                        }
                    }
                    node_to_matrix[path].push_back(matrix);
                }
                if (node_to_matrix[path].size() == 0) {
                    node_to_matrix[path].emplace_back(1);
                }
            }
        }
        {
            node_to_id.clear();
            Json const &id_json = json["node_to_id"];
            for (auto &[path, ids_json]: id_json.items()) {
                auto count = ids_json.size();
                for (auto idx = 0; idx < count; idx++) {
                    auto id_json = ids_json[idx];
                    node_to_id[path].push_back(int(id_json));
                }
                if (node_to_id[path].size() == 0) {
                    node_to_id[path].emplace_back(0);
                }
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

    static std::shared_ptr<PrimitiveObject> mats_to_prim(std::string &obj_name, std::vector<glm::mat4> &matrixs, bool use_static, const std::string& matrixMode) {
        auto prim = std::make_shared<PrimitiveObject>();
        prim->verts.resize(4 * matrixs.size());
        for (auto i = 0; i < matrixs.size(); i++) {
            auto &matrix = matrixs[i];
            auto r0 = matrix[0];
            auto r1 = matrix[1];
            auto r2 = matrix[2];
            auto t = matrix[3];
            prim->verts[0 + i * 4][0] = r0[0];
            prim->verts[0 + i * 4][1] = r1[0];
            prim->verts[0 + i * 4][2] = r2[0];
            prim->verts[1 + i * 4][0] = t[0];
            prim->verts[1 + i * 4][1] = r0[1];
            prim->verts[1 + i * 4][2] = r1[1];
            prim->verts[2 + i * 4][0] = r2[1];
            prim->verts[2 + i * 4][1] = t[1];
            prim->verts[2 + i * 4][2] = r0[2];
            prim->verts[3 + i * 4][0] = r1[2];
            prim->verts[3 + i * 4][1] = r2[2];
            prim->verts[3 + i * 4][2] = t[2];
        }

        prim->userData().set2("ResourceType", "Matrixes");
        if (use_static) {
            prim->userData().set2("stamp-change", "UnChanged");
        } else {
            prim->userData().set2("stamp-change", matrixMode);
        }
        prim->userData().set2("ObjectName", obj_name);
        return prim;
    }

    std::shared_ptr <zeno::ListObject> to_structure() {
        bool use_static = type == "static";
        auto scene = std::make_shared<zeno::ListObject>();
        {
            for (auto &[abc_path, p]: prim_list) {
                scene->arr.push_back(p);
            }
        }
        {
            for (auto &[path, stn]: scene_tree) {
                std::vector<glm::mat4> matrixs;
                if (stn.visibility) {
                    if (stn.matrix.size() && node_to_matrix.count(stn.matrix)) {
                        matrixs = node_to_matrix[stn.matrix];
                    } else {
                        matrixs = {glm::mat4(1)};
                    }
                } else {
                    matrixs = {glm::mat4(0)};
                }
                std::string object_name = path + "_m";
                if (stn.matrix.size()) {
                    object_name = path + "_m";
                }
                auto prim = mats_to_prim(object_name, matrixs, use_static, this->matrixMode);
                if (node_to_id.count(stn.matrix) && node_to_id[stn.matrix].size() == matrixs.size()) {
                    prim->loops.values = node_to_id[stn.matrix];
                }
                scene->arr.push_back(prim);
            }
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
                auto vol_mat = prim->userData().get2<std::string>("vol_mat", "");
                if (vol_mat.size()) {
                    BasicRenderInstances[path]["Material"] = vol_mat;
                }
            }
            json["BasicRenderInstances"] = BasicRenderInstances;

            Json RenderGroups = Json::object();
            for (auto &[path, stn]: scene_tree) {
                Json render_group = Json();
                for (auto &child: stn.children) {
                    render_group[child] = Json::array({path + "_m"});
                }
                for (auto &child: stn.meshes) {
                    render_group[child] = Json::array({path + "_m"});
                }
                if (render_group.is_null()) {
                    continue;
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
            st->userData().set2("SceneTreeType", this->type);
            scene->arr.push_back(st);
        }
        return scene;
    }

    void flatten() {
        std::unordered_map <std::string, SceneTreeNode> temp_scene_tree;
        std::unordered_map <std::string, std::vector<glm::mat4>> temp_node_to_matrix;

        SceneTreeNode new_root_node;
        new_root_node.visibility = this->scene_tree[this->root_name].visibility;
        new_root_node.matrix = this->scene_tree[this->root_name].matrix;
        temp_node_to_matrix[this->root_name+"_m"] = {glm::mat4(1)};
        {
            std::unordered_map<std::string, std::vector<glm::mat4>> tmp_matrix_xforms;
            std::deque<std::pair<std::string, std::vector<glm::mat4>>> worker;
            std::vector<glm::mat4> init;
            init.emplace_back(1);
            worker.emplace_back(root_name, init);
            while (worker.size()) {
                auto [path, parent_global_matrix] = worker.front();
                if (scene_tree.count(path) == 0) {
                    zeno::log_error("path: {} not found, size: {}", path, path.size());
                }
                auto stn = scene_tree.at(path);
                worker.pop_front();

                std::vector<glm::mat4> local_mats;
                if (stn.visibility) {
                    if (stn.matrix.size()) {
                        local_mats = node_to_matrix[stn.matrix];
                    }
                    else {
                        local_mats.emplace_back(1);
                    }
                } else {
                    local_mats.emplace_back(0);
                }
                std::vector<glm::mat4> global_matrix;
                for (auto const &p_g_mat: parent_global_matrix) {
                    for (auto const &l_mat: local_mats) {
                        global_matrix.push_back(p_g_mat * l_mat);

                    }
                }
                for (auto &mesh: stn.meshes) {
                    tmp_matrix_xforms[mesh].insert(tmp_matrix_xforms[mesh].end(), global_matrix.begin(), global_matrix.end());
                }
                for (auto &child: stn.children) {
                    worker.emplace_back(child, global_matrix);
                    if (child.empty()) {
                        zeno::log_info("path child empty: {}", path);
                    }
                }
            }
            for(auto const&[mesh_name, mats]: tmp_matrix_xforms) {
                SceneTreeNode stn;
                std::string node_name = mesh_name + "_node";
                stn.matrix = mesh_name + "_node_m";
                temp_node_to_matrix[stn.matrix] = mats;
                new_root_node.children.push_back(node_name);
                stn.meshes.push_back(mesh_name);
                temp_scene_tree[node_name] = stn;
            }
            temp_scene_tree[this->root_name] = new_root_node;

        }
        this->node_to_matrix = temp_node_to_matrix;
        this->scene_tree = temp_scene_tree;
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
