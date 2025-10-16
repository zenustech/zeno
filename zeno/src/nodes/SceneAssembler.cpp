//
// Created by zh on 2025/4/14.
//
#include <zeno/extra/GlobalComm.h>
#include <zeno/types/DummyObject.h>
#include <zeno/core/Graph.h>
#include "zeno/types/PrimitiveObject.h"
#include <zeno/types/IGeometryObject.h>
#include "zeno/types/ListObject_impl.h"
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
#include <zeno/funcs/PrimitiveTools.h>
#include "zeno/extra/SceneAssembler.h"
#include <zeno/geo/geometryutil.h>


namespace zeno {
#if 0
static std::shared_ptr<SceneObject> get_scene_tree_from_list(std::shared_ptr<ListObject> list_obj) {
    auto scene_tree = std::make_unique<SceneObject>();
    auto json_obj = static_cast<PrimitiveObject*>(list_obj->m_impl->m_objects.back().get());
    scene_tree->from_json(zsString2Std(json_obj->userData()->get_string("json")));
    auto prim_list_size = list_obj->m_impl->m_objects.front()->userData()->get_int("prim_count");
    for (auto i = 1; i <= prim_list_size; i++) {
        auto prim = safe_uniqueptr_cast<GeometryObject_Adapter>(list_obj->m_impl->m_objects[i]->clone());
        auto object_name = zsString2Std(prim->userData()->get_string("ObjectName"));
        scene_tree->geom_list[object_name] = std::move(prim);
    }
    return scene_tree;
}
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
    std::unordered_map<std::string, std::vector<std::unique_ptr<PrimitiveObject>>> temp_matrices;
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
        auto prim = std::make_unique<PrimitiveObject>();
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
    auto scene = std::make_unique<zeno::ListObject>();

    auto dict = std::make_unique<PrimitiveObject>();

    Json dict_json;
    int index = 0;
    for (auto &[zeno_scene_name, prims]: temp_matrices) {
        std::vector<PrimitiveObject*> prims_raws;
        for (auto &prim: prims) {
            prims_raws.push_back(prim.get());
        }
        auto prim = primMerge(prims_raws);
        prim->userData()->set_("ResourceType", "Matrixes");
        prim->userData()->set_("stamp-change", "TotalChange");
        prim->userData()->set_("ObjectName", zeno_scene_name+"_m");
        scene->push_back(prim);
        dict_json[zeno_scene_name+"_m"] = index;
        index += 1;
    }
    dict->userData()->set_("json", dict_json.dump());
    auto prim_list_size = sceneSource->prim_list.size();
    dict->userData()->set_("prim_list_size", int(prim_list_size));

    scene->insert(scene->arr.begin(), dict);
    {
        auto scene_descriptor = std::make_unique<PrimitiveObject>();
        auto ud = scene_descriptor->userData();
        ud->set_("ResourceType", std::string("SceneDescriptor"));
        Json json;
        json["BasicRenderInstances"] = Json();
        json["DynamicRenderGroups"]["Objects"] = Json();
        json["StaticRenderGroups"]["Objects"] = Json();
        for (const auto &[path, prim]: sceneSource->prim_list) {
            json["BasicRenderInstances"][path]["Geom"] = path;
            json["StaticRenderGroups"]["Objects"][path] = Json::array({path+"_m"});
        }

        ud->set_("Scene", std::string(json.dump()));
        scene->push_back(scene_descriptor);
    }

    // prim list
    std::vector<std::string> prim_list_name;
    {
        std::vector<std::unique_ptr<PrimitiveObject>> prim_list;
        for (auto &[abc_path, p]: sceneSource->prim_list) {
            p->userData()->set_("ObjectName", abc_path);
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
        scene->arr[0]->userData()->set_("prim_list_name", json.dump());
    }
    // scene tree prim
    {
        auto scene_tree = std::make_unique<JsonObject>();
        scene_tree->json = sceneSource->to_json();
        scene->push_back(scene_tree);
    }
    return scene;
}
#endif
static void get_local_matrix_map(
    Json &json
    , std::string parent_path
    , SceneObject* scene
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
    scene->node_to_matrix[node_path+"_m"].push_back(mat);
    stn.matrix = node_path+"_m";
    stn.visibility = json["visibility"] == 0? 0 : 1;
    if (json.contains("mesh")) {
        stn.meshes.push_back(node_path + "/" + std::string(json["mesh"]));
    }

    for (auto i = 0; i < json["children_name"].size(); i++) {
        std::string child_name = json["children_name"][i];
        std::string child_path = node_path + '/' + child_name;
        if(json.contains(child_name)) {
        if (json[child_name].contains("instance_source_path")) {
            stn.children.push_back(json[child_name]["instance_source_path"]);
        }
        else {
            stn.children.push_back(child_path);
        }
    }
    }
    scene->scene_tree[node_path] = stn;

    for (auto i = 0; i < json["children_name"].size(); i++) {
        std::string child_name = json["children_name"][i];
        if(json.contains(child_name)) {
        if (!json[child_name].contains("instance_source_path")) {
            get_local_matrix_map(json[child_name], node_path, scene);
        }
    }
}
}

struct FormSceneTree : zeno::INode {
    void apply() override {
        auto sceneTree = std::make_unique<SceneObject>();
        auto scene_json = ZImpl(get_input2<JsonObject>("Scene Info"));
        sceneTree->root_name = "/ABC";
        auto prim_geom_list = get_input_ListObject("Geometry List");

        //如果prim_geom_list的上游节点标记为no-cache,那这里就不应该拿到prim_geom_list.
        //或者geom分支已经不脏的情况下，这里导出sceneTree是不会导出geometry的
        sceneTree->bNeedUpdateDescriptor = prim_geom_list && 
            (!prim_geom_list->m_impl->m_new_added.empty() || !prim_geom_list->m_impl->m_modify.empty());

        if (sceneTree->bNeedUpdateDescriptor) {
            for (auto p : prim_geom_list->get()) {
                auto geom = dynamic_cast<GeometryObject_Adapter*>(p);
                if (geom) {
                    const auto& [bmin, bmax] = zeno::geomBoundingBox(geom->m_impl.get());
                    geom->userData()->set_vec3f("_bboxMin", toAbiVec3f(bmin));
                    geom->userData()->set_vec3f("_bboxMax", toAbiVec3f(bmax));
                }

                auto abc_path = zsString2Std(p->userData()->get_string("abcpath_0"));
                p->userData()->set_string("ResourceType", "Mesh");

                if (geom) {
                    geom->userData()->set_string("ObjectName", stdString2zs(abc_path));
                    sceneTree->geom_list[abc_path] = 
                        safe_uniqueptr_cast<GeometryObject_Adapter>(geom->clone());
                }
            }
        }
        get_local_matrix_map(scene_json->json, "", sceneTree.get());

        if (get_input2_bool("flattened")) {
            sceneTree->flatten();
        }
        set_output("scene", std::move(sceneTree));
    }
};

ZENDEFNODE(FormSceneTree, {
    {
        {gParamType_JsonObject, "Scene Info"},
        {gParamType_List, "Geometry List"},
        {gParamType_Bool, "flattened", "1"}
    },
    {
        {gParamType_Scene, "scene"},
    },
    {},
    {
        "Scene"
    }
});


struct ConvertXformToMatrix : zeno::INode {
    void apply() override {
        auto xform = clone_input_PrimitiveObject("xform");
        AttrVector<vec3f> verts(xform->verts.size() * 4);
        {
            auto& t_attr = xform->verts.values;
            auto& r0_attr = xform->verts.attr<vec3f>("r0");
            auto& r1_attr = xform->verts.attr<vec3f>("r1");
            auto& r2_attr = xform->verts.attr<vec3f>("r2");
            for (auto i = 0; i < xform->verts.size(); i++) {
                auto r0 = r0_attr[i];
                auto r1 = r1_attr[i];
                auto r2 = r2_attr[i];
                auto t = t_attr[i];
                verts[0 + i * 4][0] = r0[0];
                verts[0 + i * 4][1] = r1[0];
                verts[0 + i * 4][2] = r2[0];
                verts[1 + i * 4][0] = t[0];
                verts[1 + i * 4][1] = r0[1];
                verts[1 + i * 4][2] = r1[1];
                verts[2 + i * 4][0] = r2[1];
                verts[2 + i * 4][1] = t[1];
                verts[2 + i * 4][2] = r0[2];
                verts[3 + i * 4][0] = r1[2];
                verts[3 + i * 4][1] = r2[2];
                verts[3 + i * 4][2] = t[2];
            }

        }
        xform->verts = verts;

        set_output("matrix", std::move(xform));
        }
};
ZENDEFNODE(ConvertXformToMatrix, {
    {
        {gParamType_Primitive, "xform"},
	},
	{
		{gParamType_Primitive, "matrix"},
	},
	{},
	{
		"Scene",
	},
	});

/*
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
static void scene_add_prefix(
    std::string path
    , std::vector<glm::mat4> xform
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
    std::unordered_map<std::string, std::unique_ptr<PrimitiveObject>> new_prim_list;
    for (auto& [key, value] : sceneObject->prim_list) {
        auto obj_name = value->userData().get2<std::string>("ObjectName");
        obj_name = path + obj_name;
        value->userData().set2("ObjectName", obj_name);
        new_prim_list[path + key] = value;
    }
    std::unordered_map<std::string, std::vector<glm::mat4>> new_node_to_matrix;
    for (const auto& [key, value] : sceneObject->node_to_matrix) {
        new_node_to_matrix[path + key] = value;
    }
    sceneObject->scene_tree = scene_tree;
    sceneObject->prim_list = new_prim_list;
    sceneObject->node_to_matrix = new_node_to_matrix;
    {
        std::string xform_name = path + "_m";
        if (xform.empty()) {
            xform.push_back(glm::mat4(1));
        }
        sceneObject->node_to_matrix[xform_name] = xform;
    }
}

struct MergeScene : zeno::INode {
    void apply() override {
        auto main_scene = get_scene_tree_from_list(ZImpl(get_input2<ListObject>("main_scene")));
        auto second_scene = get_scene_tree_from_list(ZImpl(get_input2<ListObject>("second_scene")));
        auto namespace1 = ZImpl(get_input2<std::string>("namespace1"));
        auto namespace2 = ZImpl(get_input2<std::string>("namespace2"));
        auto insert_path = ZImpl(get_input2<std::string>("insert_path"));

        auto append_path1 = (namespace1 == ""? "" : "/") + namespace1;
        if (append_path1.size()) {
            std::vector<glm::mat4> xform1;
            if (ZImpl(has_input2<PrimitiveObject>("xform1"))) {
                xform1 = get_xform_from_prim(ZImpl(get_input2<PrimitiveObject>("xform1")));
            }
            scene_add_prefix(append_path1, xform1, main_scene);
        }
        auto append_path2 = append_path1 + (insert_path == ""? "" : "/") + insert_path + (namespace2 == ""? "" : "/") + namespace2;
        std::vector<glm::mat4> xform2;
        if (ZImpl(has_input2<PrimitiveObject>("xform2"))) {
            xform2 = get_xform_from_prim(ZImpl(get_input2<PrimitiveObject>("xform2")));
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
        main_scene->type = get_input2<std::string>("type");
        auto scene = main_scene->to_structure();
        set_output("scene", std::move(scene));
    }
};

ZENDEFNODE( MergeScene, {
    {
        {gParamType_List, "main_scene"},
        {gParamType_List, "second_scene"},
        {"enum static dynamic", "type", "static"},
        {gParamType_String , "namespace1", ""},
        {gParamType_String, "namespace2", "namespace2"},
        {gParamType_String, "insert_path", ""},
        {gParamType_Primitive, "xform1"},
        {gParamType_Primitive, "xform2"},
    },
    {
        {gParamType_List, "scene"},
    },
    {},
    {
        "Scene",
    },
});
*/

static std::vector<glm::mat4> get_xform_from_prim(PrimitiveObject* prim) {
    std::vector<glm::mat4> mats;
    for (auto i = 0; i < prim->verts.size(); i++) {
        auto pos = prim->verts[i];
        auto r0 = prim->verts.add_attr<vec3f>("r0")[i];
        auto r1 = prim->verts.add_attr<vec3f>("r1")[i];
        auto r2 = prim->verts.add_attr<vec3f>("r2")[i];
        glm::mat4 mat;
        mat[0] = {r0[0], r0[1], r0[2], 0};
        mat[1] = {r1[0], r1[1], r1[2], 0};
        mat[2] = {r2[0], r2[1], r2[2], 0};
        mat[3] = {pos[0], pos[1], pos[2], 1};
        mats.push_back(mat);
    }
    return mats;
}
static std::vector<int> get_id_from_prim(std::unique_ptr<PrimitiveObject> prim) {
    std::vector<int> ids;
    if (prim->verts.attr_is<int>("id")) {
        ids = prim->attr<int>("id");
    }
    return ids;
}

static void scene_add_prefix_node(
    std::string prefix_node_name
    , std::vector<glm::mat4> xform
    , SceneObject* sceneObject
) {
    std::unordered_map<std::string, SceneTreeNode> scene_tree;
    for (const auto &[key, value]: sceneObject->scene_tree) {
        SceneTreeNode stn;
        stn.visibility = value.visibility;
        if (value.matrix.size()) {
            stn.matrix = prefix_node_name + value.matrix;
        }
        for (auto &child: value.children) {
            stn.children.push_back(prefix_node_name + child);
        }
        for (auto &mesh: value.meshes) {
            stn.meshes.push_back(prefix_node_name + mesh);
        }
        scene_tree[prefix_node_name + key] = stn;
    }
    std::unordered_map<std::string, std::unique_ptr<GeometryObject_Adapter>> new_prim_list;
    for (auto& [key, value] : sceneObject->geom_list) {
        auto obj_name = zsString2Std(value->userData()->get_string("ObjectName"));
        obj_name = prefix_node_name + obj_name;
        value->userData()->set_string("ObjectName", stdString2zs(obj_name));
        new_prim_list[prefix_node_name + key] = 
            safe_uniqueptr_cast<GeometryObject_Adapter>(value->clone());
    }
    std::unordered_map<std::string, std::vector<glm::mat4>> new_node_to_matrix;
    for (const auto& [key, value] : sceneObject->node_to_matrix) {
        new_node_to_matrix[prefix_node_name + key] = value;
    }
    sceneObject->scene_tree = scene_tree;
    sceneObject->geom_list = std::move(new_prim_list);
    sceneObject->node_to_matrix = new_node_to_matrix;


    {
        std::string xform_name = prefix_node_name + "_m";
        if (xform.empty()) {
            xform.push_back(glm::mat4(1));
        }
        sceneObject->node_to_matrix[xform_name] = xform;
        SceneTreeNode stn;
        stn.matrix = xform_name;
        stn.children.push_back(prefix_node_name + sceneObject->root_name);
        sceneObject->scene_tree[prefix_node_name] = stn;
    }
    sceneObject->root_name = prefix_node_name;
}

void merge_scene2_into_scene1(SceneObject* main_object, SceneObject* second_object, std::string insert_path) {
    for (const auto &[key, value]: second_object->scene_tree) {
        SceneTreeNode stn;
        stn.visibility = value.visibility;
        if (value.matrix.size()) {
            stn.matrix = insert_path + value.matrix;
        }
        for (auto &child: value.children) {
            stn.children.push_back(insert_path + child);
        }
        for (auto &mesh: value.meshes) {
            stn.meshes.push_back(insert_path + mesh);
        }
        main_object->scene_tree[insert_path + key] = stn;
    }
    for (auto& [key, value] : second_object->geom_list) {
        auto obj_name = zsString2Std(value->userData()->get_string("ObjectName"));
        obj_name = insert_path + obj_name;
        value->userData()->set_string("ObjectName", stdString2zs(obj_name));
        main_object->geom_list[insert_path + key] = safe_uniqueptr_cast<GeometryObject_Adapter>(value->clone());
    }

    for (const auto& [key, value] : second_object->node_to_matrix) {
        main_object->node_to_matrix[insert_path + key] = value;
    }
    if (!insert_path.empty()) {
        main_object->scene_tree[insert_path].children.push_back(insert_path + second_object->root_name);
    }
}

struct MergeScene : zeno::INode {
    void apply() override {
        auto main_scene = safe_uniqueptr_cast<SceneObject>(clone_input("Main Scene"));
        auto namespace1 = zsString2Std(get_input2_string("namespace1"));
        bool mainscene_dirty = is_upstream_dirty("Main Scene");

        if (namespace1.size()) {
            if (!zeno::starts_with(namespace1, "/")) {
                namespace1 = "/" + namespace1;
            }
            std::vector<glm::mat4> xform1;
            if (has_link_input("xform1")) {
                xform1 = zeno::reflect::any_cast<std::vector<glm::mat4>>(m_pAdapter->get_param_result("xform1"));
            }
            scene_add_prefix_node(namespace1, xform1, main_scene.get());
        }
        if (has_link_input("Second Scene")) {
            auto second_scene = dynamic_cast<SceneObject*>(get_input("Second Scene"));
            auto scene_obj_key = zsString2Std(second_scene->key());
            auto secondscene_dirty = is_upstream_dirty("Second Scene");

            if (!secondscene_dirty) {
                //说明是添加过的场景，没有要变更的部分，先不加到scenetree
                //TODO: 如果更改namespace，估计要重构scenetree，先不讨论这种情况
            }
            else {
                //上游是脏的，但second_scene可能是全部都要更新，也有可能部分更新，看bNeedUpdateDescriptor
                auto namespace2 = zsString2Std(get_input2_string("namespace2"));
                if (namespace2.size()) {
                    if (!zeno::starts_with(namespace2, "/")) {
                        namespace2 = "/" + namespace2;
                    }
                    std::vector<glm::mat4> xform2;
                    if (has_link_input("xform2")) {
                        xform2 = zeno::reflect::any_cast<std::vector<glm::mat4>>(m_pAdapter->get_param_result("xform2"));
                    }
                    scene_add_prefix_node(namespace2, xform2, second_scene);
                }
                auto insert_path = zsString2Std(get_input2_string("insert_path"));
                if (insert_path.size()) {
                    if (!zeno::starts_with(insert_path, "/")) {
                        insert_path = "/" + insert_path;
                    }
                }
                if (mainscene_dirty && secondscene_dirty) {
                    merge_scene2_into_scene1(main_scene.get(), second_scene, namespace1 + insert_path);
                }
                else if (!mainscene_dirty && secondscene_dirty) {
                    //只有secondscene是脏的，但要考虑namespace
                    if (!second_scene->bNeedUpdateDescriptor) {
                        main_scene = safe_uniqueptr_cast<SceneObject>(second_scene->clone());
                    }
                    else {
                        //TODO: 要重构场景的情况
                    }
                }
            }
        }
        set_output("scene", std::move(main_scene));
    }
};

ZENDEFNODE( MergeScene, {
    {
        {gParamType_Scene, "Main Scene"},
        {gParamType_Scene, "Second Scene"},
        {gParamType_String, "insert_path", ""},
        {gParamType_String, "namespace1", ""},
        {gParamType_ListOfMat4, "xform1"},
        {gParamType_String, "namespace2", "namespace2"},
        {gParamType_ListOfMat4, "xform2"},
    },
    {
        {gParamType_Scene, "scene"},
    },
    {},
    {
        "Scene",
    },
});

struct SceneRootRename : zeno::INode {
    void apply() override {
        auto scene_tree = dynamic_cast<SceneObject*>(get_input("scene"));
        auto new_root_name = zsString2Std(get_input2_string("new_root_name"));
        if (zeno::ends_with(new_root_name, "/")) {
            new_root_name.pop_back();
        }
        if (new_root_name.empty()) {
            new_root_name = scene_tree->root_name;
        }
        if (zeno::starts_with(new_root_name, "/") == false) {
            new_root_name = "/" + new_root_name;
        }
        std::vector<glm::mat4> root_xform;
        if (ZImpl(has_link_input("xform"))) {
            root_xform = zeno::reflect::any_cast<std::vector<glm::mat4>>(m_pAdapter->get_param_result("xform"));
        }

        auto new_scene_tree = scene_tree->root_rename(new_root_name, root_xform);
        set_output("scene", std::move(new_scene_tree));
    }
};

ZENDEFNODE( SceneRootRename, {
    {
        {gParamType_Scene, "scene"},
        {gParamType_String, "new_root_name", "new_scene"},
        {gParamType_ListOfMat4, "xform"}
    },
    {
        {gParamType_Scene, "scene"}
    },
    {},
    {
        "Scene"
    }
});

struct MergeMultiScenes : zeno::INode {
    void apply() override {
        auto main_scene = std::make_unique<SceneObject>();
        main_scene->root_name = zsString2Std(get_input2_string("root_name"));
        if (!main_scene->root_name.empty())
        {
            if (zeno::starts_with(main_scene->root_name, "/") == false) {
                main_scene->root_name = "/" + main_scene->root_name;
            }
            {
                SceneTreeNode root_node;
                root_node.matrix = main_scene->root_name + "_m";
                main_scene->node_to_matrix[root_node.matrix] = { glm::mat4(1) };
                main_scene->scene_tree[main_scene->root_name] = root_node;
            }
        }

        std::unordered_map<std::string, int> sub_root_names;
        if (has_input("Scene List")) {
            auto input_scene_list = std::make_unique<ListObject>();
            auto scene_list = get_input_ListObject("Scene List");

            main_scene->bNeedUpdateDescriptor = false;
            for (auto i = 0; i < scene_list->size(); i++) {
                auto second_scene = dynamic_cast<SceneObject*>(scene_list->m_impl->m_objects[i].get());
                auto scene_obj_key = zsString2Std(second_scene->key());

                if (scene_list->m_impl->m_modify.find(scene_obj_key) != scene_list->m_impl->m_modify.end() ||
                    scene_list->m_impl->m_new_added.find(scene_obj_key) != scene_list->m_impl->m_new_added.end()) {
                    continue;   //说明是添加过的场景，没有要变更的部分，先不加到scenetree
                }
                else {
                    //上游脏了，但不一定需要更新mesh
                }
                auto sub_root_name = second_scene->root_name;
                sub_root_names[sub_root_name] += 1;
                if (sub_root_names[sub_root_name] > 1) {
                    zeno::log_warn("MergeMultiScenes: root_name {} is duplicate!", sub_root_name);
                }
                merge_scene2_into_scene1(main_scene.get(), second_scene, main_scene->root_name);
                main_scene->bNeedUpdateDescriptor |= second_scene->bNeedUpdateDescriptor;
            }
        }
        set_output("scene", std::move(main_scene));
    }
};
ZENDEFNODE( MergeMultiScenes, {
    {
        {gParamType_List, "Scene List"},
        {gParamType_String, "root_name", "dummyRoot"},
    },
    {
        {gParamType_Scene, "scene"},
    },
    {},
    {
        "Scene",
    },
});

struct FlattenSceneTree : zeno::INode {
    void apply() override {
        auto scene = safe_uniqueptr_cast<SceneObject>(clone_input("scene"));
        scene->flatten();
        set_output("scene", std::move(scene));
    }
};

ZENDEFNODE( FlattenSceneTree, {
    {
        {gParamType_Scene, "scene"},
    },
    {
        {gParamType_Scene, "scene"}
    },
    {},
    {
        "Scene",
    },
});


struct MakeXform : zeno::INode {
    void apply() override {
        auto translate = ZImpl(get_input2<vec3f>("translate"));
        auto eulerXYZ = ZImpl(get_input2<vec3f>("eulerXYZ"));
        auto scale = ZImpl(get_input2<vec3f>("scale"));
        glm::mat4 matScale  = glm::scale( glm::vec3(scale[0], scale[1], scale[2] ));


        auto order = ZImpl(get_input2<std::string>("EulerRotationOrder"));
        auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

        auto measure = ZImpl(get_input2<std::string>("EulerAngleMeasure"));
        auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

        glm::vec3 eularAngleXYZ = glm::vec3(eulerXYZ[0], eulerXYZ[1], eulerXYZ[2]);
        glm::mat4 matRotate = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        auto mat = matRotate * matScale;

        auto xform = std::make_unique<PrimitiveObject>();
        xform->resize(1);
        xform->verts[0] = translate;
        xform->verts.add_attr<vec3f>("r0")[0] = {mat[0][0], mat[0][1], mat[0][2]};
        xform->verts.add_attr<vec3f>("r1")[0] = {mat[1][0], mat[1][1], mat[1][2]};
        xform->verts.add_attr<vec3f>("r2")[0] = {mat[2][0], mat[2][1], mat[2][2]};
        set_output("xform", std::move(xform));
    }
};

ZENDEFNODE( MakeXform, {
    {
        {gParamType_Vec3f , "translate", "0, 0, 0"},
        {gParamType_Vec3f, "eulerXYZ", "0, 0, 0"},
        {gParamType_Vec3f, "scale", "1, 1, 1"},
    },
    {
        {gParamType_Primitive, "xform"},
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
        auto scene = safe_uniqueptr_cast<SceneObject>(clone_input("scene"));
        auto json = Json::parse(scene->to_json());
        json["type"] = zsString2Std(get_input2_string("type"));
        json["matrixMode"] = zsString2Std(get_input2_string("matrixMode"));
        scene->from_json(json.dump());
        set_output("scene", std::move(scene));
    }
};

ZENDEFNODE( MarkSceneState, {
    {
        {gParamType_Scene, "scene"},
        {"enum static dynamic", "type", "static"},
        {"enum UnChanged TotalChange", "matrixMode", "TotalChange"},
    },
    {
        {gParamType_Scene, "scene"},
    },
    {
    },
    {
        "Scene",
    },
});

struct GetNodeIds : zeno::INode {
    void apply() override {
        auto geom = get_input_Geometry("Input");
        std::vector<int> ids;
        if (geom->has_point_attr("id")) {
            ids = geom->m_impl->get_attrs<int>(zeno::ATTR_POINT, "id");
        }
        m_pAdapter->set_primitive_output("node_ids", ids);
    }
};

ZENDEFNODE(GetNodeIds, {
    {
        {gParamType_Geometry, "Input"}
    },
    {
        {gParamType_IntList, "node_ids"}
    },
    {},
    {
        "Scene",
    },
});

struct SetNodeXform : zeno::INode {
    void apply() override {
        auto scene = safe_uniqueptr_cast<SceneObject>(clone_input("scene"));
        auto node = zsString2Std(get_input2_string("node"));
        if (!zeno::starts_with(node, "/")) {
            node = "/" + node;
}
        auto st = Json::parse(scene->to_json());
        auto &node_to_matrix = st["node_to_matrix"];
        if (has_input("xforms")) {
            auto xforms = zeno::reflect::any_cast<std::vector<glm::mat4>>(m_pAdapter->get_param_result("xforms"));
            Json mats = Json::array();
            for (const auto &xform: xforms) {
                Json matrix = Json::array();
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(xform[0][j]);
                }
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(xform[1][j]);
                }
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(xform[2][j]);
                }
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(xform[3][j]);
                }
                mats.push_back(matrix);
            }
            node_to_matrix[node + "_m"] = mats;
            if (has_link_input("node_ids")) {
                auto ids = zeno::reflect::any_cast<std::vector<int>>(m_pAdapter->get_param_result("node_ids"));
                if (ids.size()) {
                    Json ids_json = Json::array();
                    for (auto id : ids) {
                        ids_json.push_back(id);
                    }
                    st["node_to_id"][node + "_m"] = ids_json;
                }
            }
            //xforms除了变换还把id搞进来？什么鬼
            /*
            auto ids = get_id_from_prim(get_input_PrimitiveObject("xforms"));
            if (ids.size()) {
                Json ids_json = Json::array();
                for (auto id: ids) {
                    ids_json.push_back(id);
                }
                st["node_to_id"][node + "_m"] = ids_json;
            }
            */
        }
        else {
            auto index = get_input2_int("index");
            auto r0 = get_input2_vec3f("r0");
            auto r1 = get_input2_vec3f("r1");
            auto r2 = get_input2_vec3f("r2");
            auto t  = get_input2_vec3f("t");

            if (node_to_matrix.contains(node + "_m")) {
                Json matrix = Json::array();
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(r0[j]);
                }
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(r1[j]);
                }
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(r2[j]);
                }
                for (auto j = 0; j < 3; j++) {
                    matrix.push_back(t[j]);
                }
                node_to_matrix[node + "_m"][index] = matrix;
            }
        }
        scene->from_json(st.dump());
        set_output("scene", std::move(scene));
    }
};

ZENDEFNODE( SetNodeXform, {
    {
        {gParamType_Scene, "scene"},
        {gParamType_String, "node", ""},
        {gParamType_Int, "index", "0"},
        {gParamType_IntList, "node_ids"},
        {gParamType_Vec3f, "r0", "1, 0, 0"},
        {gParamType_Vec3f, "r1", "0, 1, 0"},
        {gParamType_Vec3f, "r2", "0, 0, 1"},
        {gParamType_Vec3f, "t", "0, 0, 0"},
        {gParamType_ListOfMat4, "xforms"}
    },
    {
        {gParamType_Scene, "scene"},
    },
    {
    },
    {
        "Scene",
    },
});

struct SetNodeId : zeno::INode {
    void apply() override {
        auto scene = safe_uniqueptr_cast<SceneObject>(clone_input("scene"));
        auto node = zsString2Std(get_input2_string("node"));
        if (!zeno::starts_with(node, "/")) {
            node = "/" + node;
        }
        auto st = Json::parse(scene->to_json());
        auto &node_to_id = st["node_to_id"];
        {
            auto index = get_input2_int("index");
            auto id = get_input2_int("id");

            if (node_to_id.contains(node + "_m")) {
                node_to_id[node + "_m"][index] = id;
            }
        }
        scene->from_json(st.dump());
        set_output("scene", std::move(scene));
    }
};

ZENDEFNODE( SetNodeId, {
    {
        {gParamType_Scene, "scene"},
        {gParamType_String, "node", ""},
        {gParamType_Int, "index", "0"},
        {gParamType_Int, "id", "0"},
    },
    {
        {gParamType_Scene, "scene"},
    },
    {
    },
    {
        "Scene",
    },
});

struct SetSceneXform : zeno::INode {
    void apply() override {
        auto scene_tree = safe_uniqueptr_cast<SceneObject>(clone_input("scene"));
        auto xformsList = zeno::reflect::any_cast<std::vector<std::string>>(ZImpl(get_param_result("xformsList")));
        for (const auto &xforms_str: xformsList) {
            if (xforms_str.size()) {
                auto xforms = Json::parse(xforms_str);
                for (const auto& [node_name, mat]: xforms.items()) {
                    auto &stn = scene_tree->scene_tree.at(node_name);
                    if (stn.matrix.empty()) {
                        stn.matrix = node_name + "_m";
                    }
                    auto m = glm::mat4(1);
                    for (auto i = 0; i < 4; i++) {
                        for (auto j = 0; j < 3; j++) {
                            m[i][j] = float(mat[i * 3 + j]);
                        }
                    }
                    scene_tree->node_to_matrix[stn.matrix] = {m};
                }
            }
        }
        //auto scene = scene_tree->to_list();
        set_output("scene", std::move(scene_tree));
    }
};

ZENDEFNODE( SetSceneXform, {
    {
        {gParamType_Scene, "scene"},
        {gParamType_StringList, "xformsList"},
//        {"multiline_string", "xformsInfo", ""},
    },
    {
        {gParamType_Scene, "scene"},
    },
    {
    },
    {
        "Scene",
    },
});

struct SetResourceType : zeno::INode {
    void apply() override {
        auto obj = clone_input("Input");
        auto ud = obj->userData();
        if (!ud->has_string("ResourceType")) {
            ud->set_string("ResourceType", get_input2_string("ResourceType"));
        }
        if (!ud->has_string("ObjectName")) {
            ud->set_string("ObjectName", get_input2_string("ObjectName"));
        }
        set_output("Output", std::move(obj));
    }
};
ZENDEFNODE(SetResourceType, {
    {
        {gParamType_IObject, "Input"},
        {"enum Mesh Matrixes SceneDescriptor", "ResourceType", "Mesh"},
        {gParamType_String, "ObjectName", ""},
        {gParamType_String, "changeHint", ""}
    },
    {
        {gParamType_IObject, "Output"},
    },
    {
    },
    {
        "Scene",
    },
    });

struct MakeSceneNode : zeno::INode {
    void apply() override {
        auto scene_tree = std::make_unique<SceneObject>();
        scene_tree->root_name = zsString2Std(get_input2_string("root_name"));
        if (!zeno::starts_with(scene_tree->root_name, "/")) {
            scene_tree->root_name = "/" + scene_tree->root_name;
        }

        auto geom = clone_input_Geometry("prim");
        auto bbox = zeno::geomBoundingBox2(geom->m_impl.get());
        if (bbox.has_value()) {
            vec3f bmin = {};
            vec3f bmax = {};
            std::tie(bmax, bmax) = bbox.value();
            geom->userData()->set_vec3f("_bboxMin", toAbiVec3f(bmin));
            geom->userData()->set_vec3f("_bboxMax", toAbiVec3f(bmax));
        }

        SceneTreeNode root_node;
        root_node.matrix = scene_tree->root_name + "_m";
        scene_tree->node_to_matrix[root_node.matrix] = {glm::mat4(1)};
        auto obj_name = zsString2Std(geom->userData()->get_string("ObjectName"));
        scene_tree->geom_list[obj_name] = std::move(geom);
        {
            std::string node_name = scene_tree->root_name + '/' + obj_name + "_node";
            SceneTreeNode prim_node;
            prim_node.meshes.push_back(obj_name);
            root_node.children.push_back(node_name);
            prim_node.matrix = node_name + "_m";
            scene_tree->scene_tree[node_name] = prim_node;
            scene_tree->node_to_matrix[prim_node.matrix] = {glm::mat4(1)};
        }

        scene_tree->scene_tree[scene_tree->root_name] = root_node;

        if (has_link_input("xforms")) {
            auto xforms = zeno::reflect::any_cast<std::vector<glm::mat4>>(m_pAdapter->get_param_result("xforms"));
            scene_tree->node_to_matrix[scene_tree->root_name + "_m"] = xforms;
        }
        //auto scene = scene_tree->to_list();
        set_output("scene", std::move(scene_tree));
    }
};
ZENDEFNODE( MakeSceneNode, {
    {
        {gParamType_Geometry, "prim"},
        {gParamType_String, "root_name", "/ABC"},
        {gParamType_ListOfMat4, "xforms"},
    },
    {
        {gParamType_Scene, "scene"},
    },
    {
    },
    {
        "Scene",
    },
});


struct TestSceneNodeCopy : zeno::INode {
    void apply() override {
        set_output("scene", clone_input("scene"));
    }
};

ZENDEFNODE(TestSceneNodeCopy, {
    {
        {gParamType_Scene, "scene"}
    },
    {
        {gParamType_Scene, "scene"}
    },
    {},
    {
        "Scene",
    },
});

struct ObjectToXforms : zeno::INode {
    void apply() override {
        std::vector<glm::mat4> xforms;
        zany obj = clone_input("IObject");
        if (auto geoObj = dynamic_cast<GeometryObject_Adapter*>(obj.get())) {
            xforms = get_xform_from_prim(geoObj->toPrimitiveObject().get());
        } else if (auto primObj = dynamic_cast<PrimitiveObject*>(obj.get())) {
            xforms = get_xform_from_prim(primObj);
        }
        m_pAdapter->set_primitive_output("xforms", xforms);
    }
};
ZENDEFNODE(ObjectToXforms, {
    {
        {gParamType_IObject, "IObject"},
    },
    {
        {gParamType_ListOfMat4, "xforms"},
    },
    {
    },
    {
        "Scene",
    },
    });
}
