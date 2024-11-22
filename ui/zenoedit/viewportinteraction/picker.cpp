#include "picker.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"

#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/graphsmanagment.h>

#include <zenovis/Scene.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/funcs/ObjectGeometryInfo.h>

#include <sstream>
#include <functional>
#include <regex>
#include <utility>

using std::string;
using std::unordered_map;
using std::unordered_set;
using std::function;
namespace zeno {

//void Picker::pickWithRay(QVector3D ray_ori, QVector3D ray_dir,
//                         const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete) {
//    auto scene = Zenovis::GetInstance().getSession()->get_scene();
//    float min_t = std::numeric_limits<float>::max();
//    std::string name("");
//    for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
//        zeno::vec3f ro(ray_ori[0], ray_ori[1], ray_ori[2]);
//        zeno::vec3f rd(ray_dir[0], ray_dir[1], ray_dir[2]);
//        zeno::vec3f bmin, bmax;
//        if (zeno::objectGetBoundingBox(ptr, bmin, bmax) ){
//            if (auto ret = ray_box_intersect(bmin, bmax, ro, rd)) {
//                float t = *ret;
//                if (t < min_t) {
//                    min_t = t;
//                    name = key;
//                }
//            }
//        }
//    }
//    if (scene->selected.count(name) > 0) {
//        scene->selected.erase(name);
//        on_delete(name);
//    }
//    else {
//        scene->selected.insert(name);
//        on_add(name);
//    }
//    onPrimitiveSelected();
//}

//void Picker::pickWithRay(QVector3D cam_pos, QVector3D left_up, QVector3D left_down, QVector3D right_up, QVector3D right_down,
//                         const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete) {
//    auto scene = Zenovis::GetInstance().getSession()->get_scene();
//
//    auto left_normWS = QVector3D::crossProduct(left_down, left_up);
//    auto right_normWS = QVector3D::crossProduct(right_up, right_down);
//    auto up_normWS = QVector3D::crossProduct(left_up, right_up);
//    auto down_normWS = QVector3D::crossProduct(right_down, left_down);
//
//    std::vector<std::string> passed_prim;
//    for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
//        zeno::vec3f c;
//        float radius;
//        if (zeno::objectGetFocusCenterRadius(ptr, c, radius)) {
//            bool passed = test_in_selected_bounding(QVector3D(c[0], c[1], c[2]), cam_pos, left_normWS,
//                                                    right_normWS, up_normWS, down_normWS);
//            if (passed) {
//                passed_prim.push_back(key);
//                string t;
//                on_add(key);
//            }
//        }
//    }
//    scene->selected.insert(passed_prim.begin(), passed_prim.end());
//    onPrimitiveSelected();
//}

Picker::Picker(ViewportWidget *pViewport)
    : m_pViewport(pViewport)
    , draw_special_buffer_mode(false)
{
}

void Picker::initialize()
{
    auto scene = this->get_scene();
    ZASSERT_EXIT(scene);
    picker = zenovis::makeFrameBufferPicker(scene);
}

zenovis::Scene* Picker::get_scene() const
{
    auto sess = m_pViewport->getSession();
    ZASSERT_EXIT(sess, nullptr);
    return sess->get_scene();
}

void Picker::pick_pixel(int x, int y, SELECTION_MODE mode) {
    auto scene = this->get_scene();
    ZASSERT_EXIT(scene);
    auto &selected_prims = scene->selected;
    auto &selected_elements = scene->selected_elements;
    // qDebug() << scene->select_mode;
    // scene->select_mode = zenovis::PICK_MODE::PICK_FACE;
    auto selected = picker->getPicked(x, y);

    if (scene->get_select_mode() == zenovis::PICK_MODE::PICK_OBJECT) {
        if (selected.empty()) {
            selected_prims.clear();
            return;
        }
        if (selected_prims.count(selected) > 0) {
            selected_prims.erase(selected);
        } else {
            selected_prims.clear();
            selected_prims.insert(selected);
        }
    }
    else if (scene->get_select_mode() == zenovis::PICK_MODE::PICK_FACE_ATTR) {
        load_from_str(selected, scene->get_select_mode(), SELECTION_MODE::NORMAL);
        if (picked_elem_attrs_callback) picked_elem_attrs_callback(mode);
    }
    else {
        load_from_str(selected, scene->get_select_mode(), mode);
        if (picked_elems_callback) picked_elems_callback();
    }
    // qDebug() << "clicked (" << x << "," << y <<") selected " << selected_obj.c_str();
    // scene->selected.insert(selected_obj);
    // onPrimitiveSelected();
}

void Picker::pick_rect(int x0, int y0, int x1, int y1, SELECTION_MODE mode) {
//    zeno::log_info("rect: {}, {}, {}, {}", x0, y0, x1, y1);
    auto scene = this->get_scene();
    ZASSERT_EXIT(scene);
    if (scene->get_select_mode() == zenovis::PICK_MODE::PAINT) {
        auto selected = picker->getPaintPicked(x0, y0, x1, y1);
        if (paint_elems_callback) paint_elems_callback(selected, zeno::vec4f(x0, y0, x1, y1));
    }
    else if (scene->get_select_mode() == zenovis::PICK_MODE::PICK_OBJECT) {
        auto selected = picker->getPicked(x0, y0, x1, y1);
        if (selected.empty()) {
            scene->selected.clear();
            return;
        }
        load_from_str(selected, zenovis::PICK_MODE::PICK_OBJECT, SELECTION_MODE::NORMAL);
    }
    else if (scene->get_select_mode() == zenovis::PICK_MODE::PICK_FACE_ATTR) {
        auto selected = picker->getPicked(x0, y0, x1, y1);
        load_from_str(selected, scene->get_select_mode(), SELECTION_MODE::NORMAL);
        if (picked_elem_attrs_callback) picked_elem_attrs_callback(mode);
    }
    else {
        auto selected = picker->getPicked(x0, y0, x1, y1);
        load_from_str(selected, scene->get_select_mode(), mode);
        if (picked_elems_callback) picked_elems_callback();
    }
}

void Picker::pick_depth(int x, int y) {
    auto depth = picker->getDepth(x, y);
    picked_depth_callback(depth, x, y);
    qDebug() << "picker: " << depth;
}

string Picker::just_pick_prim(int x, int y) {
    auto scene = this->get_scene();
    ZASSERT_EXIT(scene, "");

    auto store_mode = scene->get_select_mode();
    scene->set_select_mode(zenovis::PICK_MODE::PICK_OBJECT);
    auto res = picker->getPicked(x, y);
    scene->set_select_mode(store_mode);
    return res;
}

void Picker::load_from_str(const string& str, zenovis::PICK_MODE mode, SELECTION_MODE sel_mode) {
    if (str.empty()) return;
    auto scene = this->get_scene();
    ZASSERT_EXIT(scene);
    auto &selected_prims = scene->selected;
    auto &selected_elements = scene->selected_elements;
    // parse selected string
    std::regex reg(" ");
    std::sregex_token_iterator p(str.begin(), str.end(), reg, -1);
    std::sregex_token_iterator end;

    if (mode == zenovis::PICK_MODE::PICK_OBJECT) {
        while (p != end) {
            selected_prims.insert(*p);
            p++;
        }
    }
    else {
        if (sel_mode == SELECTION_MODE::NORMAL) {
            selected_elements.clear();
        }
        while (p != end) {
            string result = *p++;
            // qDebug() << result.c_str();
            auto t = result.find_last_of(':');
            auto obj_id = result.substr(0, t);
            std::stringstream ss;
            ss << result.substr(t+1);
            int elem_id; ss >> elem_id;
            if (selected_elements.find(obj_id) != selected_elements.end()) {
                auto &elements = selected_elements[obj_id];
                if (sel_mode == SELECTION_MODE::REMOVE)
                    elements.erase(elem_id);
                else
                    elements.insert(elem_id);
            } else selected_elements[obj_id] = {elem_id};
        }
    }
}

void Picker::focus(const string& prim_name) {
    picker->focus(prim_name);
}

void Picker::set_picked_depth_callback(std::function<void(float, int, int)> callback) {
    picked_depth_callback = std::move(callback);
}

void Picker::set_picked_elems_callback(function<void()> callback) {
    picked_elems_callback = std::move(callback);
}

void Picker::set_picked_elem_attrs_callback(function<void(SELECTION_MODE)> callback) {
    picked_elem_attrs_callback = std::move(callback);
}

void Picker::set_paint_elems_callback(function<void(std::unordered_map<std::string, std::unordered_map<uint32_t, zeno::vec2f>>, zeno::vec4f)> callback) {
    paint_elems_callback = std::move(callback);
}

bool Picker::get_draw_special_buffer_mode() const {
    return draw_special_buffer_mode;
}

void Picker::set_draw_special_buffer_mode(bool enable) {
    draw_special_buffer_mode = enable;
}

void Picker::load_painter_attr(const std::string& prim_name, std::map<int, float> attr) {
    auto scene = get_scene();
    auto prims_shared = scene->objectsMan->pairsShared<zeno::PrimitiveObject>();
    for (const auto& [k, v] : prims_shared) {
        if (prim_name == k) {
            scene->painter_data[prim_name].resize(v->verts.size());
            auto &data = scene->painter_data[prim_name];
            for (auto [k, v] : attr) {
                data[k] = {v, 0, 0};
            }
        }
    }
}

std::optional<float> ray_box_intersect(
    zeno::vec3f const &bmin,
    zeno::vec3f const &bmax,
    zeno::vec3f const &ray_pos,
    zeno::vec3f const &ray_dir
) {
    //objectGetBoundingBox(IObject *ptr, vec3f &bmin, vec3f &bmax);

    auto &min = bmin;
    auto &max = bmax;
    auto &p = ray_pos;
    auto &d = ray_dir;
    //auto &t = t;

    float t1 = (min[0] - p[0]) / (CMP(d[0], 0.0f) ? 0.00001f : d[0]);
    float t2 = (max[0] - p[0]) / (CMP(d[0], 0.0f) ? 0.00001f : d[0]);
    float t3 = (min[1] - p[1]) / (CMP(d[1], 0.0f) ? 0.00001f : d[1]);
    float t4 = (max[1] - p[1]) / (CMP(d[1], 0.0f) ? 0.00001f : d[1]);
    float t5 = (min[2] - p[2]) / (CMP(d[2], 0.0f) ? 0.00001f : d[2]);
    float t6 = (max[2] - p[2]) / (CMP(d[2], 0.0f) ? 0.00001f : d[2]);

    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    // if tmax < 0, ray is intersecting AABB
    // but entire AABB is behing it's origin
    if (tmax < 0) {
        return std::nullopt;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax) {
        return std::nullopt;
    }

    float t_result = tmin;

    // If tmin is < 0, tmax is closer
    if (tmin < 0.0f) {
        t_result = tmax;
    }
    //zeno::vec3f  final_t = p + d * t_result;
    return t_result;
}


bool test_in_selected_bounding(
    QVector3D centerWS,
    QVector3D cam_posWS,
    QVector3D left_normWS,
    QVector3D right_normWS,
    QVector3D up_normWS,
    QVector3D down_normWS
) {
    QVector3D dir =  centerWS - cam_posWS;
    dir.normalize();
    bool left_test = QVector3D::dotProduct(dir, left_normWS) > 0;
    bool right_test = QVector3D::dotProduct(dir, right_normWS) > 0;
    bool up_test = QVector3D::dotProduct(dir, up_normWS) > 0;
    bool down_test = QVector3D::dotProduct(dir, down_normWS) > 0;
    return left_test && right_test && up_test && down_test;
}
}