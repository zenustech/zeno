#include "viewportpicker.h"
#include "zenovis.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"

#include <zenovis/Scene.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/funcs/ObjectGeometryInfo.h>

#include <functional>
#include <regex>

namespace zeno {

void Picker::pickWithRay(QVector3D ray_ori, QVector3D ray_dir,
                         const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete) {
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    float min_t = std::numeric_limits<float>::max();
    std::string name("");
    for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
        zeno::vec3f ro(ray_ori[0], ray_ori[1], ray_ori[2]);
        zeno::vec3f rd(ray_dir[0], ray_dir[1], ray_dir[2]);
        zeno::vec3f bmin, bmax;
        if (zeno::objectGetBoundingBox(ptr, bmin, bmax) ){
            if (auto ret = ray_box_intersect(bmin, bmax, ro, rd)) {
                float t = *ret;
                if (t < min_t) {
                    min_t = t;
                    name = key;
                }
            }
        }
    }
    if (scene->selected.count(name) > 0) {
        scene->selected.erase(name);
        on_delete(name);
    }
    else {
        scene->selected.insert(name);
        on_add(name);
    }
    onPrimitiveSelected();
}

void Picker::pickWithRay(QVector3D cam_pos, QVector3D left_up, QVector3D left_down, QVector3D right_up, QVector3D right_down,
                         const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete) {
    auto scene = Zenovis::GetInstance().getSession()->get_scene();

    auto left_normWS = QVector3D::crossProduct(left_down, left_up);
    auto right_normWS = QVector3D::crossProduct(right_up, right_down);
    auto up_normWS = QVector3D::crossProduct(left_up, right_up);
    auto down_normWS = QVector3D::crossProduct(right_down, left_down);

    std::vector<std::string> passed_prim;
    for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
        zeno::vec3f c;
        float radius;
        if (zeno::objectGetFocusCenterRadius(ptr, c, radius)) {
            bool passed = test_in_selected_bounding(QVector3D(c[0], c[1], c[2]), cam_pos, left_normWS,
                                                    right_normWS, up_normWS, down_normWS);
            if (passed) {
                passed_prim.push_back(key);
                string t;
                on_add(key);
            }
        }
    }
    scene->selected.insert(passed_prim.begin(), passed_prim.end());
    onPrimitiveSelected();
}

void Picker::pickWithFrameBuffer(int x, int y, const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete) {
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    if (!picker) picker = zenovis::makeFrameBufferPicker(scene);
    picker->draw();
    auto selected_obj = picker->getPicked(x, y);
    if (scene->selected.count(selected_obj) > 0) {
        scene->selected.erase(selected_obj);
        on_delete(selected_obj);
    }
    else {
        scene->selected.insert(selected_obj);
        on_add(selected_obj);
    }
    // qDebug() << "clicked (" << x << "," << y <<") selected " << selected_obj.c_str();
    // scene->selected.insert(selected_obj);
    onPrimitiveSelected();
}

void Picker::pickWithFrameBuffer(int x0, int y0, int x1, int y1,
                                 const std::function<void(string)>& on_add, const std::function<void(string)>& on_delete) {
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    if (!picker) picker = zenovis::makeFrameBufferPicker(scene);
    picker->draw();
    auto selected_objs = picker->getPicked(x0, y0, x1, y1);
    // qDebug() << "clicked (" << x0 << "," << y0 <<  ") to (" << x1 << "," << y1 << ") selected " << selected_obj.c_str();

    // parse selected string
    std::regex reg(" ");
    std::sregex_token_iterator p(selected_objs.begin(), selected_objs.end(), reg, -1);
    std::sregex_token_iterator end;
    while (p != end) {
        scene->selected.insert(*p);
        on_add(*p);
        p++;
    }
    onPrimitiveSelected();
}

void Picker::onPrimitiveSelected() {
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    mainWin->onPrimitiveSelected(scene->selected);
}

static std::optional<float> ray_box_intersect(
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


static bool test_in_selected_bounding(
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