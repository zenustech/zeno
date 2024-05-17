#ifndef __VIEWPORT_TRANSFORM_H__
#define __VIEWPORT_TRANSFORM_H__
#include "zenoapplication.h"
#include <viewport/zenovis.h>
#include <zenomodel/include/graphsmanagment.h>
#include <zeno/types/PrimitiveObject.h>
#include "nodesync.h"

#include <QtWidgets>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <unordered_map>

class ViewportWidget;

namespace zeno {

enum class TransOpt {
    NONE,
    TRANSLATE,
    ROTATE,
    SCALE
};

class FakeTransformer {
public:
    FakeTransformer(ViewportWidget* viewport);
    void addObjects(const std::unordered_set<std::string>& names);
    bool calcTransformStart(glm::vec3 ori, glm::vec3 dir, glm::vec3 front);
    bool clickedAnyHandler(QVector3D ori, QVector3D dir, glm::vec3 front);
    bool hoveredAnyHandler(QVector3D ori, QVector3D dir, glm::vec3 front);
    void transform(QVector3D camera_pos, QVector3D ray_dir, glm::vec2 mouse_start, glm::vec2 mouse_pos, glm::vec3 front, glm::mat4 vp);
    void startTransform();
    void endTransform(bool moved);
    bool isTransforming() const;
    void toTranslate();
    void toRotate();
    void toScale();
    void resizeHandler(int dir);
    void changeTransOpt();
    void changeCoordSys();
    TransOpt getTransOpt();
    void setTransOpt(TransOpt opt);
    bool isTransformMode() const;
    void clear();

private:
    void addObject(const std::string& name);
    zenovis::Scene* scene() const;
    zenovis::Session* session() const;

    // 计算translate并调用doTransform
    void translate(glm::vec3 start, glm::vec3 end, glm::vec3 axis, glm::mat3 to_local, glm::mat3 to_world, glm::mat3 org_to_local);
    // 计算scale并调用doTransform
    void scale(float scale_size, vec3i axis);
    // 计算rotate并调用doTransform
    void rotate(glm::vec3 start_vec, glm::vec3 end_vec, glm::vec3 axis);

    void createNewTransformNode(NodeLocation& node_location, const std::string& _obj_name);
    void createNewTransformNodeNameWhenMissing(std::string const&node_name);
    void syncToTransformNode(NodeLocation& node_location,
                             const std::string& obj_name);

    // 把FakeTransform上的SRT应用到primitive上
    void doTransform();

    static glm::vec3 QVec3ToGLMVec3(QVector3D QVec3) {
        return {QVec3.x(), QVec3.y(), QVec3.z()};
    }
    void markObjectInteractive(const std::string& obj_name);
    void unmarkObjectInteractive(const std::string& obj_name);
    void markObjectsInteractive();
    void unmarkObjectsInteractive();

    static void print_mat4(std::string name, glm::mat4 mat) {
        qDebug() << name.c_str() << ":";
        for (int i=0; i<4; i++) {
            qDebug() << mat[i][0] << " " << mat[i][1] << " " << mat[i][2] << " " << mat[i][3];
        }
    }
    static void print_mat3(std::string name, glm::mat3 mat) {
        qDebug() << name.c_str() << ":";
        for (int i=0; i<3; i++) {
            qDebug() << mat[i][0] << " " << mat[i][1] << " " << mat[i][2];
        }
    }

    static std::optional<glm::vec3> hitOnPlane(glm::vec3 ori, glm::vec3 dir, glm::vec3 n, glm::vec3 p) {
        auto t = glm::dot((p - ori), n) / glm::dot(dir, n);
        if (t > 0)
            return ori + dir * t;
        else
            return {};
    }

private:
    std::unordered_map<std::string, PrimitiveObject*> m_objects;

    glm::vec3 m_init_pivot = {};
    glm::vec3 m_init_localXOrg = {1, 0, 0};
    glm::vec3 m_init_localYOrg = {0, 1, 0};
    glm::vec3 m_transaction_start_translation;
    glm::vec3 m_transaction_start_scaling;
    glm::quat m_transaction_start_rotation;

    glm::vec3 _transaction_trans = {};
    glm::quat _transaction_rotate = {1, 0, 0, 0};
    glm::vec3 _transaction_scale = {1, 1, 1};

    glm::vec3 m_move_start;

    bool m_isTransforming = false;
    TransOpt m_operation = TransOpt::NONE;
    zenovis::OPERATION_MODE m_operation_mode;
    zenovis::COORD_SYS m_coord_sys = zenovis::COORD_SYS::LOCAL_COORD_SYS;
    float m_handler_scale = 1;
    std::shared_ptr<zenovis::IGraphicHandler> m_handler;
    ViewportWidget* m_viewport;
    glm::mat4 last_transform_matrix = glm::mat4(1);

    glm::mat4 get_pivot_to_world() {
        auto lX = m_init_localXOrg;
        auto lY = m_init_localYOrg;
        auto lZ = glm::cross(lX, lY);
        auto pivot_to_world = glm::mat4(1);
        pivot_to_world[0] = {lX[0], lX[1], lX[2], 0};
        pivot_to_world[1] = {lY[0], lY[1], lY[2], 0};
        pivot_to_world[2] = {lZ[0], lZ[1], lZ[2], 0};
        pivot_to_world[3] = {m_init_pivot[0], m_init_pivot[1], m_init_pivot[2], 1};
        return pivot_to_world;
    }
    glm::mat4 get_pivot_to_local() {
        return glm::inverse(get_pivot_to_world());
    }

    glm::mat4 get_cur_local_transform() {
        auto S = glm::scale(_transaction_scale * m_transaction_start_scaling);
        auto R = glm::toMat4(_transaction_rotate) * glm::toMat4(m_transaction_start_rotation);
        auto T = glm::translate(_transaction_trans + m_transaction_start_translation);
        return T * R * S;
    }

    glm::vec3 get_cur_self_center() {
        auto transform = get_pivot_to_world() * get_cur_local_transform() * get_pivot_to_local();
        auto pos = transform * glm::vec4(m_init_pivot, 1);
        return {pos.x, pos.y, pos.z};
    }

    glm::vec3 get_cur_self_X() {
        if (m_coord_sys == zenovis::COORD_SYS::WORLD_COORD_SYS) {
            return {1, 0, 0};
        }
        auto transform = get_pivot_to_world() * get_cur_local_transform() * get_pivot_to_local();
        auto dir = transform * glm::vec4(m_init_localXOrg, 0);
        return glm::normalize(glm::vec3(dir.x, dir.y, dir.z));
    }
    glm::vec3 get_cur_self_Y() {
        if (m_coord_sys == zenovis::COORD_SYS::WORLD_COORD_SYS) {
            return {0, 1, 0};
        }
        auto transform = get_pivot_to_world() * get_cur_local_transform() * get_pivot_to_local();
        auto dir = transform * glm::vec4(m_init_localYOrg, 0);
        return glm::normalize(glm::vec3(dir.x, dir.y, dir.z));
    }

    glm::mat4 get_transaction_start_transform() {
        auto S = glm::scale(m_transaction_start_scaling);
        auto R = glm::toMat4(m_transaction_start_rotation);
        auto T = glm::translate(m_transaction_start_translation);
        return T * R * S;
    }
    glm::vec3 get_transaction_start_center() {
        auto transform = get_pivot_to_world() * get_transaction_start_transform() * get_pivot_to_local();
        auto pos = transform * glm::vec4(m_init_pivot, 1);
        return {pos.x, pos.y, pos.z};
    }

    glm::vec3 get_transaction_start_X() {
        if (m_coord_sys == zenovis::COORD_SYS::WORLD_COORD_SYS) {
            return {1, 0, 0};
        }
        auto transform = get_pivot_to_world() * get_transaction_start_transform() * get_pivot_to_local();
        auto dir = transform * glm::vec4(m_init_localXOrg, 0);
        return glm::normalize(glm::vec3(dir.x, dir.y, dir.z));
    }
    glm::vec3 get_transaction_start_Y() {
        if (m_coord_sys == zenovis::COORD_SYS::WORLD_COORD_SYS) {
            return {0, 1, 0};
        }
        auto transform = get_pivot_to_world() * get_transaction_start_transform() * get_pivot_to_local();
        auto dir = transform * glm::vec4(m_init_localYOrg, 0);
        return glm::normalize(glm::vec3(dir.x, dir.y, dir.z));
    }
};

}

#endif //__VIEWPORT_TRANSFORM_H__
