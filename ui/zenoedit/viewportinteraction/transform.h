#ifndef __VIEWPORT_TRANSFORM_H__
#define __VIEWPORT_TRANSFORM_H__
#include "zenoapplication.h"
#include <viewport/zenovis.h>
#include <zenomodel/include/graphsmanagment.h>
#include <zeno/types/PrimitiveObject.h>
#include "nodesync.h"

#include <QtWidgets>

#include <glm/glm.hpp>

#include <unordered_map>

class ViewportWidget;

namespace zeno {

enum {
    NONE,
    TRANSLATE,
    ROTATE,
    SCALE
};

class FakeTransformer {
public:
    FakeTransformer(ViewportWidget* viewport);
    void addObject(const std::string& name);
    void addObject(const std::unordered_set<std::string>& names);
    void removeObject(const std::string& name);
    void removeObject(const std::unordered_set<std::string>& names);
    bool calcTransformStart(glm::vec3 ori, glm::vec3 dir, glm::vec3 front);
    bool clickedAnyHandler(QVector3D ori, QVector3D dir, glm::vec3 front);
    void transform(QVector3D camera_pos, glm::vec2 mouse_pos, QVector3D ray_dir, glm::vec3 front, glm::mat4 vp);
    void startTransform();
    void endTransform(bool moved);
    bool isTransforming() const;
    void toTranslate();
    void toRotate();
    void toScale();
    void resizeHandler(int dir);
    void changeTransOpt();
    void changeCoordSys();
    int getTransOpt();
    void setTransOpt(int opt);
    bool isTransformMode() const;
    glm::vec3 getCenter() const;
    void clear();

private:
    zenovis::Scene* scene() const;
    zenovis::Session* session() const;

    void translate(glm::vec3 start, glm::vec3 end, glm::vec3 axis);
    void scale(float scale_size, vec3i axis);
    void rotate(glm::vec3 start_vec, glm::vec3 end_vec, glm::vec3 axis);

    void createNewTransformNode(NodeLocation& node_location,
                                const std::string& obj_name);
    void syncToTransformNode(NodeLocation& node_location,
                             const std::string& obj_name);

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

    template <class T>
    static void print_vec3(std::string name, T& vec) {
        qDebug() << name.c_str() << ": " << vec[0] << " " << vec[1] << " " << vec[2];
    }

    template <class T>
    static void print_vec4(std::string name, T& vec) {
        qDebug() << name.c_str() << ": " << vec[0] << " " << vec[1] << " " << vec[2] << " " << vec[3];
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

    glm::vec3 m_objects_center;

    glm::vec3 m_trans;
    glm::vec4 m_rotate;
    glm::vec3 m_scale;

    glm::vec3 m_last_trans;
    glm::vec4 m_last_rotate;
    glm::vec3 m_last_scale;

    glm::vec3 m_trans_start;
    glm::vec3 m_rotate_start;
    // glm::vec3 m_scale_start;

    bool m_status;
    int m_operation;
    int m_operation_mode;
    int m_coord_sys;
    float m_handler_scale;
    std::shared_ptr<zenovis::IGraphicHandler> m_handler;
    ViewportWidget* m_viewport;
};

}

#endif //__VIEWPORT_TRANSFORM_H__
