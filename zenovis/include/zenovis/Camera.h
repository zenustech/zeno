#pragma once

#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <zeno/types/CameraObject.h>

namespace zenovis {
enum class CameraLookToDir {
    front_view,
    right_view,
    top_view,
    back_view,
    left_view,
    bottom_view,
    back_to_origin,
};

namespace opengl {
    class Program;
}
struct ZOptixCameraSettingInfo {
    float aperture = 2.0;
    float shutter_speed = 1.0/25;
    float iso = 150;
    bool aces = false;
    bool exposure = false;
};

struct Camera {
    float inf_z_near = 0.001f;
    int m_nx{512}, m_ny{512};
    glm::mat4x4 m_view{1}, m_proj{1};

    float m_near = 0.01f;
    float m_far = 20000.0f;
    float m_fov = 45.f;

    float m_aperture = 0.0f;
    float focalPlaneDistance = 2.0f;
    float m_dof = -1.f;
    float m_safe_frames = 0;

    glm::vec3 m_pos{0, 0, 5};
    glm::vec3 m_pivot = {};
    glm::quat m_rotation = {1, 0, 0, 0};

    bool m_block_window = false;
public:
    glm::vec3 get_lodfront() {
        return m_rotation * glm::vec3(0, 0, -1);
    }
    glm::vec3 get_lodup() {
        return m_rotation * glm::vec3(0, 1, 0);
    }
    bool m_ortho_mode = false;
    float get_radius() {
        return glm::distance(m_pos, m_pivot);
    }
    glm::vec3 getPos() {
        return m_pos;
    }
    void setPos(glm::vec3 value) {
        m_pos = value;
    }
    glm::vec3 getPivot() {
        return m_pivot;
    }
    void setPivot(glm::vec3 value) {
        m_pivot = value;
    }

    zeno::vec2i viewport_offset = {};
    ZOptixCameraSettingInfo zOptixCameraSettingInfo = {};

    float getAspect() const {
        return (float)m_nx / (float)m_ny;
    }

    void setResolution(int nx, int ny);
    void setResolutionInfo(bool block, int nx, int ny);
    void set_safe_frames(bool bLock, int nx, int ny);
    float get_safe_frames() const;
    bool is_locked_window() const;
    void setCamera(zeno::CameraData const &cam);
    void setPhysicalCamera(float aperture, float shutter_speed, float iso, bool aces, bool exposure);
    void placeCamera(glm::vec3 pos, glm::vec3 view, glm::vec3 up);
    void placeCamera(glm::vec3 pos, glm::quat rotation);
    void focusCamera(float cx, float cy, float cz, float radius);
    void set_program_uniforms(opengl::Program *pro);
    void updateMatrix();
};

} // namespace zenovis
