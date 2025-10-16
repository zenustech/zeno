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
    int  renderRatio = 1;
    bool aces = false;
    bool exposure = false;
    bool panorama_camera = false;
    bool panorama_vr180 = false;
    float pupillary_distance = 0.06;
};

struct Camera {
    float inf_z_near = 0.001f;
    int m_nx{512}, m_ny{512};

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
    void reset() {
        m_pos = {0, 0, 5};
        m_pivot = {};
        m_rotation = {1, 0, 0, 0};
        updateMatrix();
    }
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
    void setPhysicalCamera(float aperture, float shutter_speed, float iso, int scale, bool aces, bool exposure, bool panorama_camera, bool panorama_vr180, float pupillary_distance);
    void placeCamera(glm::vec3 pos, glm::vec3 view, glm::vec3 up);
    void placeCamera(glm::vec3 pos, glm::quat rotation);
    void focusCamera(float cx, float cy, float cz, float radius);
    void set_program_uniforms(opengl::Program *pro);
    void updateMatrix();
    glm::mat4x4 get_view_matrix() {
        return glm::lookAt(m_pos, m_pos + get_lodfront(), get_lodup());
    }
    static glm::mat4 MakeInfReversedZProjRH(float fovY_radians, float aspectWbyH, float zNear) {
        float f = 1.0f / tan(fovY_radians / 2.0f);
        return glm::mat4(
                f / aspectWbyH, 0.0f,  0.0f,  0.0f,
                0.0f,    f,  0.0f,  0.0f,
                0.0f, 0.0f,  0.0f, -1.0f,
                0.0f, 0.0f, zNear,  0.0f);
    }
    glm::mat4x4 get_proj_matrix() {
        if (m_ortho_mode) {
            auto radius = get_radius();
            return glm::orthoZO(-radius * getAspect(), radius * getAspect(), -radius,
                radius, m_far, m_near);
        } else {
            return MakeInfReversedZProjRH(glm::radians(m_fov), getAspect(), inf_z_near);
        }
    }
};

} // namespace zenovis
