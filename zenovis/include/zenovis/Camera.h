#pragma once

#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <zeno/types/CameraObject.h>

namespace zenovis {

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
    int m_nx{512}, m_ny{512};
    glm::mat4x4 m_view{1}, m_proj{1};

    float m_near = 0.01f;
    float m_far = 20000.0f;
    float m_fov = 45.f;

    float m_aperture = 0.0f;
    float focalPlaneDistance = 2.0f;
    float m_dof = -1.f;
    float m_safe_frames = 0;

    glm::vec3 m_lodcenter{0, 0, -1};
    glm::vec3 m_lodfront{0, 0, 1};
    glm::vec3 m_lodup{0, 1, 0};

    bool m_need_sync = false;
    bool m_block_window = false;
    bool m_auto_radius = false;

    float m_theta = 0;
    float m_phi = 0;
    float m_roll = 0;
    zeno::vec3f m_center = {};
    bool m_ortho_mode = false;
    float m_radius = 5;

    zeno::vec2i viewport_offset = {};
    ZOptixCameraSettingInfo zOptixCameraSettingInfo = {};

    // only used in real-shader
    struct ZxxHappyLookParam {
        float cx = 0;
        float cy = 0;
        float cz = 0;
        float theta = 0;
        float phi = 0;
        float radius = 0;
        float fov = 0;
        bool ortho_mode = false;
        float aperture = 0;
        float focalPlaneDistance = 0;
    };
    struct ZxxHappyLookParam m_zxx;

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
    void placeCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up);
    void lookCamera(float cx, float cy, float cz, float theta, float phi, float radius, bool ortho_mode, float fov, float aperture, float focalPlaneDistance);
    void focusCamera(float cx, float cy, float cz, float radius);
    void set_program_uniforms(opengl::Program *pro);
    void updateMatrix();
};

} // namespace zenovis
