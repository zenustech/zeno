#pragma once

#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <zenovis/opengl/shader.h>
#include <zeno/types/CameraObject.h>

namespace zenovis {

struct Camera {
    int m_nx{512}, m_ny{512};
    glm::mat4x4 m_view{1}, m_proj{1};

    float m_near = 0.01f;
    float m_far = 20000.0f;
    float m_fov = 45.f;

    float m_aperture = 0.1f;
    float focalPlaneDistance = 2.0f;
    float m_dof = -1.f;

    glm::vec3 m_lodcenter{0, 0, -1};
    glm::vec3 m_lodfront{0, 0, 1};
    glm::vec3 m_lodup{0, 1, 0};

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
    } m_zxx;

    float getAspect() const {
        return (float)m_nx / (float)m_ny;
    }

    void setResolution(int nx, int ny);
    void setCamera(zeno::CameraData const &cam);
    void placeCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, float fov, float fnear, float ffar);
    void lookCamera(float cx, float cy, float cz, float theta, float phi, float radius, float fov, float aperture, float focalPlaneDistance);
    void focusCamera(float cx, float cy, float cz, float radius);
    void set_program_uniforms(opengl::Program *pro);
};

} // namespace zenovis
