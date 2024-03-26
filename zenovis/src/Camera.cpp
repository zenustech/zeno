#include <zenovis/Camera.h>
#include <zenovis/opengl/shader.h>
#include "zeno/utils/logger.h"

namespace zenovis {

void Camera::setCamera(zeno::CameraData const &cam) {
    zeno::log_info("set camera");
    m_far = cam.ffar;
    m_near = cam.fnear;
    m_ortho_mode = cam.fov <= 0;
    m_fov = cam.fov;
    this->placeCamera(
            glm::vec3(cam.pos[0], cam.pos[1], cam.pos[2]),
            glm::vec3(cam.view[0], cam.view[1], cam.view[2]),
            glm::vec3(cam.up[0], cam.up[1], cam.up[2]));
    //this->m_dof = cam.dof;
    this->m_aperture = cam.aperture;
    this->focalPlaneDistance = cam.focalPlaneDistance;

//    zeno::log_info("radius {}", m_zxx.radius);

    if (cam.isSet) {
        m_center = cam.center;
        m_theta = cam.theta;
        m_phi = cam.phi;
        m_radius = cam.radius;
    }
    else {
        auto view = zeno::normalize(cam.view);
        zeno::vec3f center = cam.pos + m_radius * zeno::normalize(cam.view);
        float theta = M_PI_2 - glm::acos(zeno::dot(view, zeno::vec3f(0, 1, 0)));
        float phi = M_PI_2 + std::atan2(view[2], view[0]);
//        zeno::log_info("theta: {}", theta);
//        zeno::log_info("phi: {}", phi);

        m_center = center;
        m_theta = theta;
        m_phi = phi;

        float cos_t = glm::cos(m_theta), sin_t = glm::sin(m_theta);
        float cos_p = glm::cos(m_phi), sin_p = glm::sin(m_phi);
        glm::vec3 front(cos_t * sin_p, sin_t, -cos_t * cos_p);
        glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);
        glm::vec3 left = glm::cross(up, front);
        float map_to_up = glm::dot(up, zeno::vec_to_other<glm::vec3>(cam.up));
        float map_to_left = glm::dot(left, zeno::vec_to_other<glm::vec3>(cam.up));
        m_roll = glm::atan(map_to_left, map_to_up);
    }

    this->m_auto_radius = !cam.isSet;
    this->m_need_sync = true;
}

void Camera::setPhysicalCamera(float aperture, float shutter_speed, float iso, bool aces, bool exposure) {
    this->zOptixCameraSettingInfo.aperture = aperture;
    this->zOptixCameraSettingInfo.shutter_speed = shutter_speed;
    this->zOptixCameraSettingInfo.iso = iso;
    this->zOptixCameraSettingInfo.aces = aces;
    this->zOptixCameraSettingInfo.exposure = exposure;
}

void Camera::placeCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up) {
    front = glm::normalize(front);
    up = glm::normalize(up);

    m_lodcenter = pos;
    m_lodfront = front;
    m_lodup = up;

    m_view = glm::lookAt(m_lodcenter, m_lodcenter + m_lodfront, m_lodup);
    if (m_ortho_mode) {
        auto radius = m_radius;
        m_proj = glm::orthoZO(-radius * getAspect(), radius * getAspect(), -radius,
                radius, m_far, m_near);
    } else {
        m_proj = glm::perspectiveZO(glm::radians(m_fov), getAspect(), m_far, m_near);
    }
}

void Camera::updateMatrix() {
    auto center = zeno::vec_to_other<glm::vec3>(m_center) ;
    float cos_t = glm::cos(m_theta), sin_t = glm::sin(m_theta);
    float cos_p = glm::cos(m_phi), sin_p = glm::sin(m_phi);
    glm::vec3 front(cos_t * sin_p, sin_t, -cos_t * cos_p);
    glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);
    glm::vec3 left = glm::cross(up, front);
    up = glm::cos(m_roll) * up + glm::sin(m_roll) * left;

    if (!m_ortho_mode) {
        m_near = 0.05f;
        m_far = 20000.0f * std::max(1.0f, (float)m_radius / 10000.f);
        placeCamera(center - front * m_radius, front, up);
    } else {
        placeCamera(center - front * m_radius * 0.4f, front, up);
    }
}

void Camera::setResolution(int nx, int ny) {
    m_nx = nx;
    m_ny = ny;
    m_proj = glm::perspectiveZO(glm::radians(m_fov), getAspect(), m_far, m_near);
}
void Camera::setResolutionInfo(bool block, int nx, int ny)
{
    set_safe_frames(block, nx, ny);
}

void Camera::set_safe_frames(bool bLock, int nx, int ny) {
    m_block_window = bLock;
    m_safe_frames = (float)nx / ny;
}

float Camera::get_safe_frames() const {
    return m_safe_frames;
}

bool Camera::is_locked_window() const {
    return m_block_window;
}

void Camera::focusCamera(float cx, float cy, float cz, float radius) {
    auto center = glm::vec3(cx, cy, cz);
    placeCamera(center - m_lodfront * radius, m_lodfront, m_lodup);
}
void Camera::lookCamera(float cx, float cy, float cz, float theta, float phi, float radius, bool ortho_mode, float fov, float aperture, float focalPlaneDistance) {
    m_zxx.cx = cx;
    m_zxx.cy = cy;
    m_zxx.cz = cz;
    m_zxx.theta = theta;
    m_zxx.phi = phi;
    m_zxx.radius = radius;
    m_zxx.fov = fov;
    m_zxx.ortho_mode = ortho_mode;
    m_zxx.aperture = aperture;
    m_zxx.focalPlaneDistance = focalPlaneDistance;

    m_ortho_mode = ortho_mode;
    m_aperture = aperture;
    this->focalPlaneDistance = focalPlaneDistance;

    updateMatrix();
}

void Camera::set_program_uniforms(opengl::Program *pro) {
    pro->use();

    auto vp = m_proj * m_view;
    pro->set_uniform("mVP", vp);
    pro->set_uniform("mInvVP", glm::inverse(vp));
    pro->set_uniform("mView", m_view);
    pro->set_uniform("mProj", m_proj);
    pro->set_uniform("mInvView", glm::inverse(m_view));
    pro->set_uniform("mInvProj", glm::inverse(m_proj));

    //pro->set_uniform("mCameraRadius", m_camRadius);
    //pro->set_uniform("mCameraCenter", m_camCenter);
    //pro->set_uniform("mGridScale", m_grid_scale);
    //pro->set_uniform("mGridBlend", m_grid_blend);
    //pro->set_uniform("mSampleWeight", m_sample_weight);
}

}
