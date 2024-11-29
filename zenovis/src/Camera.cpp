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
    this->m_aperture = cam.aperture;
    this->focalPlaneDistance = cam.focalPlaneDistance;

    if (cam.pivot.has_value()) {
        this->m_pivot = zeno::vec_to_other<glm::vec3>(cam.pivot.value());
    }
    else {
        this->m_pivot = zeno::vec_to_other<glm::vec3>(cam.pos);
    }
}

void Camera::setPhysicalCamera(float aperture, float shutter_speed, float iso, bool aces, bool exposure, bool panorama_camera, bool panorama_vr180, float pupillary_distance) {
    this->zOptixCameraSettingInfo.aperture = aperture;
    this->zOptixCameraSettingInfo.shutter_speed = shutter_speed;
    this->zOptixCameraSettingInfo.iso = iso;
    this->zOptixCameraSettingInfo.aces = aces;
    this->zOptixCameraSettingInfo.exposure = exposure;
    this->zOptixCameraSettingInfo.panorama_camera = panorama_camera;
    this->zOptixCameraSettingInfo.panorama_vr180 = panorama_vr180;
    this->zOptixCameraSettingInfo.pupillary_distance = pupillary_distance;
}

void Camera::placeCamera(glm::vec3 pos, glm::vec3 view, glm::vec3 up) {
    auto right = glm::cross(glm::normalize(view), glm::normalize(up));
    glm::mat3 rotation;
    rotation[0] = right;
    rotation[1] = up;
    rotation[2] = -view;

    Camera::placeCamera(pos, glm::quat_cast(rotation));
}

void Camera::placeCamera(glm::vec3 pos, glm::quat rotation) {
    m_pos = pos;
    m_rotation = rotation;
}

void Camera::updateMatrix() {
    auto center = zeno::vec_to_other<glm::vec3>(m_pivot) ;

    if (!m_ortho_mode) {
        m_near = 0.05f;
        m_far = 20000.0f * std::max(1.0f, get_radius() / 10000.f);
        placeCamera(getPos(), m_rotation);
    } else {
        placeCamera(getPos(), m_rotation);
    }
}

void Camera::setResolution(int nx, int ny) {
    m_nx = nx;
    m_ny = ny;
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
    placeCamera(center - get_lodfront() * radius, m_rotation);
}

void Camera::set_program_uniforms(opengl::Program *pro) {
    pro->use();
    auto m_view = get_view_matrix();
    auto m_proj = get_proj_matrix();
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
