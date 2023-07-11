#include <zenovis/Camera.h>
#include <zenovis/opengl/shader.h>
#include "zeno/utils/logger.h"

namespace zenovis {

void Camera::setCamera(zeno::CameraData const &cam) {
    this->placeCamera(
            glm::vec3(cam.pos[0], cam.pos[1], cam.pos[2]),
            glm::vec3(cam.view[0], cam.view[1], cam.view[2]),
            glm::vec3(cam.up[0], cam.up[1], cam.up[2]),
            cam.fov, cam.fnear, cam.ffar);
    //this->m_dof = cam.dof;
    this->m_aperture = cam.aperture;
    this->focalPlaneDistance = cam.focalPlaneDistance;

//    zeno::log_info("radius {}", m_zxx.radius);

    if (cam.isSet) {
        this->m_zxx_in.cx = cam.center[0];
        this->m_zxx_in.cy = cam.center[1];
        this->m_zxx_in.cz = cam.center[2];

        this->m_zxx_in.theta = cam.theta;
        this->m_zxx_in.phi = cam.phi;
        this->m_zxx_in.radius = cam.radius;
    }
    else {
        float radius = m_zxx.radius;
        auto view = zeno::normalize(cam.view);
        zeno::vec3f center = cam.pos + radius * zeno::normalize(cam.view);
        float theta = M_PI_2 - glm::acos(zeno::dot(view, zeno::vec3f(0, 1, 0)));
        float phi = M_PI_2 + std::atan2(view[2], view[0]);
//        zeno::log_info("theta: {}", theta);
//        zeno::log_info("phi: {}", phi);

        this->m_zxx_in.cx = center[0];
        this->m_zxx_in.cy = center[1];
        this->m_zxx_in.cz = center[2];

        this->m_zxx_in.theta = theta;
        this->m_zxx_in.phi = phi;
        this->m_zxx_in.radius = radius;
    }

    this->m_auto_radius = !cam.isSet;
    this->m_need_sync = true;
}

void Camera::placeCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, float fov, float fnear, float ffar) {
    front = glm::normalize(front);
    up = glm::normalize(up);

    if (fov <= 0) {
        auto radius = glm::length(pos);
        m_view = glm::lookAt(pos, pos + front, up);
        m_proj = glm::ortho(-radius * getAspect(), radius * getAspect(), -radius,
                radius, fnear, ffar);
    } else {
        //ZENO_P(pos);
        //ZENO_P(front);
        //ZENO_P(up);
        m_view = glm::lookAt(pos, pos + front, up);
        m_proj = glm::perspectiveZO(glm::radians(fov), getAspect(), ffar, fnear);
        //ZENO_P(m_view);
        //ZENO_P(m_proj);
    }
    m_lodcenter = pos;
    m_lodfront = front;
    m_lodup = up;
    m_near = fnear;
    m_far = ffar;
    m_fov = fov;
}

void Camera::setResolution(int nx, int ny) {
    m_nx = nx;
    m_ny = ny;
    m_proj = glm::perspectiveZO(glm::radians(m_fov), getAspect(), m_far, m_near);
}

void Camera::set_safe_frames(bool bLock, int nx, int ny) {
    if (bLock) {
        safe_frame_ratio = float(nx) / float(ny);
    }
    else {
        safe_frame_ratio = std::nullopt;
    }
}

float Camera::get_safe_frames() const {
    return safe_frame_ratio.value();
}

bool Camera::is_locked_window() const {
    return safe_frame_ratio.has_value();
}

void Camera::focusCamera(float cx, float cy, float cz, float radius) {
    auto center = glm::vec3(cx, cy, cz);
    placeCamera(center - m_lodfront * radius, m_lodfront, m_lodup, m_fov, m_near, m_far);
}
void Camera::lookCamera(float cx, float cy, float cz, float theta, float phi, float radius, float fov, float aperture, float focalPlaneDistance) {
    m_zxx.cx = cx;
    m_zxx.cy = cy;
    m_zxx.cz = cz;
    m_zxx.theta = theta;
    m_zxx.phi = phi;
    m_zxx.radius = radius;
    m_zxx.fov = fov;
    m_zxx.ortho_mode = fov <= 0.0f;
    m_zxx.aperture = aperture;
    m_zxx.focalPlaneDistance = focalPlaneDistance;

    auto center = glm::vec3(cx, cy, cz);

    float cos_t = glm::cos(theta), sin_t = glm::sin(theta);
    float cos_p = glm::cos(phi), sin_p = glm::sin(phi);
    glm::vec3 front(cos_t * sin_p, sin_t, -cos_t * cos_p);
    glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

    if (!(fov <= 0)) {
        auto fnear = 0.05f;
        auto ffar = 20000.0f * std::max(1.0f, (float)radius / 10000.f);
        placeCamera(center - front * radius, front, up, fov, fnear, ffar);
    } else {
        placeCamera(center - front * radius * 0.4f, front, up, 0.f, -100.f, 100.f);
    }
    m_aperture = aperture;
    this->focalPlaneDistance = focalPlaneDistance;
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

zeno::vec4i Camera::viewport() const {
    zeno::log_info("m_nx, m_ny {} {}", m_nx, m_ny);
    zeno::vec4i out;
    if (safe_frame_ratio.has_value()) {
        zeno::log_info("safe_frame_ratio {}", safe_frame_ratio.value());
        float deviceRatio = getDeviceAspect();
        if (deviceRatio > safe_frame_ratio.value()) {
            int w = m_ny * safe_frame_ratio.value();
            int h = m_ny;
            int x = (m_nx - w) / 2;
            int y = 0;
            out = {x, y, w, h};
        }
        else {
            int w = m_nx;
            int h = m_nx / safe_frame_ratio.value();
            int x = 0;
            int y = (m_ny - h) / 2;
            out = {x, y, w, h};
        }
    }
    else {
        out = {0, 0, m_nx, m_ny};
    }
    zeno::log_info("viewport {}", out);
    return out;
}

}
