#include <zenovis/Camera.h>

namespace zenovis {

void Camera::setCamera(zeno::CameraData const &cam) {
    this->setCamera(
            glm::vec3(cam.pos[0], cam.pos[1], cam.pos[2]),
            glm::vec3(cam.view[0], cam.view[1], cam.view[2]),
            glm::vec3(cam.up[0], cam.up[1], cam.up[2]),
            cam.fov, cam.fnear, cam.ffar, 2.0f);
    /* this->m_nx = cam.nx; */
    /* this->m_ny = cam.ny; */
    this->m_dof = cam.dof;
    this->m_aperature = cam.aperature;
}

void Camera::setCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, float fov, float fnear, float ffar, float radius) {
    front = glm::normalize(front);
    up = glm::normalize(up);
    if (fov <= 0) {
        m_view = glm::lookAt(pos, pos + front, up);
        m_proj = glm::ortho(-radius * getAspect(), radius * getAspect(), -radius,
                radius, fnear, ffar);
    } else {
        //ZENO_P(pos);
        //ZENO_P(front);
        //ZENO_P(up);
        m_view = glm::lookAt(pos, pos + front, up);
        m_proj = glm::perspective(glm::radians(fov), getAspect(), fnear, ffar);
        //ZENO_P(m_view);
        //ZENO_P(m_proj);
    }
    m_lodradius = radius;
    m_lodcenter = pos;
    m_lodfront = front;
    m_lodup = up;
    m_near = fnear;
    m_far = ffar;
    m_fov = fov;
}

void Camera::setCamera(float cx, float cy, float cz, float theta, float phi, float radius, float fov) {
    auto center = glm::vec3(cx, cy, cz);

    float cos_t = glm::cos(theta), sin_t = glm::sin(theta);
    float cos_p = glm::cos(phi), sin_p = glm::sin(phi);
    glm::vec3 front(cos_t * sin_p, sin_t, -cos_t * cos_p);
    glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

    if (!(fov <= 0)) {
        auto fnear = 0.1f;
        auto ffar = 20000.0f * std::max(1.0f, (float)radius / 10000.f);
        setCamera(center - front * radius, front, up, fov, fnear, ffar, radius);
    } else {
        setCamera(center - front * radius, front, up, 0.f, -100.f, 100.f, radius);
    }
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
