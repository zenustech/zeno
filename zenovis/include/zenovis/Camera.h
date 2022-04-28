#pragma once

#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <zenovis/opengl/shader.h>
#include <zeno/utils/zeno_p.h>

namespace zenovis {

struct Camera {
    int m_nx{512}, m_ny{512};
    glm::mat4x4 m_view{1}, m_proj{1};
    float m_near = 0.1f;
    float m_far = 20000.0f;
    float m_fov = 45.f;
    float m_lodradius = 1.0f;
    glm::vec3 m_lodcenter{0};

    float getAspect() const {
        return (float)m_nx / (float)m_ny;
    }

    void setCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, float fov, float fnear, float ffar, float radius) {
        front = glm::normalize(front);
        up = glm::normalize(up);
        if (fov <= 0) {
            m_view = glm::lookAt(pos, pos + front, up);
            m_proj = glm::ortho(-radius * getAspect(), radius * getAspect(), -radius,
                              radius, fnear, ffar);
        } else {
            ZENO_P(pos);
            ZENO_P(front);
            ZENO_P(up);
            m_view = glm::lookAt(pos, pos + front, up);
            m_proj = glm::perspective(glm::radians(fov), getAspect(), fnear, ffar);
            ZENO_P(m_view);
            ZENO_P(m_proj);
        }
        m_lodradius = radius;
        m_lodcenter = pos;
        m_near = fnear;
        m_far = ffar;
        m_fov = fov;
    }

    void setCamera(float cx, float cy, float cz, float theta, float phi, float radius, float fov) {
        auto center = glm::vec3(cx, cy, cz);

        float cos_t = glm::cos(theta), sin_t = glm::sin(theta);
        float cos_p = glm::cos(phi), sin_p = glm::sin(phi);
        glm::vec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

        if (!(fov <= 0)) {
            auto fnear = 0.1f;
            auto ffar = 20000.0f * std::max(1.0f, (float)radius / 10000.f);
            setCamera(center + back * radius, -back, up, fov, fnear, ffar, radius);
        } else {
            setCamera(center + back * radius, -back, up, 0.f, -100.f, 100.f, radius);
        }
    }

    void set_program_uniforms(opengl::Program *pro) {
        pro->use();

        auto vp = m_proj * m_view;
        pro->set_uniform("mVP", vp);
        pro->set_uniform("mInvVP", glm::inverse(vp));
        pro->set_uniform("mView", m_view);
        pro->set_uniform("mProj", m_proj);
        pro->set_uniform("mInvView", glm::inverse(m_view));
        pro->set_uniform("mInvProj", glm::inverse(m_proj));

        auto point_scale = 21.6f / std::tan(m_fov * 0.5f * 3.1415926f / 180.0f);
        pro->set_uniform("mPointScale", point_scale);

        //pro->set_uniform("mCameraRadius", m_camRadius);
        //pro->set_uniform("mCameraCenter", m_camCenter);
        //pro->set_uniform("mGridScale", m_grid_scale);
        //pro->set_uniform("mGridBlend", m_grid_blend);
        //pro->set_uniform("mSampleWeight", m_sample_weight);
    }
};

} // namespace zenovis
