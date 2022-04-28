#pragma once

#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <zenovis/opengl/shader.h>

namespace zenovis::zhxx {

struct ZhxxCamera {
    glm::mat4x4 view{1}, proj{1};

    int m_nx{512}, m_ny{512};

    float m_point_scale = 1.f;
    float m_grid_scale = 1.f;
    float m_grid_blend = 0.f;
    float m_aperature = 0.05f;
    float m_sample_weight = 1.0f;

    float m_dof = -1.f;
    float m_near = 0.1f;
    float m_far = 20000.0f;
    float m_fov = 45.0f;

    float m_camRadius = 1.f;
    glm::vec3 m_camCenter{0, 0, 0};
    glm::vec3 m_camPos{0, 0, 1};
    glm::vec3 m_camView{0, 0, -1};
    glm::vec3 m_camUp{0, 1, 0};

    void setDOF(float _dof) {
        m_dof = _dof;
    }

    void setAperature(float _apt) {
        m_aperature = _apt;
    }

    void setCamera(std::array<float, 16> const &viewArr,
                   std::array<float, 16> const &projArr) {
        std::memcpy(glm::value_ptr(view), viewArr.data(), viewArr.size());
        std::memcpy(glm::value_ptr(proj), projArr.data(), projArr.size());
    }
    /* int g_camSetFromNode = 0; */
    /* glm::mat4 cview, cproj; */

    //void clearCameraControl() {
        //[> g_camSetFromNode = 0; <]
        //proj = glm::perspective(glm::radians(45.0f), getAspect(), 0.1f, 20000.0f);
        //m_dof = -1;
        //m_fov = 45.0;
        //m_near = 0.1;
        //m_far = 20000;
    //}

    float getAspect() const {
        return (float)m_nx / (float)m_ny;
    }

    void setCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, float _fov,
                   float fnear, float ffar, float _dof) {
        front = glm::normalize(front);
        up = glm::normalize(up);
        view = glm::lookAt(pos, pos + front, up);
        proj = glm::perspective(glm::radians(_fov), getAspect(), fnear, ffar);
        m_fov = _fov;
        m_near = fnear;
        m_far = ffar;
        m_camPos = pos;
        m_camView = glm::normalize(front);
        m_camUp = glm::normalize(up);
        m_dof = _dof;
        /* g_camSetFromNode = set; */
    }

    void setCamera(float cx, float cy, float cz, float theta,
                   float phi, float radius, float fov,
                   bool ortho_mode) {
        /* if (g_camSetFromNode == 1) { */
        /*     view = cview; */
        /*     proj = cproj; */
            /* auto &scene = ZhxxScene::getInstance(); */
            /* auto &lights = scene.lights; */
            /* for (auto &light : lights) { */
            /*     light->gfov = fov; */
            /*     light->gaspect = g_aspect; */
            /* } */
        /*     return; */
        /* } */
        /* auto &scene = ZhxxScene::getInstance(); */
        /* auto &lights = scene.lights; */
        /* for (auto &light : lights) { */
        /*     light->gfov = fov; */
        /*     light->gaspect = g_aspect; */
        /* } */
        auto center = glm::vec3(cx, cy, cz);

        m_point_scale = 21.6f / tan(fov * 0.5f * 3.1415926f / 180.0f);

        float cos_t = glm::cos(theta), sin_t = glm::sin(theta);
        float cos_p = glm::cos(phi), sin_p = glm::sin(phi);
        glm::vec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

        if (ortho_mode) {
            view = glm::lookAt(center - back, center, up);
            proj = glm::ortho(-radius * getAspect(), radius * getAspect(), -radius,
                              radius, -100.0f, 100.0f);
        } else {
            view = glm::lookAt(center - back * (float)radius, center, up);
            proj = glm::perspective(
                glm::radians(fov), getAspect(), 0.1f,
                20000.0f * std::max(1.0f, (float)radius / 10000.f));
            m_fov = fov;
            m_near = 0.1f;
            m_far = 20000.0f * std::max(1.0f, (float)radius / 10000.f);
            m_camPos = center - back * (float)radius;
            m_camView = back * (float)radius;
            m_camUp = up;
        }
        float level = std::max(std::log(radius) / std::log(5.0f) - 1.0f, -1.0f);
        m_grid_scale = std::pow(5.f, std::floor(level));
        auto ratio_clamp = [](float value, float lower_bound,
                              float upper_bound) {
            float ratio = (value - lower_bound) / (upper_bound - lower_bound);
            return std::min(std::max(ratio, 0.0f), 1.0f);
        };
        m_grid_blend = ratio_clamp(level - std::floor(level), 0.8f, 1.0f);
        //gizmo_view = glm::lookAt(back, glm::vec3(0), up);
        //gizmo_proj = glm::ortho(-radius * getAspect(), radius * getAspect(), -radius,
                                //radius, -100.0f, 100.0f);
        m_camCenter = center;
        m_camRadius = radius;
    }

    void set_program_uniforms(opengl::Program *pro) {
        pro->use();

        auto pers = proj * view;
        pro->set_uniform("mVP", pers);
        pro->set_uniform("mInvVP", glm::inverse(pers));
        pro->set_uniform("mView", view);
        pro->set_uniform("mProj", proj);
        pro->set_uniform("mInvView", glm::inverse(view));
        pro->set_uniform("mInvProj", glm::inverse(proj));
        pro->set_uniform("mPointScale", m_point_scale);
        pro->set_uniform("mCameraRadius", m_camRadius);
        pro->set_uniform("mCameraCenter", m_camCenter);
        pro->set_uniform("mGridScale", m_grid_scale);
        pro->set_uniform("mGridBlend", m_grid_blend);
        pro->set_uniform("mSampleWeight", m_sample_weight);
    }

    int _m_default_num_samples = 8;
    int _m_ready_num_samples = -1;
    void setNumSamples(int num_samples) {
        _m_default_num_samples = num_samples;
        _m_ready_num_samples = -1;
    }
    int getNumSamples() {
        if (_m_ready_num_samples < 0) {
            /* begin cihou mesa */
            int max_num_samples = _m_default_num_samples;
            CHECK_GL(glGetIntegerv(GL_MAX_INTEGER_SAMPLES, &max_num_samples));
            _m_ready_num_samples = std::min(_m_default_num_samples, max_num_samples);
            /* printf("num samples: %d\n", _m_ready_num_samples); */
            /* end cihou mesa */
        }
        return _m_ready_num_samples;
    }
};

} // namespace zenovis
