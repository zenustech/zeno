#pragma once

#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>

namespace zenovis {

struct Camera {
    glm::vec3 bgcolor{0.23f, 0.23f, 0.23f};
    std::unique_ptr<opengl::Buffer> vao;
    int nx{512}, ny{512};
    int oldnx{512}, oldny{512};

    double g_aspect{1};
    double last_xpos{}, last_ypos{};
    glm::vec3 center;

    glm::mat4x4 view{1}, proj{1};
    glm::mat4x4 gizmo_view{1}, gizmo_proj{1};
    float point_scale = 1.f;
    float camera_radius = 1.f;
    float grid_scale = 1.f;
    float grid_blend = 0.f;
    float g_dof = -1.f;
    float g_aperature = 0.05f;
    float m_sample_weight = 0.0f;

    void setAspect(float _aspect) {
        g_aspect = _aspect;
    }

    void setDOF(float _dof) {
        g_dof = _dof;
    }

    void setAperature(float _apt) {
        g_aperature = _apt;
    }

    void set_perspective(std::array<double, 16> viewArr,
                         std::array<double, 16> projArr) {
        std::memcpy(glm::value_ptr(view), viewArr.data(), viewArr.size());
        std::memcpy(glm::value_ptr(proj), projArr.data(), projArr.size());
    }

    float g_near, g_far, g_fov;
    glm::mat4 g_view, g_proj;
    glm::vec3 g_camPos, g_camView, g_camUp;
    int g_camSetFromNode = 0;
    glm::mat4 cview, cproj;

    void clearCameraControl() {
        g_camSetFromNode = 0;
        proj = glm::perspective(glm::radians(45.0), g_aspect, 0.1, 20000.0);
        g_dof = -1;
        g_fov = 45.0;
        g_near = 0.1;
        g_far = 20000;
        g_proj = proj;
    }

    bool smooth_shading = false;
    bool normal_check = false;

    void set_smooth_shading(bool smooth) {
        smooth_shading = smooth;
    }

    void set_normal_check(bool check) {
        normal_check = check;
    }

    void setCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, double _fov,
                   double fnear, double ffar, double _dof, int set) {
        front = glm::normalize(front);
        up = glm::normalize(up);
        cview = glm::lookAt(pos, pos + front, up);
        cproj = glm::perspective(glm::radians(_fov), g_aspect, fnear, ffar);
        g_fov = _fov;
        g_near = fnear;
        g_far = ffar;
        g_view = cview;
        g_proj = cproj;
        g_camPos = pos;
        g_camView = glm::normalize(front);
        g_camUp = glm::normalize(up);
        g_dof = _dof;
        g_camSetFromNode = set;
    }

    void look_perspective(double cx, double cy, double cz, double theta,
                          double phi, double radius, double fov,
                          bool ortho_mode) {
        if (g_camSetFromNode == 1) {
            view = cview;
            proj = cproj;
            /* auto &scene = Scene::getInstance(); */
            /* auto &lights = scene.lights; */
            /* for (auto &light : lights) { */
            /*     light->gfov = fov; */
            /*     light->gaspect = g_aspect; */
            /* } */
            return;
        }
        /* auto &scene = Scene::getInstance(); */
        /* auto &lights = scene.lights; */
        /* for (auto &light : lights) { */
        /*     light->gfov = fov; */
        /*     light->gaspect = g_aspect; */
        /* } */
        center = glm::vec3(cx, cy, cz);

        point_scale = 21.6f / tan(fov * 0.5f * 3.1415926f / 180.0f);

        double cos_t = glm::cos(theta), sin_t = glm::sin(theta);
        double cos_p = glm::cos(phi), sin_p = glm::sin(phi);
        glm::vec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

        if (ortho_mode) {
            view = glm::lookAt(center - back, center, up);
            proj = glm::ortho(-radius * g_aspect, radius * g_aspect, -radius,
                              radius, -100.0, 100.0);
            g_view = view;
            g_proj = proj;
        } else {
            view = glm::lookAt(center - back * (float)radius, center, up);
            proj = glm::perspective(
                glm::radians(fov), g_aspect, 0.1,
                20000.0 * std::max(1.0f, (float)radius / 10000.f));
            g_fov = fov;
            g_near = 0.1;
            g_far = 20000.0 * std::max(1.0f, (float)radius / 10000.f);
            g_view = view;
            g_proj = proj;
            g_camPos = center - back * (float)radius;
            g_camView = back * (float)radius;
            g_camUp = up;
        }
        camera_radius = radius;
        float level = std::fmax(std::log(radius) / std::log(5) - 1.0, -1);
        grid_scale = std::pow(5, std::floor(level));
        auto ratio_clamp = [](float value, float lower_bound,
                              float upper_bound) {
            float ratio = (value - lower_bound) / (upper_bound - lower_bound);
            return std::fmin(std::fmax(ratio, 0.0), 1.0);
        };
        grid_blend = ratio_clamp(level - std::floor(level), 0.8, 1.0);
        center = glm::vec3(0, 0, 0);
        radius = 5.0;
        gizmo_view = glm::lookAt(center - back, center, up);
        gizmo_proj = glm::ortho(-radius * g_aspect, radius * g_aspect, -radius,
                                radius, -100.0, 100.0);
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
        pro->set_uniform("mPointScale", point_scale);
        pro->set_uniform("mSmoothShading", smooth_shading);
        pro->set_uniform("mNormalCheck", normal_check);
        pro->set_uniform("mCameraRadius", camera_radius);
        pro->set_uniform("mCameraCenter", center);
        pro->set_uniform("mGridScale", grid_scale);
        pro->set_uniform("mGridBlend", grid_blend);
        pro->set_uniform("mSampleWeight", m_sample_weight);
    }

    bool show_grid = true;
    bool render_wireframe = false;

    int _m_default_num_samples = 8;
    int _m_ready_num_samples = -1;
    void setNumSamples(int num_samples) {
        _m_default_num_samples = num_samples;
        _m_ready_num_samples = -1;
    }
    int getNumSamples() {
        if (_m_ready_num_samples < 0) {
            /* begin cihou mesa */
            int max_num_samples = _m_ready_num_samples;
            CHECK_GL(glGetIntegerv(GL_MAX_INTEGER_SAMPLES, &max_num_samples));
            _m_ready_num_samples = std::min(_m_ready_num_samples, max_num_samples);
            printf("num samples: %d\n", _m_ready_num_samples);
            /* end cihou mesa */
        }
        return _m_ready_num_samples;
    }
};

} // namespace zenovis
