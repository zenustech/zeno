#pragma once


#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>


namespace z2::GL {


struct Camera {
    int nx = 512, ny = 512;
    float point_scale = 1.f;
    glm::mat4x4 view;
    glm::mat4x4 proj;

    void look_at(double cx, double cy, double cz,
                 double theta, double phi, double radius,
                 double fov, bool ortho_mode);
};


}
