#pragma once


#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>


namespace zeno2::GL {


struct Program;


class Camera {
    int nx = 512, ny = 512;
    float point_scale = 1.f;
    glm::mat4x4 view;
    glm::mat4x4 proj;

    glm::dvec3 center;
    double theta = 0.f;
    double phi = 0.f;
    double radius = 4.f;
    double fov = 30.f;
    bool ortho_mode = false;

    void update();

public:
    Camera() {
        update();
    }

    void resize(int nx, int ny);
    void move(double dx, double dy, bool pan_mode);
    void zoom(double dy, bool fov_mode);
    void uniform(Program *prog);
};


}
