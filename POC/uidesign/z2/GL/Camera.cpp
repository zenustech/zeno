#include <z2/GL/Camera.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>
#include <numbers>


namespace z2::GL {


void Camera::zoom(double dy, bool fov_mode) {
    auto scale = std::pow(0.89, dy);
    if (fov_mode)
        fov /= scale;
    radius *= scale;
    update();
}


void Camera::move(double dx, double dy, bool pan_mode) {
    if (pan_mode) {
        auto cos_t = std::cos(theta);
        auto sin_t = std::sin(theta);
        auto cos_p = std::cos(phi);
        auto sin_p = std::sin(phi);
        glm::dvec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        glm::dvec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);
        glm::dvec3 right = normalize(cross(up, back));
        up = normalize(cross(right, back));
        auto delta = right * dx + up * dy;
        center += delta * radius;
    } else {
        theta += dy * std::numbers::pi;
        phi += dx * std::numbers::pi;
    }
    update();
}


void Camera::update() {
    point_scale = ny / (50.f * tanf(fov*0.5f*3.1415926f/180.0f));

    double cos_t = glm::cos(theta), sin_t = glm::sin(theta);
    double cos_p = glm::cos(phi), sin_p = glm::sin(phi);
    glm::dvec3 back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    glm::dvec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);

    if (ortho_mode) {
        view = glm::lookAt(center - back, center, up);
        proj = glm::ortho(-radius * nx / ny, radius * nx / ny, -radius, radius,
                          -100.0, 100.0);
    } else {
        view = glm::lookAt(center - back * radius, center, up);
        proj = glm::perspective(glm::radians(fov), nx * 1.0 / ny, 0.05, 20000.0);
    }
}


}
