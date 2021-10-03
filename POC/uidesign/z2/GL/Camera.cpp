#include <z2/GL/Camera.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>


namespace z2::GL {


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
