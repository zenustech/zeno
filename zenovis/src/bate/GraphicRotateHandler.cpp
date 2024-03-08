#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/bate/GraphicHandlerUtils.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/scope.h>
#include <glm/gtx/transform.hpp>

namespace zenovis {
namespace {

using opengl::Buffer;
using opengl::Program;
using zeno::vec3f;

static const char *vert_code = R"(
    #version 120

    uniform mat4 mVP;
    uniform mat4 mInvVP;
    uniform mat4 mView;
    uniform mat4 mProj;
    uniform mat4 mInvView;
    uniform mat4 mInvProj;

    uniform mat4 mScale;

    attribute vec3 vPosition;
    attribute vec3 vColor;

    varying vec3 position;
    varying vec3 color;

    void main() {
        position = vPosition;
        color = vColor;

        gl_Position = mVP * vec4(position, 1.0);
    }
)";

static const char *frag_code = R"(
    #version 120

    uniform mat4 mVP;
    uniform mat4 mInvVP;
    uniform mat4 mView;
    uniform mat4 mProj;
    uniform mat4 mInvView;
    uniform mat4 mInvProj;

    uniform float alpha;

    varying vec3 position;
    varying vec3 color;

    void main() {
        gl_FragColor = vec4(color, alpha);
    }
)";

struct RotateHandler final : IGraphicHandler {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    std::unique_ptr<Buffer> ibo;
    size_t vertex_count;

    vec3f center;
    vec3f localX;
    vec3f localY;
    float bound;
    float scale;
    int coord_sys;

    Program *lines_prog;
    std::unique_ptr<Buffer> lines_ebo;
    size_t lines_count;

    explicit RotateHandler(Scene *scene_, vec3f &center_, vec3f localX_, vec3f localY_, float scale_)
        : scene(scene_), center(center_), localX(localX_), localY(localY_), scale(scale_) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        ibo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
    }

    void draw() override {
        auto dist = glm::distance(scene->camera->m_lodcenter, glm::vec3(center[0], center[1], center[2]));

        bound = dist / 5.0f * scale;

        constexpr float color_factor = 0.8f;
        vec3f color_yz = vec3f(0.8, 0.2, 0.2) * (hover_mode == INTERACT_YZ ? 1.0 : color_factor);
        vec3f color_xz = vec3f(0.2, 0.6, 0.2) * (hover_mode == INTERACT_XZ ? 1.0 : color_factor);
        vec3f color_xy = vec3f(0.2, 0.2, 0.8) * (hover_mode == INTERACT_XY ? 1.0 : color_factor);
        vec3f color_xyz = vec3f(1.0, 1.0, 1.0) * (hover_mode == INTERACT_XYZ ? 1.0 : color_factor);

        lines_prog = scene->shaderMan->compile_program(vert_code, frag_code);

        lines_prog->use();
        scene->camera->set_program_uniforms(lines_prog);

        auto x_axis = localX;
        auto y_axis = localY;
        auto z_axis = zeno::cross(x_axis, y_axis);

        lines_prog->set_uniform("alpha", 1.0f);

        float size = 0.02f;

        if (mode == INTERACT_NONE || mode == INTERACT_YZ)
            drawCircle(center, y_axis, z_axis, color_yz, bound, bound * size, vbo);

        if (mode == INTERACT_NONE || mode == INTERACT_XZ)
            drawCircle(center, z_axis, x_axis, color_xz, bound, bound * size, vbo);

        if (mode == INTERACT_NONE || mode == INTERACT_XY)
            drawCircle(center, x_axis, y_axis, color_xy, bound, bound * size, vbo);

        lines_prog->set_uniform("alpha", 0.3f);

        if (mode == INTERACT_NONE || mode == INTERACT_XYZ)
            drawSphere(center, color_xyz, bound * 0.9f, vbo, ibo);
    }

    virtual int collisionTest(glm::vec3 ray_origin, glm::vec3 ray_direction) override {
        auto x_axis = zeno::vec_to_other<glm::vec3>(localX);
        auto y_axis = zeno::vec_to_other<glm::vec3>(localY);
        auto z_axis = glm::cross(x_axis, y_axis);

        auto model_matrix =glm::translate(zeno::vec_to_other<glm::vec3>(center));

        auto ctr = glm::vec3(0);
        float i_radius = bound * 0.9f;
        float o_radius = bound * 1.1f;
        float thickness = bound * 0.1f;

        // xy handler
        if (rayIntersectRing(ray_origin, ray_direction, ctr, o_radius, i_radius, y_axis, z_axis, thickness, model_matrix)) {
            return INTERACT_YZ;
        }

        // yz handler
        if (rayIntersectRing(ray_origin, ray_direction, ctr, o_radius, i_radius, z_axis, x_axis, thickness, model_matrix)) {
            return INTERACT_XZ;
        }

        // xz handler
        if (rayIntersectRing(ray_origin, ray_direction, ctr, o_radius, i_radius, x_axis, y_axis, thickness, model_matrix)) {
            return INTERACT_XY;
        }

        // xyz handler
        if (rayIntersectSphere(ray_origin, ray_direction, zeno::vec_to_other<glm::vec3>(center), i_radius).has_value()) {
            return INTERACT_XYZ;
        }

        return INTERACT_NONE;
    }

    virtual void setCenter(zeno::vec3f c, zeno::vec3f x, zeno::vec3f y) override {
        center = c;
        localX = x;
        localY = y;
    }

    virtual void setCoordSys(int c) override {
        coord_sys = c;
    }

    virtual std::optional<glm::vec3> getIntersect(glm::vec3 ray_origin, glm::vec3 ray_direction) override {
        auto intersect = rayIntersectSphere(ray_origin, ray_direction,
                                    zeno::vec_to_other<glm::vec3>(center), bound * 0.9f);
        if (intersect.has_value()) return ray_origin + intersect.value() * ray_direction;
        return std::nullopt;
    }

    virtual void resize(float s) override {
        scale = s;
    }
};

} // namespace

std::shared_ptr<IGraphicHandler> makeRotateHandler(Scene *scene, vec3f center, vec3f localX_, vec3f localY_, float scale) {
    return std::make_shared<RotateHandler>(scene, center, localX_, localY_, scale);
}

} // namespace zenovis
