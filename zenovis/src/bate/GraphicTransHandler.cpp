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

    varying vec3 position;
    varying vec3 color;

    void main() {
        gl_FragColor = vec4(color, 1.0);
    }
)";

struct TransHandler final : IGraphicHandler {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    vec3f center;
    float bound;
    float scale;
    int mode;
    int coord_sys;

    Program *lines_prog;
    std::unique_ptr<Buffer> lines_ebo;
    size_t lines_count;

    explicit TransHandler(Scene *scene_, vec3f &center_, float scale_)
        : scene(scene_), center(center_), scale(scale_), mode(INTERACT_NONE) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    }

    void draw() override {
        auto dist = glm::distance(scene->camera->m_lodcenter, glm::vec3(center[0], center[1], center[2]));

        bound = dist / 5.0f * scale;

        vec3f r = vec3f(0.8, 0.2, 0.2);
        vec3f g = vec3f(0.2, 0.6, 0.2);
        vec3f b = vec3f(0.2, 0.2, 0.8);

        lines_prog = scene->shaderMan->compile_program(vert_code, frag_code);

        lines_prog->use();
        scene->camera->set_program_uniforms(lines_prog);

        auto x_axis = vec3f(1, 0, 0);
        auto y_axis = vec3f(0, 1, 0);
        auto z_axis = vec3f(0, 0, 1);

        // x axis
        if (mode == INTERACT_NONE || mode == INTERACT_X) {
            drawAxis(center, x_axis, r, bound, vbo);
            drawCone(center + bound * x_axis, y_axis, z_axis, r, bound * 0.1f, vbo);
        }

        if (mode == INTERACT_NONE || mode == INTERACT_Y) {
            drawAxis(center, y_axis, g, bound, vbo);
            drawCone(center + bound * y_axis, z_axis, x_axis, g, bound * 0.1f, vbo);
        }

        if (mode == INTERACT_NONE || mode == INTERACT_Z) {
            drawAxis(center, z_axis, b, bound, vbo);
            drawCone(center + bound * z_axis, x_axis, y_axis, b, bound * 0.1f, vbo);
        }

        if (mode == INTERACT_NONE || mode == INTERACT_YZ)
            drawSquare(center, y_axis, z_axis, {0.6, 0.2, 0.2}, bound * 0.1f, vbo);

        if (mode == INTERACT_NONE || mode == INTERACT_XZ)
            drawSquare(center, z_axis, x_axis, {0.2, 0.6, 0.2}, bound * 0.1f, vbo);

        if (mode == INTERACT_NONE || mode == INTERACT_XY)
            drawSquare(center, x_axis, y_axis, {0.2, 0.2, 0.6}, bound * 0.1f, vbo);

        if (mode == INTERACT_NONE || mode == INTERACT_XYZ)
            drawCube(center, y_axis, z_axis, {0.51, 0.17, 0.85}, bound * 0.08f, vbo);
    }

    virtual int collisionTest(glm::vec3 ori, glm::vec3 dir) override {
        auto x_axis = glm::vec3(1, 0, 0);
        auto y_axis = glm::vec3(0, 1, 0);
        auto z_axis = glm::vec3(0, 0, 1);

        auto model_matrix =glm::translate(zeno::vec_to_other<glm::vec3>(center));

        // xyz handler
        float cube_size = bound * 0.08f; float t;
        auto xyz_handler_max = cube_size * glm::vec3(1.0f);
        auto xyz_handler_min = -xyz_handler_max;
        if (rayIntersectOBB(ori, dir, xyz_handler_min, xyz_handler_max, model_matrix, t)) {
            mode = INTERACT_XYZ;
            return INTERACT_XYZ;
        }

        // axis handlers
        float axis_handler_l = bound * 1.1f;
        float axis_handler_w = bound * 0.1f;

        // x handler
        auto x_handler_max = axis_handler_l * x_axis + axis_handler_w * y_axis + axis_handler_w * z_axis;
        auto x_handler_min = glm::vec3(0, -axis_handler_w, -axis_handler_w);
        if (rayIntersectOBB(ori, dir, x_handler_min, x_handler_max, model_matrix, t)) {
            mode = INTERACT_X;
            return INTERACT_X;
        }

        // y handler
        auto y_handler_max = axis_handler_w * x_axis + axis_handler_l * y_axis + axis_handler_w * z_axis;
        auto y_handler_min = glm::vec3(-axis_handler_w, 0, -axis_handler_w);
        if (rayIntersectOBB(ori, dir, y_handler_min, y_handler_max, model_matrix, t)) {
            mode = INTERACT_Y;
            return INTERACT_Y;
        }

        // z handler
        auto z_handler_max = axis_handler_w * x_axis + axis_handler_w * y_axis + axis_handler_l * z_axis;
        auto z_handler_min = glm::vec3(-axis_handler_w, -axis_handler_w, 0);
        if (rayIntersectOBB(ori, dir, z_handler_min, z_handler_max, model_matrix, t)) {
            mode = INTERACT_Z;
            return INTERACT_Z;
        }

        // plane handlers
        float square_ctr_offset = bound;
        float square_size = bound * 0.1f;

        // xy handler
        auto xy_base = glm::normalize(x_axis + y_axis);
        auto xy_handler_max = xy_base * (square_ctr_offset + square_size);
        auto xy_handler_min = xy_base * (square_ctr_offset - square_size);
        if (rayIntersectSquare(ori, dir, xy_handler_min, xy_handler_max, z_axis, model_matrix)) {
            mode = INTERACT_XY;
            return INTERACT_XY;
        }

        // yz handler
        auto yz_base = glm::normalize(y_axis + z_axis);
        auto yz_handler_max = yz_base * (square_ctr_offset + square_size);
        auto yz_handler_min = yz_base * (square_ctr_offset - square_size);
        if (rayIntersectSquare(ori, dir, yz_handler_min, yz_handler_max, x_axis, model_matrix)) {
            mode = INTERACT_YZ;
            return INTERACT_YZ;
        }

        // xz handler
        auto xz_base = glm::normalize(x_axis + z_axis);
        auto xz_handler_max = xz_base * (square_ctr_offset + square_size);
        auto xz_handler_min = xz_base * (square_ctr_offset - square_size);
        if (rayIntersectSquare(ori, dir, xz_handler_min, xz_handler_max, y_axis, model_matrix)) {
            mode = INTERACT_XZ;
            return INTERACT_XZ;
        }

        mode = INTERACT_NONE;
        return INTERACT_NONE;
    }

    virtual void setCenter(zeno::vec3f c) override {
        center = c;
    }

    virtual void setMode(int m) override {
        mode = m;
    }

    virtual void setCoordSys(int c) override {
        coord_sys = c;
    }

    virtual std::optional<glm::vec3> getIntersect(glm::vec3 ray_origin, glm::vec3 ray_direction) override {
        return std::nullopt;
    }

    virtual void resize(float s) override {
        scale = s;
    }
};

} // namespace

std::shared_ptr<IGraphicHandler> makeTransHandler(Scene *scene, vec3f center, float scale) {
    return std::make_shared<TransHandler>(scene, center, scale);
}

} // namespace zenovis
