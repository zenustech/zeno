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

struct ScaleHandler final : IGraphicHandler {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    vec3f center;
    float bound;
    float scale;
    int coord_sys;

    Program *lines_prog;
    std::unique_ptr<Buffer> lines_ebo;
    size_t lines_count;

    static constexpr float cube_factor = 0.08f;
    static constexpr float axis_factor = 0.8f;
    static constexpr float square_factor = 0.1f;
    static constexpr float icircle_factor = 0.2f; // inner circle
    static constexpr float ocircle_factor = 1.0f; // outer circle
    static constexpr float color_factor = 0.8f;

    explicit ScaleHandler(Scene *scene_, vec3f &center_, float scale_)
        : scene(scene_), center(center_), scale(scale_) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    }

    void draw() override {
        auto dist = glm::distance(scene->camera->m_pos, glm::vec3(center[0], center[1], center[2]));

        bound = dist / 5.0f * scale;

        vec3f color_x = vec3f(0.8, 0.2, 0.2) * (hover_mode == INTERACT_X ? 1.0 : color_factor);
        vec3f color_y = vec3f(0.2, 0.6, 0.2) * (hover_mode == INTERACT_Y ? 1.0 : color_factor);
        vec3f color_z = vec3f(0.2, 0.2, 0.8) * (hover_mode == INTERACT_Z ? 1.0 : color_factor);
        vec3f color_yz = vec3f(0.6, 0.2, 0.2) * (hover_mode == INTERACT_YZ ? 1.0 : color_factor);
        vec3f color_xz = vec3f(0.2, 0.6, 0.2) * (hover_mode == INTERACT_XZ ? 1.0 : color_factor);
        vec3f color_xy = vec3f(0.2, 0.2, 0.6) * (hover_mode == INTERACT_XY ? 1.0 : color_factor);
        vec3f color_circle = vec3f(1.0, 1.0, 1.0) * (hover_mode == INTERACT_XYZ ? 1.0 : color_factor);

        lines_prog = scene->shaderMan->compile_program(vert_code, frag_code);

        lines_prog->use();
        scene->camera->set_program_uniforms(lines_prog);

        auto x_axis = vec3f(1, 0, 0);
        auto y_axis = vec3f(0, 1, 0);
        auto z_axis = vec3f(0, 0, 1);

        auto axis_size = bound * axis_factor;
        auto cube_size = bound * cube_factor;
        auto square_size = bound * square_factor;
        auto icircle_size = bound * icircle_factor;
        auto ocircle_size = bound * ocircle_factor;

        // x axis
        if (mode == INTERACT_NONE || mode == INTERACT_X) {
            drawAxis(center, x_axis, color_x, axis_size, vbo);
            drawCube(center + axis_size * x_axis, y_axis, z_axis, color_x, cube_size, vbo);
        }
        // y axis
        if (mode == INTERACT_NONE || mode == INTERACT_Y) {
            drawAxis(center, y_axis, color_y, axis_size, vbo);
            drawCube(center + axis_size * y_axis, z_axis, x_axis, color_y, cube_size, vbo);
        }
        // z axis
        if (mode == INTERACT_NONE || mode == INTERACT_Z) {
            drawAxis(center, z_axis, color_z, axis_size, vbo);
            drawCube(center + axis_size * z_axis, x_axis, y_axis, color_z, cube_size, vbo);
        }
        // yz
        if (mode == INTERACT_NONE || mode == INTERACT_YZ) {
            drawSquare(center, y_axis, z_axis, color_yz, square_size, vbo);
        }
        // xz
        if (mode == INTERACT_NONE || mode == INTERACT_XZ) {
            drawSquare(center, z_axis, x_axis, color_xz, square_size, vbo);
        }
        // xy
        if (mode == INTERACT_NONE || mode == INTERACT_XY) {
            drawSquare(center, x_axis, y_axis, color_xy, square_size, vbo);
        }
        // xyz
        if (mode == INTERACT_NONE || mode == INTERACT_XYZ) {
            const auto& view = scene->camera->get_view_matrix();
            // http://www.opengl-tutorial.org/cn/intermediate-tutorials/billboards-particles/billboards/
            // always face camera
            // This is equivalent to mlutiplying (1,0,0) and (0,1,0) by inverse(ViewMatrix).
            // ViewMatrix is orthogonal (it was made this way), 
            // so its inverse is also its transpose, 
            // and transposing a matrix is "free" (inversing is slooow)
            auto right_world = vec3f(view[0][0], view[1][0], view[2][0]);
            auto up_world = vec3f(view[0][1], view[1][1], view[2][1]);
            drawCircle(center, right_world, up_world, color_circle, icircle_size, bound * 0.01f, vbo);
            drawCircle(center, right_world, up_world, color_circle, ocircle_size, bound * 0.01f, vbo);
        }
    }

    virtual int collisionTest(glm::vec3 ori, glm::vec3 dir) override {
        auto x_axis = glm::vec3(1, 0, 0);
        auto y_axis = glm::vec3(0, 1, 0);
        auto z_axis = glm::vec3(0, 0, 1);

        auto model_matrix = glm::translate(zeno::vec_to_other<glm::vec3>(center));
        const auto& view = scene->camera->get_view_matrix();
        
        float t;

        // axis handlers
        float axis_handler_l = bound * axis_factor;
        float axis_handler_w = bound * 0.1f;

        // x handler
        auto x_handler_max = axis_handler_l * x_axis + axis_handler_w * y_axis + axis_handler_w * z_axis;
        auto x_handler_min = glm::vec3(0, -axis_handler_w, -axis_handler_w);
        if (rayIntersectOBB(ori, dir, x_handler_min, x_handler_max, model_matrix, t)) {
            return INTERACT_X;
        }

        // y handler
        auto y_handler_max = axis_handler_w * x_axis + axis_handler_l * y_axis + axis_handler_w * z_axis;
        auto y_handler_min = glm::vec3(-axis_handler_w, 0, -axis_handler_w);
        if (rayIntersectOBB(ori, dir, y_handler_min, y_handler_max, model_matrix, t)) {
            return INTERACT_Y;
        }

        // z handler
        auto z_handler_max = axis_handler_w * x_axis + axis_handler_w * y_axis + axis_handler_l * z_axis;
        auto z_handler_min = glm::vec3(-axis_handler_w, -axis_handler_w, 0);
        if (rayIntersectOBB(ori, dir, z_handler_min, z_handler_max, model_matrix, t)) {
            return INTERACT_Z;
        }

        // plane handlers
        float square_ctr_offset = bound;
        float square_size = bound * square_factor;

        // xy handler
        auto xy_base = glm::normalize(x_axis + y_axis);
        auto xy_handler_max = xy_base * (square_ctr_offset + square_size);
        auto xy_handler_min = xy_base * (square_ctr_offset - square_size);
        if (rayIntersectSquare(ori, dir, xy_handler_min, xy_handler_max, z_axis, model_matrix)) {
            return INTERACT_XY;
        }

        // yz handler
        auto yz_base = glm::normalize(y_axis + z_axis);
        auto yz_handler_max = yz_base * (square_ctr_offset + square_size);
        auto yz_handler_min = yz_base * (square_ctr_offset - square_size);
        if (rayIntersectSquare(ori, dir, yz_handler_min, yz_handler_max, x_axis, model_matrix)) {
            return INTERACT_YZ;
        }

        // xz handler
        auto xz_base = glm::normalize(x_axis + z_axis);
        auto xz_handler_max = xz_base * (square_ctr_offset + square_size);
        auto xz_handler_min = xz_base * (square_ctr_offset - square_size);
        if (rayIntersectSquare(ori, dir, xz_handler_min, xz_handler_max, y_axis, model_matrix)) {
            return INTERACT_XZ;
        }

        // xyz handler
        // the other handlers are in the range of xyz handler, so check xyz at last 
        float i_radius = bound * icircle_factor;
        float o_radius = bound * ocircle_factor;
        float thickness = bound * 0.1f;
        auto right_world = vec3f(view[0][0], view[1][0], view[2][0]);
        auto up_world = vec3f(view[0][1], view[1][1], view[2][1]);
        if (rayIntersectRing(ori, dir, glm::vec3(0), o_radius, i_radius, zeno::vec_to_other<glm::vec3>(right_world),
            zeno::vec_to_other<glm::vec3>(up_world), thickness, model_matrix)) {
            return INTERACT_XYZ;
        }

        return INTERACT_NONE;
    }

    virtual void setCenter(zeno::vec3f c) override {
        center = c;
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

std::shared_ptr<IGraphicHandler> makeScaleHandler(Scene *scene, vec3f center, float scale) {
    return std::make_shared<ScaleHandler>(scene, center, scale);
}

} // namespace zenovis
