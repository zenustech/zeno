#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/scope.h>
#include <glm/glm.hpp>

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

struct GraphicInteractAxis final : IGraphicInteractDraw {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    vec3f center;
    bool hover;

    Program *lines_prog;
    std::unique_ptr<Buffer> lines_ebo;
    size_t lines_count;

    explicit GraphicInteractAxis(Scene *scene_, vec3f &center_)
        : scene(scene_), center(center_), hover(false) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    }

    virtual void draw() override {
        std::vector<zeno::vec3f> mem;

        float cx = scene->camera->m_zxx.cx;
        float cy = scene->camera->m_zxx.cy;
        float cz = scene->camera->m_zxx.cz;
        float theta = scene->camera->m_zxx.theta;
        float phi = scene->camera->m_zxx.phi;
        float radius = scene->camera->m_zxx.radius;

        auto camera_center = glm::vec3(cx, cy, cz);
        float cos_t = glm::cos(theta), sin_t = glm::sin(theta);
        float cos_p = glm::cos(phi), sin_p = glm::sin(phi);
        glm::vec3 front(cos_t * sin_p, sin_t, -cos_t * cos_p);

        auto camera_pos = camera_center - front * radius;

        auto dist = glm::distance(camera_pos, glm::vec3(center[0], center[1], center[2]));

        float bound = dist / 10;

        vec3f r = vec3f(0.8, 0.2, 0.2);
        vec3f g = vec3f(0.2, 0.6, 0.2);
        vec3f b = vec3f(0.2, 0.2, 0.8);

        auto [x, y, z] = center;

        mem.push_back(center);
        mem.push_back(r);
        mem.emplace_back(x + bound, y, z);
        mem.push_back(r);

        mem.push_back(center);
        mem.push_back(g);
        mem.emplace_back(x, y + bound, z);
        mem.push_back(g);

        mem.push_back(center);
        mem.push_back(b);
        mem.emplace_back(x, y, z + bound);
        mem.push_back(b);

        vertex_count = mem.size() / 2;
        lines_count = vertex_count / 2;

        vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

        lines_prog = scene->shaderMan->compile_program(vert_code, frag_code);

        vbo->bind();
        vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
        vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

        lines_prog->use();
        scene->camera->set_program_uniforms(lines_prog);

//        float scale_factor = 1 / scene->camera->m_zxx.radius;
//        glm::mat4 scale_mat = glm::mat4(1.0f);
//        scale_mat = glm::scale(scale_mat, glm::vec3(scale_factor, scale_factor, scale_factor));
//
//        lines_prog->set_uniform("mScale", scale_mat);

        float lwidth = 2.f;
        {
            //auto _1 = opengl::scopeGLEnable(GL_LINE_SMOOTH);
            //auto _2 = opengl::scopeGLLineWidth(lwidth);
            CHECK_GL(glDrawArrays(GL_LINES, 0, vertex_count));
        }

        vbo->disable_attribute(0);
        vbo->disable_attribute(1);
        vbo->unbind();
    }

    void setHovered(bool hovered) override {

    }
};

} // namespace

std::unique_ptr<IGraphicInteractDraw> makeGraphicInteractAxis(Scene *scene, vec3f center) {
    return std::make_unique<GraphicInteractAxis>(scene, center);
}

} // namespace zenovis
