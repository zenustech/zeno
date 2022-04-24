#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/IGraphic.h>
#include <zenovis/Scene.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>

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

struct GraphicAxis : IGraphic {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    Program *lines_prog;
    std::unique_ptr<Buffer> lines_ebo;
    size_t lines_count;

    explicit GraphicAxis(Scene *scene_) : scene(scene_) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        std::vector<zeno::vec3f> mem;
        int bound = 5;
        vec3f origin = vec3f(0, 0, 0);

        vec3f r = vec3f(0.8, 0.2, 0.2);
        vec3f g = vec3f(0.2, 0.6, 0.2);
        vec3f b = vec3f(0.2, 0.2, 0.8);

        mem.push_back(origin);
        mem.push_back(r);
        mem.push_back(vec3f(bound, 0, 0));
        mem.push_back(r);

        mem.push_back(origin);
        mem.push_back(g);
        mem.push_back(vec3f(0, bound, 0));
        mem.push_back(g);

        mem.push_back(origin);
        mem.push_back(b);
        mem.push_back(vec3f(0, 0, bound));
        mem.push_back(b);

        vertex_count = mem.size() / 2;
        lines_count = vertex_count / 2;

        vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

        lines_prog = scene->shaderMan->compile_program(vert_code, frag_code);
    }
    virtual void drawShadow(Light *light) override {
    }
    virtual void draw(bool reflect, bool depthPass) override {
        vbo->bind();
        vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
        vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

        lines_prog->use();
        scene->camera->set_program_uniforms(lines_prog);
        CHECK_GL(glDrawArrays(GL_LINES, 0, vertex_count));

        vbo->disable_attribute(0);
        vbo->disable_attribute(1);
        vbo->unbind();
    }
};

} // namespace

std::unique_ptr<IGraphic> makeGraphicAxis(Scene *scene) {
    return std::make_unique<GraphicAxis>(scene);
}

} // namespace zenovis
