#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/scope.h>

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
    uniform float mCameraRadius;

    attribute vec3 vPosition;

    void main() {
        vec3 pos = vPosition;
        gl_Position = vec4(pos, 1.0);
    }
)";

static const char *frag_code = R"(
    #version 120

    uniform float alpha;

    void main() {
        gl_FragColor = vec4(100, 100, 100, alpha);
    }
)";

struct GraphicRing final : IGraphicInteractDraw {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    vec3f center;
    bool hover;

    Program *prog;

    explicit GraphicRing(Scene *scene_, vec3f center_)
        : scene(scene_), center(center_), hover(false) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    }

    virtual void draw() override {
        std::vector<zeno::vec3f> mem;

        float r1 = 0.1, r2 = 0.08;
        float pi = 3.1415926;

        float ar = scene->camera->getAspect();

        auto vp = scene->camera->m_proj * scene->camera->m_view;
        auto screen_center = vp * glm::vec4(center[0], center[1], center[2], 1.0);

        float x = screen_center[0] / screen_center[3], y = screen_center[1] / screen_center[3];

        for (float theta = 0; theta <= 1.0; theta += 0.01) {
            float sin_theta = sin(theta * 2 * pi);
            float cos_theta = cos(theta * 2 * pi);
            zeno::vec3f p1(x + r1 * cos_theta / ar, y + r1 * sin_theta, 0);
            zeno::vec3f p2(x + r2 * cos_theta / ar, y + r2 * sin_theta, 0);
            mem.emplace_back(p1);
            mem.emplace_back(p2);
        }

        vertex_count = mem.size();

        vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

        prog = scene->shaderMan->compile_program(vert_code, frag_code);

        vbo->bind();
        vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

        prog->use();

        scene->camera->set_program_uniforms(prog);

        if (hover)
            prog->set_uniform("alpha", 1.0);
        else
            prog->set_uniform("alpha", 0.5);

        CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, vertex_count));

        vbo->disable_attribute(0);
        vbo->unbind();
    }

    void setHovered(bool hovered) override {
        hover = hovered;
    }
};

} // namespace

std::unique_ptr<IGraphicInteractDraw> makeGraphicRing(Scene *scene, vec3f center) {
    return std::make_unique<GraphicRing>(scene, center);
}

} // namespace zenovis
