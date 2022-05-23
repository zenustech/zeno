#include <zeno/utils/vec.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
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
    uniform mat4 mTrans;

    attribute vec3 vPosition;


    void main() {
        gl_Position = mTrans * vec4(vPosition, 1.0);
    }
)";

static const char *frag_code = R"(
    #version 120

    void main() {
        gl_FragColor = vec4(0, 0, 0, 0.2);
    }
)";

struct GraphicSelectBox final : IGraphicDraw {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    Program *prog;

    explicit GraphicSelectBox(Scene *scene_) : scene(scene_) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        std::vector<zeno::vec3f> mem;
        float bound = 1.0f;
        mem.push_back(vec3f(-1, 1, 0));
        mem.push_back(vec3f(-1, -1, 0));
        mem.push_back(vec3f(1, -1, 0));
        mem.push_back(vec3f(-1, 1, 0));
        mem.push_back(vec3f(1, -1, 0));
        mem.push_back(vec3f(1, 1, 0));
        vertex_count = mem.size();

        vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

        prog = scene->shaderMan->compile_program(vert_code, frag_code);
    }

    virtual void draw() override {
        if (scene->select_box.has_value()) {
            zeno::vec4f rect = scene->select_box.value();
            auto rect_ = rect * 2 - 1;
            rect_[1] *= -1;
            rect_[3] *= -1;
            vbo->bind();
            vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

            prog->use();

            glm::mat4 mTrans = glm::mat4(1.0f);
            mTrans = glm::translate(mTrans, glm::vec3((rect_[0] + rect_[2]) * 0.5f, (rect_[1] + rect_[3]) * 0.5f, 0));
            mTrans = glm::scale(mTrans, glm::vec3(rect[2] - rect[0], rect[3] - rect[1], 0.0));

            prog->set_uniform("mTrans", mTrans);

            CHECK_GL(glDrawArrays(GL_TRIANGLES, 0, vertex_count));

            vbo->disable_attribute(0);
            vbo->unbind();
        }
    }
};

} // namespace

std::unique_ptr<IGraphicDraw> makeGraphicSelectBox(Scene *scene) {
    return std::make_unique<GraphicSelectBox>(scene);
}

} // namespace zenovis
