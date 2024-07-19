#include <zeno/utils/vec.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/DrawOptions.h>

namespace zenovis {
namespace {

using opengl::Buffer;
using opengl::Program;
using zeno::vec3f;

static const char *vert_code = R"(
    #version 120
    uniform mat4 mTrans;
    uniform float mBrushSize;

    attribute vec3 vPosition;

    void main() {
        gl_PointSize = mBrushSize;
        gl_Position = mTrans * vec4(vPosition, 1.0);
    }
)";

static const char *frag_code = R"(
    #version 120

    void main() {
        vec2 coor = gl_PointCoord * 2 - 1;
        float len2 = dot(coor, coor);
        if (len2 > 1) {
            discard;
        }
        gl_FragColor = vec4(1, 0, 0, 1);
    }
)";

struct GraphicPainterCursor final : IGraphicDraw {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    Program *prog;

    explicit GraphicPainterCursor(Scene *scene_) : scene(scene_) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        std::vector<zeno::vec3f> mem;
        mem.emplace_back(0, 0, 0);
        vertex_count = mem.size();

        vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

        prog = scene->shaderMan->compile_program(vert_code, frag_code);
    }

    virtual void draw() override {
        if (scene->painter_cursor.has_value()) {
            vbo->bind();
            vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

            prog->use();

            glm::mat4 mTrans = glm::mat4(1.0f);
            zeno::vec4f painter_cursor = scene->painter_cursor.value();
            float x_trans = painter_cursor[0] / painter_cursor[2] * 2.0f - 1.0f;
            float y_trans = 1.0f - painter_cursor[1] / painter_cursor[3] * 2.0f;
            mTrans = glm::translate(mTrans, glm::vec3(x_trans, y_trans, 0));

            prog->set_uniform("mTrans", mTrans);
            prog->set_uniform("mBrushSize", scene->drawOptions->brush_size);

            CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
            glDisable(GL_DEPTH_TEST);
            CHECK_GL(glDrawArrays(GL_POINTS, 0, vertex_count));
            glEnable(GL_DEPTH_TEST);
            CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));

            vbo->disable_attribute(0);
            vbo->unbind();
        }
    }
};

} // namespace

std::unique_ptr<IGraphicDraw> makeGraphicPainterCursor(Scene *scene) {
    return std::make_unique<GraphicPainterCursor>(scene);
}

} // namespace zenovis
