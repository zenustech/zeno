#include <zeno/funcs/PrimitiveTools.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/vao.h>


namespace zenovis {
namespace {

using opengl::Buffer;
using opengl::VAO;

using zeno::vec3f;
using zeno::PrimitiveObject;

using std::vector;
using std::unique_ptr;
using std::make_unique;

static const char* vert_shader = R"(
    # version 330
    layout (location = 0) in vec3 position;

    uniform mat4 mVP;
    uniform mat4 mInvVP;
    uniform mat4 mView;
    uniform mat4 mProj;
    uniform mat4 mInvView;
    uniform mat4 mInvProj;

    void main()
    {
        gl_Position = mVP * vec4(position, 1.0);
    }
)";

static const char * frag_shader = R"(
    # version 330
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(0.89, 0.57, 0.15, 1.0);
    }
)";

struct PrimitiveHighlight : IGraphicDraw {
    Scene* scene;

    unique_ptr<Buffer> vbo;
    unique_ptr<Buffer> ebo;
    // unique_ptr<VAO> vao;

    opengl::Program* shader;

    explicit PrimitiveHighlight(Scene* s) : scene(s) {
        vbo = make_unique<Buffer>(GL_ARRAY_BUFFER);
        ebo = make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        // vao = make_unique<VAO>();

        shader = scene->shaderMan->compile_program(vert_shader, frag_shader);
    }

    virtual void draw() override {
        CHECK_GL(glClear(GL_DEPTH_BUFFER_BIT));
        for (const auto& [prim_id, elements] : scene->selected_elements) {
            // ----- get primitive -----
            auto prim = dynamic_cast<PrimitiveObject*>(scene->objectsMan->get(prim_id).value());

            // ----- prepare data -----
            auto const &pos = prim->attr<zeno::vec3f>("pos");
            auto vertex_count = prim->size();
            auto selected_count = elements.size();
            vector<vec3f> mem(vertex_count);
            for (int i = 0; i < vertex_count; i++)
                mem[i] = pos[i];

            // ----- bind buffers -----
            // vao->bind();
            vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
            vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);
            // vao->unbind();

            // ----- draw selected vertices -----
//            if (scene->select_mode == zenovis::PICK_VERTEX) {
//
//            }

            // ----- draw selected edges -----
//            if (scene->select_mode == zenovis::PICK_LINE) {
//
//            }

            // ----- draw selected meshes -----
            if (scene->select_mode == zenovis::PICK_MESH) {
                // prepare indices
                vector<vec3f> ind(selected_count);
                int i = 0;
                for (const auto& idx : elements)
                    ind[i++] = prim->tris[idx];
                // draw meshes
                shader->use();
                scene->camera->set_program_uniforms(shader);
                // vao->bind();
                ebo->bind_data(ind.data(), selected_count * sizeof(ind[0]));
                CHECK_GL(glDrawElements(GL_TRIANGLES, selected_count * 3, GL_UNSIGNED_INT, 0));
//                ebo->bind_data(prim->tris.data(), prim->tris.size() * sizeof(prim->tris[0]));
//                CHECK_GL(glDrawElements(GL_TRIANGLES, prim->tris.size() * 3, GL_UNSIGNED_INT, 0));
                ebo->unbind();
            }
        }

        vbo->disable_attribute(0);
        vbo->unbind();
        // vao->unbind();
    }
};

}

std::unique_ptr<IGraphicDraw> makePrimitiveHighlight(Scene* scene) {
    return std::make_unique<PrimitiveHighlight>(scene);
}

}