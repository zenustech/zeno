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

using zeno::vec2i;
using zeno::vec3i;
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

        vec3 posEye = vec3(mView * vec4(position, 1.0));
        float dist = length(posEye);
        gl_PointSize = max(5.0, 10.0 / dist);
    }
)";

static const char * frag_shader = R"(
    # version 330
    out vec4 FragColor;

    void main()
    {
        gl_FragDepth = gl_FragCoord.z + 0.0001f;
        FragColor = vec4(0.89, 0.57, 0.15, 1.0);
    }
)";

struct PrimitiveHighlight : IGraphicDraw {
    Scene* scene;

    unique_ptr<Buffer> vbo;
    unique_ptr<Buffer> ebo;

    opengl::Program* shader;

    explicit PrimitiveHighlight(Scene* s) : scene(s) {
        vbo = make_unique<Buffer>(GL_ARRAY_BUFFER);
        ebo = make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);

        shader = scene->shaderMan->compile_program(vert_shader, frag_shader);
    }

    virtual void draw() override {
        CHECK_GL(glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE));
        glDepthFunc(GL_GREATER);
        CHECK_GL(glClearDepth(0.0));
        if (scene->select_mode == PICK_MODE::PICK_OBJECT) {
            for (const auto &prim_id : scene->selected) {
                // ----- get primitive -----
                PrimitiveObject *prim = nullptr;
                auto optional_prim = scene->objectsMan->get(prim_id);
                if (optional_prim.has_value())
                    prim = dynamic_cast<PrimitiveObject *>(scene->objectsMan->get(prim_id).value());
                else {
                    auto node_id = prim_id.substr(0, prim_id.find_first_of(':'));
                    for (const auto &[n, p] : scene->objectsMan->pairsShared()) {
                        if (n.find(node_id) != std::string::npos) {
                            prim = dynamic_cast<PrimitiveObject *>(p.get());
                            break;
                        }
                    }
                }
                if (!prim) continue;
                // ----- draw selected particles -----
                if (prim->tris->empty() && prim->polys->empty()) {
                    // prepare data
                    auto const &verts = prim->verts;

                    // bind buffers
                    vbo->bind_data(verts.data(), verts.size() * sizeof(verts[0]));
                    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

                    // draw particles
                    CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
                    shader->use();
                    scene->camera->set_program_uniforms(shader);
                    CHECK_GL(glDrawArrays(GL_POINTS, 0, verts.size()));
                    CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));

                    // unbind buffers
                    vbo->disable_attribute(0);
                    vbo->unbind();
                } else
                    continue;
            }
        }

        for (const auto& [prim_id, elements] : scene->selected_elements) {
            // ----- get primitive -----
            PrimitiveObject* prim = nullptr;
            auto optional_prim = scene->objectsMan->get(prim_id);
            if (optional_prim.has_value())
                prim = dynamic_cast<PrimitiveObject*>(scene->objectsMan->get(prim_id).value());
            else {
                auto node_id = prim_id.substr(0, prim_id.find_first_of(':'));
                for (const auto& [n, p] : scene->objectsMan->pairsShared()) {
                    if (n.find(node_id) != std::string::npos) {
                        prim = dynamic_cast<PrimitiveObject*>(p.get());
                        break;
                    }
                }
            }
            if (prim == nullptr) {
                return;
            }
            auto selected_count = elements.size();
            // ----- prepare data -----
            auto const &pos = prim->attr<zeno::vec3f>("pos");

            // ----- bind buffers -----
            vbo->bind_data(pos.data(), pos.size() * sizeof(pos[0]));
            vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

            // ----- draw selected vertices -----
            if (scene->select_mode == PICK_MODE::PICK_VERTEX) {
                // prepare indices
                CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
                vector<int> ind(selected_count);
                int i = 0;
                for (const auto& idx : elements)
                    ind[i++] = idx;
                // draw points
                shader->use();
                scene->camera->set_program_uniforms(shader);
                ebo->bind_data(ind.data(), selected_count * sizeof(ind[0]));
                CHECK_GL(glDrawElements(GL_POINTS, selected_count, GL_UNSIGNED_INT, 0));
                CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));
            }

            // ----- draw selected edges -----
            if (scene->select_mode == PICK_MODE::PICK_LINE) {
                if (prim->lines->empty()) return;
                // prepare indices
                vector<vec2i> ind(selected_count);
                int i = 0;
                for (const auto& idx : elements)
                    ind[i++] = prim->lines[idx];
                // draw lines
                shader->use();
                scene->camera->set_program_uniforms(shader);
                ebo->bind_data(ind.data(), selected_count * sizeof(ind[0]));
                CHECK_GL(glDrawElements(GL_LINES, selected_count * 2, GL_UNSIGNED_INT, 0));
                ebo->unbind();
            }

            // ----- draw selected meshes -----
            if (scene->select_mode == PICK_MODE::PICK_MESH) {
                // prepare indices
                vector<vec3i> ind(selected_count);
                int i = 0;
                for (const auto& idx : elements)
                    ind[i++] = prim->tris[idx];
                // draw meshes
                shader->use();
                scene->camera->set_program_uniforms(shader);
                ebo->bind_data(ind.data(), selected_count * sizeof(ind[0]));
                CHECK_GL(glDrawElements(GL_TRIANGLES, selected_count * 3, GL_UNSIGNED_INT, 0));
                ebo->unbind();
            }
        }

        vbo->disable_attribute(0);
        vbo->unbind();
    }
};

}

std::unique_ptr<IGraphicDraw> makePrimitiveHighlight(Scene* scene) {
    return std::make_unique<PrimitiveHighlight>(scene);
}

}