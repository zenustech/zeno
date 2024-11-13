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
        gl_FragDepth = gl_FragCoord.z * 1.02;
        FragColor = vec4(0.89, 0.57, 0.15, 1.0);
    }
)";

static const char* face_vert_shader = R"(
    # version 330
    layout (location = 0) in vec3 vPosition;

    out vec3 position;
    out vec3 posRWS;

    uniform mat4 mVP;
    uniform mat4 mInvVP;
    uniform mat4 mView;
    uniform mat4 mProj;
    uniform mat4 mInvView;
    uniform mat4 mInvProj;
    uniform vec3 mCameraCenter;

    void main()
    {
        position = vPosition;
        posRWS = vPosition - mCameraCenter;
        gl_Position = mVP * vec4(vPosition, 1.0);

        vec3 posEye = vec3(mView * vec4(vPosition, 1.0));
        float dist = length(posEye);
        gl_PointSize = max(5.0, 10.0 / dist);
    }
)";

static const char * face_frag_shader = R"(
    # version 330
    uniform mat4 mView;
    uniform mat4 mInvView;

    in vec3 position;
    in vec3 posRWS;
    out vec4 FragColor;

    vec3 pbr(vec3 albedo, float roughness, float metallic, float specular, vec3 nrm, vec3 idir, vec3 odir) {

      vec3 hdir = normalize(idir + odir);
      float NoH = max(1e-5, dot(hdir, nrm));
      float NoL = max(1e-5, dot(idir, nrm));
      float NoV = max(1e-5, dot(odir, nrm));
      float VoH = clamp(dot(odir, hdir), 1e-5, 1.);
      float LoH = clamp(dot(idir, hdir), 1e-5, 1.);

      vec3 f0 = metallic * albedo + (1. - metallic) * 0.16 * specular;
      vec3 fdf = f0 + (1. - f0) * pow(1. - VoH, 5.);

      roughness *= roughness;
      float k = (roughness + 1.) * (roughness + 1.) / 8.;
      float vdf = 0.25 / ((NoV * k + 1. - k) * (NoL * k + 1. - k));

      float alpha2 = max(0., roughness * roughness);
      float denom = 1. - NoH * NoH * (1. - alpha2);
      float ndf = alpha2 / (denom * denom);

      vec3 brdf = fdf * vdf * ndf * f0 + (1. - f0) * albedo;
      return brdf * NoL;
    }
    vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal) {
        vec3 color = vec3(0.0);
        vec3 light_dir;

        light_dir = normalize((mInvView * vec4(1., 2., 5., 0.)).xyz);
        color += vec3(0.45, 0.47, 0.5) * pbr(albedo, 0.44, 0.0, 1.0, normal, light_dir, view_dir);

        light_dir = normalize((mInvView * vec4(-4., -2., 1., 0.)).xyz);
        color += vec3(0.3, 0.23, 0.18) * pbr(albedo, 0.37, 0.0, 1.0, normal, light_dir, view_dir);

        light_dir = normalize((mInvView * vec4(3., -5., 2., 0.)).xyz);
        color += vec3(0.15, 0.2, 0.22) * pbr(albedo, 0.48, 0.0, 1.0, normal, light_dir, view_dir);
        color = pow(clamp(color, 0., 1.), vec3(1./2.2));
        return color;
    }

    vec3 calcRayDir(vec3 pos) {
        vec4 vpos = mView * vec4(pos, 1);
        return normalize(vpos.xyz);
    }

    void main() {
        gl_FragDepth = gl_FragCoord.z * 1.02;
        FragColor = vec4(0.89, 0.57, 0.15, 1.0);
        vec3 viewdir = -calcRayDir(position);
        vec3 normal = normalize(cross(dFdx(posRWS), dFdy(posRWS)));
        FragColor = vec4(studioShading(vec3(0.95, 0.27, 0), viewdir, normal), 1.0);
    }
)";

struct PrimitiveHighlight : IGraphicDraw {
    Scene* scene;

    unique_ptr<Buffer> vbo;
    unique_ptr<Buffer> ebo;

    opengl::Program* shader;
    opengl::Program* face_shader;

    explicit PrimitiveHighlight(Scene* s) : scene(s) {
        vbo = make_unique<Buffer>(GL_ARRAY_BUFFER);
        ebo = make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);

        shader = scene->shaderMan->compile_program(vert_shader, frag_shader);
        face_shader = scene->shaderMan->compile_program(face_vert_shader, face_frag_shader);
    }

    virtual void draw() override {
        if (scene->get_select_mode() == PICK_MODE::PAINT) {
            return;
        }
        CHECK_GL(glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE));
        glDepthFunc(GL_GREATER);
        CHECK_GL(glClearDepth(0.0));
        if (scene->get_select_mode() == PICK_MODE::PICK_OBJECT) {
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
        else if (scene->get_select_mode() == PICK_MODE::PICK_FACE_ATTR) {
            for (const auto& [prim_id, elements] : scene->selected_int_attr) {
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

                // ----- prepare data -----
                auto const &pos = prim->attr<zeno::vec3f>("pos");

                // ----- bind buffers -----
                vbo->bind_data(pos.data(), pos.size() * sizeof(pos[0]));
                vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

                // ----- draw selected meshes -----
                {
                    // prepare indices
                    vector<vec3i> ind;
                    std::string select_attr_str = elements.first;
                    if (prim->tris.attr_is<int>(select_attr_str)) {
                        ind.reserve(prim->tris.size());
                        auto &attr = prim->tris.attr<int>(select_attr_str);
                        for (auto i = 0; i < prim->tris.size(); i++) {
                            if (elements.second.count(attr[i])) {
                                ind.push_back(prim->tris[i]);
                            }
                        }
                        ind.shrink_to_fit();
                    }
                    // draw meshes
                    face_shader->use();
                    scene->camera->set_program_uniforms(face_shader);
                    ebo->bind_data(ind.data(), ind.size() * sizeof(ind[0]));
                    CHECK_GL(glDrawElements(GL_TRIANGLES, ind.size() * 3, GL_UNSIGNED_INT, 0));
                    ebo->unbind();
                }
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
            if (scene->get_select_mode() == PICK_MODE::PICK_VERTEX) {
                // prepare indices
                CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
                vector<int> ind(elements.begin(), elements.end());
                // draw points
                shader->use();
                scene->camera->set_program_uniforms(shader);
                ebo->bind_data(ind.data(), selected_count * sizeof(ind[0]));
                CHECK_GL(glDrawElements(GL_POINTS, selected_count, GL_UNSIGNED_INT, 0));
                CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));
            }

            // ----- draw selected edges -----
            if (scene->get_select_mode() == PICK_MODE::PICK_LINE) {
                if (prim->lines->empty()) return;
                // prepare indices
                vector<vec2i> ind(selected_count);
                int i = 0;
                for (const auto& idx : elements) {
                    if (idx < prim->lines.size()) {
                        ind[i++] = prim->lines[idx];
                    }
                }
                // draw lines
                shader->use();
                scene->camera->set_program_uniforms(shader);
                ebo->bind_data(ind.data(), selected_count * sizeof(ind[0]));
                CHECK_GL(glDrawElements(GL_LINES, selected_count * 2, GL_UNSIGNED_INT, 0));
                ebo->unbind();
            }

            // ----- draw selected meshes -----
            if (scene->get_select_mode() == PICK_MODE::PICK_FACE) {
                // prepare indices
                vector<vec3i> ind(selected_count);
                int i = 0;
                for (const auto& idx : elements) {
                    if (idx < prim->tris.size()) {
                        ind[i++] = prim->tris[idx];
                    }
                }
                // draw meshes
                face_shader->use();
                scene->camera->set_program_uniforms(face_shader);
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