#include <zeno/funcs/PrimitiveTools.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/texture.h>
#include <zenovis/opengl/vao.h>

#include <unordered_map>
#include <fstream>
#include <random>
#include <algorithm>

namespace zenovis {
namespace {

using opengl::FBO;
using opengl::VAO;
using opengl::Buffer;
using opengl ::Texture;
using opengl::Program;

using zeno::vec2i;
using zeno::vec3i;
using zeno::vec3f;
using zeno::PrimitiveObject;

using std::unique_ptr;
using std::make_unique;
using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;

static const char * obj_vert_code = R"(
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

static const char * obj_frag_code = R"(
    # version 330
    out uvec3 FragColor;

    uniform uint gObjectIndex;

    void main()
    {
        FragColor = uvec3(gObjectIndex, 0, 0);
    }
)";

static const char * vert_vert_code = R"(
    # version 330
    layout (location = 0) in vec3 position;
    flat out uint gVertexIndex;

    uniform mat4 mVP;
    uniform mat4 mInvVP;
    uniform mat4 mView;
    uniform mat4 mProj;
    uniform mat4 mInvView;
    uniform mat4 mInvProj;

    uniform sampler2D depthTexture;

    void main()
    {
        gVertexIndex = uint(gl_VertexID);
        gl_Position = mVP * vec4(position, 1.0);
    }
)";

static const char* vert_frag_code = R"(
    # version 330
    flat in uint gVertexIndex;
    out uvec3 FragColor;

    uniform uint gObjectIndex;

    void main()
    {
        FragColor = uvec3(gObjectIndex, gVertexIndex + 1u, 0);
    }
)";

static const char* prim_frag_code = R"(
    # version 330
    out uvec3 FragColor;

    uniform uint gObjectIndex;

    void main()
    {
        FragColor = uvec3(gObjectIndex, gl_PrimitiveID + 1, 0);
    }
)";

static const char* empty_frag_code = R"(
    # version 330
    out uvec3 FragColor;

    void main()
    {
        FragColor = uvec3(0, 0, 0);
    }
)";

static const char* empty_and_offset_frag_code = R"(
    # version 330
    out uvec3 FragColor;

    uniform float offset;

    void main()
    {
        gl_FragDepth = gl_FragCoord.z + offset;
        FragColor = uvec3(0, 0, 0);
    }
)";



static void load_buffer_to_image(unsigned int* ids, int w, int h, const std::string& file_name = "output.ppm") {
    unordered_map<unsigned int, vec3i> color_set;
    color_set[0] = {20, 20, 20};
    color_set[1] = {90, 20, 20};
    color_set[1047233823] = {10, 10, 10};

    auto random_color = [](std::default_random_engine &e) -> vec3i{
        std::uniform_int_distribution<int> u(0, 255);
        auto r = u(e);
        auto g = u(e);
        auto b = u(e);
        return {r, g, b};
    };

    unordered_map<unsigned int, int> obj_count;

    std::ofstream os;
    os.open(file_name, std::ios::out);
    os << "P3\n" << w << " " << h << "\n255\n";
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            auto id = ids[w * j + i];
            vec3i color;
            if (color_set.find(id) != color_set.end()) {
                color = color_set[id];
                obj_count[id]++;
            }
            else {
                printf("found obj id : %u\n", id);
                std::default_random_engine e(id);
                color = random_color(e);
                color_set[id] = color;
            }
            os << color[0] << " " << color[1] << " " << color[2] << "\t";
        }
        os << "\n";
    }
    for (auto [key, value] : obj_count)
        printf("obj id: %u, count: %d, color: (%d, %d, %d)\n", key, value, color_set[key][0],
               color_set[key][1], color_set[key][2]);
    printf("load done.\n");
}

// framebuffer picker referring to https://doc.yonyoucloud.com/doc/wiki/project/modern-opengl-tutorial/tutorial29.html
struct FrameBufferPicker : IPicker {
    Scene* scene;
    string focus_prim_name;

    unique_ptr<FBO> fbo;
    unique_ptr<Texture> picking_texture;
    unique_ptr<Texture> depth_texture;

    unique_ptr<Buffer> vbo;
    unique_ptr<Buffer> ebo;
    unique_ptr<VAO> vao;

    Program* obj_shader;
    Program* vert_shader;
    Program* prim_shader;
    Program* empty_shader;
    Program* empty_and_offset_shader;

    int w, h;
    unordered_map<unsigned int, string> id_table;

    struct PixelInfo {
        unsigned int obj_id;
        unsigned int elem_id;
        unsigned int blank;

        PixelInfo() {
            obj_id = 0;
            elem_id = 0;
            blank = 0;
        }

        bool has_object() const {
            return obj_id != blank;
        }

        bool has_element() const {
            return elem_id != blank;
        }
    };

    explicit FrameBufferPicker(Scene* s) : scene(s) {
        // generate draw buffer
        vbo = make_unique<Buffer>(GL_ARRAY_BUFFER);
        ebo = make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        vao = make_unique<VAO>();

        // prepare shaders
        obj_shader = scene->shaderMan->compile_program(obj_vert_code, obj_frag_code);
        vert_shader = scene->shaderMan->compile_program(vert_vert_code, vert_frag_code);
        prim_shader = scene->shaderMan->compile_program(obj_vert_code, prim_frag_code);
        empty_shader = scene->shaderMan->compile_program(obj_vert_code, empty_frag_code);
        empty_and_offset_shader = scene->shaderMan->compile_program(obj_vert_code, empty_and_offset_frag_code);
    }

    ~FrameBufferPicker() {
        destroy_buffers();
    }

    void generate_buffers() {
        // generate framebuffer
        fbo = make_unique<FBO>();
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo));

        // get viewport size
        w = scene->camera->m_nx;
        h = scene->camera->m_ny;

        // generate picking texture
        picking_texture = make_unique<Texture>();
        CHECK_GL(glBindTexture(picking_texture->target, picking_texture->tex));
        CHECK_GL(glTexImage2D(picking_texture->target, 0, GL_RGB32UI, w, h,
                              0, GL_RGB_INTEGER, GL_UNSIGNED_INT, NULL));
        CHECK_GL(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                        GL_TEXTURE_2D, picking_texture->tex, 0));

        // generate depth texture
        depth_texture = make_unique<Texture>();
        CHECK_GL(glBindTexture(depth_texture->target, depth_texture->tex));
        CHECK_GL(glTexImage2D(depth_texture->target, 0, GL_DEPTH_COMPONENT, w, h,
                              0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
        CHECK_GL(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                        GL_TEXTURE_2D, depth_texture->tex, 0));

        // check fbo
        if(!fbo->complete()) printf("fbo error\n");

        // unbind fbo & texture
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
        fbo->unbind();
    }

    void destroy_buffers() {
        fbo.reset();
        picking_texture.reset();
        depth_texture.reset();
    }

    virtual void draw() override {
        // enable framebuffer writing
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        // construct prim set
        // use focus_prim if focus_prim_name is not empty else all prims
        vector<std::pair<string, std::shared_ptr<zeno::IObject>>> prims;
        auto prims_shared = scene->objectsMan->pairsShared();
        if (!focus_prim_name.empty()) {
            std::shared_ptr<zeno::IObject> focus_prim;
            for (const auto& [k, v] : prims_shared) {
                if (focus_prim_name == k)
                    focus_prim = v;
            }
            if (focus_prim) prims.emplace_back(focus_prim_name, focus_prim);
        }
        else
            prims = std::move(prims_shared);

        // shading primitive objects
        for (unsigned int id = 0; id < prims.size(); id++) {
            auto it = prims.begin() + id;
            auto prim = dynamic_cast<PrimitiveObject*>(it->second.get());
            if (prim && prim->has_attr("pos")) {
                // prepare vertices data
                auto const &pos = prim->attr<zeno::vec3f>("pos");
                vao->bind();
                vbo->bind_data(pos.data(), pos.size() * sizeof(pos[0]));
                vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

                bool pick_particle = false;
                if (scene->select_mode == PICK_MODE::PICK_OBJECT) {
                    pick_particle = prim->tris->empty() && prim->quads->empty() && prim->polys->empty() && prim->loops->empty();
                    CHECK_GL(glEnable(GL_DEPTH_TEST));
                    CHECK_GL(glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE));
                    glDepthFunc(GL_GREATER);
                    CHECK_GL(glClearDepth(0.0));

                    // shader uniform
                    obj_shader->use();
                    scene->camera->set_program_uniforms(obj_shader);
                    CHECK_GL(glUniform1ui(glGetUniformLocation(obj_shader->pro, "gObjectIndex"), id + 1));
                    // draw prim
                    if (prim->tris.size()) {
                        ebo->bind_data(prim->tris.data(), prim->tris.size() * sizeof(prim->tris[0]));
                        CHECK_GL(glDrawElements(GL_TRIANGLES, prim->tris.size() * 3, GL_UNSIGNED_INT, 0));
                    }
                    else if (prim->polys.size()) {
                        std::vector<vec3i> tris;
                        for (auto [start, len]: prim->polys) {
                            for (auto i = 2; i < len; i++) {
                                tris.emplace_back(
                                    prim->loops[start],
                                    prim->loops[start + i - 1],
                                    prim->loops[start + i]
                                );
                            }
                        }
                        ebo->bind_data(tris.data(), tris.size() * sizeof(tris[0]));
                        CHECK_GL(glDrawElements(GL_TRIANGLES, tris.size() * 3, GL_UNSIGNED_INT, 0));
                    }
                    ebo->unbind();
                    CHECK_GL(glDisable(GL_DEPTH_TEST));
                }

                if (scene->select_mode == PICK_MODE::PICK_VERTEX || pick_particle) {
                    // ----- enable depth test -----
                    CHECK_GL(glEnable(GL_DEPTH_TEST));
                    CHECK_GL(glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE));
                    glDepthFunc(GL_GREATER);
                    CHECK_GL(glClearDepth(0.0));
                    // CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

                    // ----- draw points -----
                    vert_shader->use();
                    scene->camera->set_program_uniforms(vert_shader);
                    CHECK_GL(glUniform1ui(glGetUniformLocation(vert_shader->pro, "gObjectIndex"), id + 1));
                    CHECK_GL(glDrawArrays(GL_POINTS, 0, pos.size()));

                    // ----- draw object to cover invisible points -----
                    empty_and_offset_shader->use();
                    empty_and_offset_shader->set_uniform("offset", -0.00001f);
                    scene->camera->set_program_uniforms(empty_and_offset_shader);

                    auto tri_count = prim->tris.size();
                    ebo->bind_data(prim->tris.data(), tri_count * sizeof(prim->tris[0]));
                    CHECK_GL(glDrawElements(GL_TRIANGLES, tri_count * 3, GL_UNSIGNED_INT, 0));
                    ebo->unbind();

                    // ----- disable depth test -----
                    CHECK_GL(glDisable(GL_DEPTH_TEST));
                }

                if (scene->select_mode == PICK_MODE::PICK_LINE) {
                    // ----- enable depth test -----
                    CHECK_GL(glEnable(GL_DEPTH_TEST));
                    CHECK_GL(glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE));
                    glDepthFunc(GL_GREATER);
                    CHECK_GL(glClearDepth(0.0));
                    // CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
                    // ----- draw lines -----
                    prim_shader->use();
                    scene->camera->set_program_uniforms(prim_shader);
                    CHECK_GL(glUniform1ui(glGetUniformLocation(prim_shader->pro, "gObjectIndex"), id + 1));
                    auto line_count = prim->lines.size();
                    if (!line_count) {
                        // compute lines' indices
                        struct cmp_line {
                            bool operator()(vec2i v1, vec2i v2) const {
                                return (v1[0] == v2[0] && v1[1] == v2[1]) || (v1[0] == v2[1] && v1[1] == v2[0]);
                            }
                        };
                        struct hash_line {
                            size_t operator()(const vec2i& v) const {
                                return std::hash<int>()(v[0]) ^ std::hash<int>()(v[1]);
                            }
                        };
                        unordered_set<zeno::vec2i, hash_line, cmp_line> lines;
                        for (auto & tri : prim->tris) {
                            auto& a = tri[0];
                            auto& b = tri[1];
                            auto& c = tri[2];
                            lines.insert(vec2i{a, b});
                            lines.insert(vec2i{b, c});
                            lines.insert(vec2i{c, a});
                        }
                        for (auto l : lines) prim->lines.push_back(l);
                        line_count = prim->lines.size();
                    }
                    ebo->bind_data(prim->lines.data(), line_count * sizeof(prim->lines[0]));
                    CHECK_GL(glDrawElements(GL_LINES, line_count * 2, GL_UNSIGNED_INT, 0));
                    ebo->unbind();

                    // ----- draw object to cover invisible lines -----
                    empty_shader->use();
                    scene->camera->set_program_uniforms(empty_shader);
                    auto tri_count = prim->tris.size();
                    ebo->bind_data(prim->tris.data(), tri_count * sizeof(prim->tris[0]));
                    CHECK_GL(glDrawElements(GL_TRIANGLES, tri_count * 3, GL_UNSIGNED_INT, 0));
                    ebo->unbind();
                    // ----- disable depth test -----
                    CHECK_GL(glDisable(GL_DEPTH_TEST));
                }

                if (scene->select_mode == PICK_MODE::PICK_MESH) {
                    // ----- enable depth test -----
                    CHECK_GL(glEnable(GL_DEPTH_TEST));
                    CHECK_GL(glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE));
                    glDepthFunc(GL_GREATER);
                    CHECK_GL(glClearDepth(0.0));
                    // CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
                    // ----- draw triangles -----
                    prim_shader->use();
                    scene->camera->set_program_uniforms(prim_shader);
                    CHECK_GL(glUniform1ui(glGetUniformLocation(prim_shader->pro, "gObjectIndex"), id + 1));
                    auto tri_count = prim->tris.size();
                    ebo->bind_data(prim->tris.data(), tri_count * sizeof(prim->tris[0]));
                    CHECK_GL(glDrawElements(GL_TRIANGLES, tri_count * 3, GL_UNSIGNED_INT, 0));
                    ebo->unbind();
                    // ----- disable depth test -----
                    CHECK_GL(glDisable(GL_DEPTH_TEST));
                }

                // unbind vbo
                vbo->disable_attribute(0);
                vbo->unbind();
                vao->unbind();

                // store object's name
                id_table[id + 1] = it->first;
            }
        }
        fbo->unbind();
    }

    virtual string getPicked(int x, int y) override {
        // re-generate buffers for possible window resize
        generate_buffers();

        // draw framebuffer
        draw();

        // check fbo
        if (!fbo->complete()) return "";
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->fbo));
        CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0));

        PixelInfo pixel;
        // qt coordinate is from left up to right bottom
        //  (x, y)------> w
        //    | \
        //    |  \
        //    |   .
        //    v h
        // read pixel from left bottom to right up
        //    ^ h
        //    |   .
        //    |  /
        //    | /
        //  (x, y)------> w

        CHECK_GL(glReadPixels(x, h - y - 1, 1, 1, GL_RGB_INTEGER, GL_UNSIGNED_INT, &pixel));

        // output buffer to image
//        auto* pixels = new PixelInfo[w * h];
//        CHECK_GL(glReadPixels(0, 0, w, h, GL_RGB_INTEGER, GL_UNSIGNED_INT, pixels));
//        auto* ids = new unsigned int[w * h];
//        for (int i=0; i<w*h; i++)
//            ids[i] = pixels[i].obj_id;
//        load_buffer_to_image(ids, w, h);

        CHECK_GL(glReadBuffer(GL_NONE));
        fbo->unbind();

        string result;
        if (scene->select_mode == PICK_MODE::PICK_OBJECT) {
            if (!pixel.has_object() || !id_table.count(pixel.obj_id)) return "";
            result = id_table[pixel.obj_id];
        }
        else {
            if (!pixel.has_object() || !pixel.has_element() || !id_table.count(pixel.obj_id)) return "";
            result = id_table[pixel.obj_id] + ":" + std::to_string(pixel.elem_id - 1);
        }

        destroy_buffers();

        return result;
    }

    virtual string getPicked(int x0, int y0, int x1, int y1) override {
        // re-generate buffers for possible window resize
        generate_buffers();

        // draw framebuffer
        draw();

        // check fbo
        if (!fbo->complete()) return "";

        // prepare fbo
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->fbo));
        CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0));

        // convert coordinates
        int start_x = x0 < x1 ? x0 : x1;
        int start_y = y0 > y1 ? y0 : y1;
        start_y = h - start_y - 1;
        int rect_w = abs(x0 - x1);
        int rect_h = abs(y0 - y1);

        // read pixels
        int pixel_count = rect_w * rect_h;
        std::vector<PixelInfo> pixels(pixel_count);
        CHECK_GL(glReadPixels(start_x, start_y, rect_w, rect_h, GL_RGB_INTEGER, GL_UNSIGNED_INT, pixels.data()));

        // output buffer to image
//        auto* img_pixels = new PixelInfo[w * h];
//        CHECK_GL(glReadPixels(0, 0, w, h, GL_RGB_INTEGER, GL_UNSIGNED_INT, img_pixels));
//        auto* ids = new unsigned int[w * h];
//        for (int i=0; i<w*h; i++)
//            ids[i] = img_pixels[i].obj_id;
//        load_buffer_to_image(ids, w, h);

        // unbind fbo
        CHECK_GL(glReadBuffer(GL_NONE));
        fbo->unbind();

        string result;
        if (scene->select_mode == PICK_MODE::PICK_OBJECT) {
            unordered_set<unsigned int> selected_obj;
            // fetch selected objects' ids
            for (int i = 0; i < pixel_count; i++) {
                if (pixels[i].has_object())
                    selected_obj.insert(pixels[i].obj_id);
            }
            // generate select result
            for (auto id: selected_obj) {
                if (id_table.find(id) != id_table.end())
                    result += id_table[id] + " ";
            }
        }
        else {
            unordered_map<unsigned int, unordered_set<unsigned int>> selected_elem;
            for (int i = 0; i < pixel_count; i++) {
                if (pixels[i].has_object() && pixels[i].has_element()) {
                    if (selected_elem.find(pixels[i].obj_id) != selected_elem.end())
                        selected_elem[pixels[i].obj_id].insert(pixels[i].elem_id);
                    else selected_elem[pixels[i].obj_id] = {pixels[i].elem_id};
                }
            }
            // generate select result
            for (auto &[obj_id, elem_ids] : selected_elem) {
                if (id_table.find(obj_id) != id_table.end()) {
                    for (auto &elem_id : elem_ids)
                        result += id_table[obj_id] + ":" + std::to_string(elem_id - 1) + " ";
                }
            }
        }
        destroy_buffers();

        return result;
    }

    virtual float getDepth(int x, int y) override {
        generate_buffers();
        draw();

        if (!fbo->complete()) return 0;
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->fbo));
        // CHECK_GL(glReadBuffer(GL_DEPTH_ATTACHMENT));

        float depth;
        CHECK_GL(glReadPixels(x, h - y - 1, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth));

        // CHECK_GL(glReadBuffer(GL_NONE));
        fbo->unbind();
        destroy_buffers();

        return depth;
    }

    virtual void focus(const std::string& prim_name) override {
        focus_prim_name = prim_name;
    }
};

}

unique_ptr<IPicker> makeFrameBufferPicker(Scene *scene) {
    return make_unique<FrameBufferPicker>(scene);
}

}