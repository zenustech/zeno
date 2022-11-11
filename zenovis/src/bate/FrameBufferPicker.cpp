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

namespace zenovis {
namespace {

using opengl::FBO;
using opengl::VAO;
using opengl::Buffer;
using opengl ::Texture;
using opengl::Program;

using zeno::vec3i;
using zeno::vec3f;
using zeno::PrimitiveObject;

using std::unique_ptr;
using std::make_unique;
using std::string;
using std::unordered_map;
using std::unordered_set;

static const char *obj_vert_code = R"(
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

static const char *obj_frag_code = R"(
    # version 330
    out uvec3 FragColor;

    uniform uint gObjectIndex = uint(0);

    void main()
    {
       FragColor = uvec3(gObjectIndex, 0, 0);
    }
)";

static void load_buffer_to_image(unsigned int* ids, int w, int h, const std::string& file_name = "output.ppm") {
    unordered_map<unsigned int, vec3i> color_set;
    color_set[0] = {0, 0, 0};

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
            auto id = ids[h * i + j];
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
            os << color[0] << " " << color[1] << " " << color[2] << "\n";
        }
    }
    for (auto [key, value] : obj_count)
        printf("obj id: %u, count: %d, color: (%d, %d, %d)\n", key, value, color_set[key][0],
               color_set[key][1], color_set[key][2]);
    printf("load done.\n");
}

// framebuffer picker referring to https://doc.yonyoucloud.com/doc/wiki/project/modern-opengl-tutorial/tutorial29.html
struct FrameBufferPicker : IPicker {
    Scene* scene;

    unique_ptr<FBO> fbo;
    unique_ptr<Texture> picking_texture;
    unique_ptr<Texture> depth_texture;

    unique_ptr<Buffer> vbo;
    unique_ptr<Buffer> ebo;
    unique_ptr<VAO> vao;

    Program* shader;

    int w, h;
    unordered_map<unsigned int, string> id_table;

    struct PixelInfoWithObj {
        unsigned int obj_id;
        unsigned int no_use1;
        unsigned int no_use2;

        PixelInfoWithObj() {
            obj_id = 0;
            no_use1 = 0;
            no_use2 = 0;
        }
    };

    explicit FrameBufferPicker(Scene* s) : scene(s) {
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

        // generate draw buffer
        vbo = make_unique<Buffer>(GL_ARRAY_BUFFER);
        ebo = make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        vao = make_unique<VAO>();

        // unbind fbo & texture
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
        fbo->unbind();
    }

    ~FrameBufferPicker() {
        if (fbo->fbo) CHECK_GL(glDeleteFramebuffers(1, &fbo->fbo));
        if (picking_texture->tex) CHECK_GL(glDeleteTextures(1, &picking_texture->tex));
        if (depth_texture->tex) CHECK_GL(glDeleteTextures(1, &depth_texture->tex));
    }

    virtual void draw() override {
        // prepare shader
        shader = scene->shaderMan->compile_program(obj_vert_code, obj_frag_code);
        scene->camera->set_program_uniforms(shader);

        // enable framebuffer writing
        CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        // shading primitive objects
        auto prims = scene->objectsMan->pairsShared();
        for (unsigned int id = 0; id < prims.size(); id++) {
            auto it = prims.begin() + id;
            auto prim = dynamic_cast<PrimitiveObject*>(it->second.get());
            if (prim->has_attr("pos")) {
                // set object id
                shader->use();
                CHECK_GL(glUniform1ui(glGetUniformLocation(shader->pro, "gObjectIndex"), id + 1));

                // prepare vertices data
                auto const &pos = prim->attr<zeno::vec3f>("pos");
                auto vertex_count = prim->size();
                std::vector<vec3f> mem(vertex_count);
                for (int i = 0; i < vertex_count; i++)
                    mem[i] = pos[i];
                vao->bind();
                vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
                vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);
                ebo->bind_data(prim->tris.data(), prim->tris.size() * sizeof(prim->tris[0]));

                // draw prim
                CHECK_GL(glDrawElements(GL_TRIANGLES, prim->tris.size() * 3, GL_UNSIGNED_INT, 0));

                // unbind vbo
                vbo->disable_attribute(0);
                vbo->unbind();
                vao->unbind();
                ebo->unbind();

                // store object's name
                id_table[id + 1] = it->first;
            }
        }
        fbo->unbind();
    }

    virtual string getPicked(int x, int y) override {
        if (!fbo->complete()) return "";
        CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->fbo));
        CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0));

        PixelInfoWithObj pixel;
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
//        auto* pixels = new PixelInfoWithObj[w * h];
//        CHECK_GL(glReadPixels(0, 0, w, h, GL_RGB_INTEGER, GL_UNSIGNED_INT, pixels));
//        auto* ids = new unsigned int[w * h];
//        for (int i=0; i<w*h; i++)
//            ids[i] = pixels[i].obj_id;
//        load_buffer_to_image(ids, w, h);

        CHECK_GL(glReadBuffer(GL_NONE));
        fbo->unbind();

        return id_table[pixel.obj_id];
    }

    virtual string getPicked(int x0, int y0, int x1, int y1) override {
        if (!fbo->complete()) return "";
        string result;
        unordered_set<unsigned int> selected_obj;

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
        auto* pixels = new PixelInfoWithObj[pixel_count];
        CHECK_GL(glReadPixels(start_x, start_y, rect_w, rect_h, GL_RGB_INTEGER, GL_UNSIGNED_INT, pixels));

        // fetch selected objects' ids
        for (int i = 0; i < pixel_count; i++) {
            selected_obj.insert(pixels[i].obj_id);
        }

        // generate select result
        for (auto id : selected_obj) {
            if (id_table.find(id) != id_table.end())
                result += id_table[id] + " ";
        }
        return result;
    }
};

}

unique_ptr<IPicker> makeFrameBufferPicker(Scene *scene) {
    return make_unique<FrameBufferPicker>(scene);
}

}