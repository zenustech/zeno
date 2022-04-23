#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/IGraphic.h>
#include <zenovis/Scene.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/makeGraphic.h>
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
    uniform float mCameraRadius;

    attribute vec3 vPosition;
    attribute vec3 vColor;

    varying vec3 pos;

    void main() {
        pos = vPosition * max(1, mCameraRadius / 1000);

        gl_Position = mVP * vec4(pos, 1.0);
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
    uniform float mCameraRadius;
    uniform vec3 mCameraCenter;
    uniform float mGridScale;
    uniform float mGridBlend;

    varying vec3 pos;

    float calc_color(vec2 pos, float scale) {
        float gridSize = 5.0;
        float gridWidth = 2.0;

        float wx = pos.x / scale;
        float wz = pos.y / scale;

        float x0 = abs(fract(wx / gridSize - 0.5) - 0.5) / fwidth(wx) * gridSize / gridWidth;
        float z0 = abs(fract(wz / gridSize - 0.5) - 0.5) / fwidth(wz) * gridSize / gridWidth;

        float x1 = abs(fract(wx - 0.5) - 0.5) / fwidth(wx);
        float z1 = abs(fract(wz - 0.5) - 0.5) / fwidth(wz);

        float v0 = clamp(min(x0, z0), 0.0, 1.0);
        float v1 = clamp(min(x1, z1), 0.0, 1.0);
        float alpha = 1.0 - min(v0, v1);
        return alpha;
    }
    float ratio_clamp(float value, float lower_bound, float upper_bound) {
        float ratio = (value - lower_bound) / (upper_bound - lower_bound);
        return clamp(ratio, 0.0, 1.0);
    }
    void main() {
        float alpha = calc_color(pos.xz, mGridScale);
        if (mGridBlend > 0) {
            float alpha2 = calc_color(pos.xz, mGridScale * 5);
            alpha = mix(alpha, alpha2, mGridBlend);
        }
        vec3 cam_on_grid = vec3(mCameraCenter.x, 0, mCameraCenter.z);
        alpha *= 1 - ratio_clamp(distance(cam_on_grid, pos), mCameraRadius, mCameraRadius * 2);
        gl_FragColor = vec4(0.5, 0.5, 0.5, alpha);
    }
)";

struct GraphicGrid : IGraphic {
    Scene *scene;

    std::unique_ptr<Buffer> vbo;
    size_t vertex_count;

    Program *prog;

    explicit GraphicGrid(Scene *scene_) : scene(scene_) {
        vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        std::vector<zeno::vec3f> mem;
        float bound = 1000;
        mem.push_back(vec3f(-bound, 0, -bound));
        mem.push_back(vec3f(-bound, 0, bound));
        mem.push_back(vec3f(bound, 0, -bound));
        mem.push_back(vec3f(-bound, 0, bound));
        mem.push_back(vec3f(bound, 0, bound));
        mem.push_back(vec3f(bound, 0, -bound));
        vertex_count = mem.size();

        vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

        prog = scene->shaderMan->compile_program(vert_code, frag_code);
    }
    virtual void drawShadow(Light *light) override {
    }

    virtual void draw(bool reflect, bool depthPass) override {
        vbo->bind();
        vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

        prog->use();
        scene->camera->set_program_uniforms(prog);
        CHECK_GL(glDrawArrays(GL_TRIANGLES, 0, vertex_count));

        vbo->disable_attribute(0);
        vbo->disable_attribute(1);
        vbo->unbind();
    }
};

} // namespace

std::unique_ptr<IGraphic> makeGraphicGrid(Scene *scene) {
    return std::make_unique<GraphicGrid>(scene);
}

} // namespace zenovis
