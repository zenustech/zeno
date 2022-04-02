#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <zeno/utils/vec.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>

using std::string;
using zeno::vec3f;

static string vert_code = R"(
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
        pos = vPosition;
        gl_Position = vec4(vPosition, 1.0);
    }
)";

static string frag_code = R"(
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
    uniform float mNX;
    uniform float mNY;

    varying vec3 pos;

    void main() {
      float ratio = mNX / mNY / 2.35;
      if (ratio < 1) {
        if (pos.y < -ratio || pos.y > ratio) {
          gl_FragColor = vec4(0, 0, 0, 1);
        } else {
          gl_FragColor = vec4(0, 0, 0, 0);
        }
      } else {
        if (pos.x < -(1.0 / ratio) || pos.x > (1.0 / ratio)) {
          gl_FragColor = vec4(0, 0, 0, 1);
        } else {
          gl_FragColor = vec4(0, 0, 0, 0);
        }
      }
    }
)";


namespace zenvis {

struct GraphicSafetyFrame : IGraphic {
  std::unique_ptr<Buffer> vbo;
  size_t vertex_count;

  Program *prog;

  GraphicSafetyFrame() {
    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem;
    float bound = 1;
    mem.push_back(vec3f(-bound, -bound, 0));
    mem.push_back(vec3f(-bound, bound, 0));
    mem.push_back(vec3f(bound, -bound, 0));
    mem.push_back(vec3f(-bound, bound, 0));
    mem.push_back(vec3f(bound, bound, 0));
    mem.push_back(vec3f(bound, -bound, 0));
    vertex_count = mem.size();

    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    prog = compile_program(vert_code, frag_code);
  }
  virtual void drawShadow(Light *light) override
  {
    
  }

  virtual void draw(bool reflect) override {
    vbo->bind();
    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 3, GL_FLOAT, 3);

    prog->use();
    set_program_uniforms(prog);
    CHECK_GL(glDrawArrays(GL_TRIANGLES, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->unbind();
  }
};
std::unique_ptr<IGraphic> makeGraphicSafetyFrame() {
  return std::make_unique<GraphicSafetyFrame>();
}

}
