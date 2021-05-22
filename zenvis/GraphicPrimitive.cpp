#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <zen/vec.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>
#include <zen/PrimitiveObject.h>

namespace zenvis {

struct GraphicPrimitive : IGraphic {
  std::unique_ptr<Buffer> vbo;
  size_t vertex_count;
  bool draw_all_points;

  Program *points_prog;
  std::unique_ptr<Buffer> points_ebo;
  size_t points_count;

  Program *lines_prog;
  std::unique_ptr<Buffer> lines_ebo;
  size_t lines_count;

  Program *tris_prog;
  std::unique_ptr<Buffer> tris_ebo;
  size_t tris_count;

  GraphicPrimitive
    ( zenbase::PrimitiveObject *prim
    , std::string const &path
    ) {
    auto const &pos = prim->add_attr<zen::vec3f>("pos");
    auto const &clr = prim->add_attr<zen::vec3f>("clr");
    auto const &nrm = prim->add_attr<zen::vec3f>("nrm");
    vertex_count = prim->size();

    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zen::vec3f> mem(vertex_count * 3);
    for (int i = 0; i < vertex_count; i++) {
        mem[3 * i + 0] = pos[i];
        mem[3 * i + 1] = clr[i];
        mem[3 * i + 2] = nrm[i];
    }
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    points_count = prim->points.size();
    if (points_count) {
        points_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        points_ebo->bind_data(prim->points.data(), points_count * sizeof(prim->points[0]));
        points_prog = get_points_program(path);
    }

    lines_count = prim->lines.size();
    if (lines_count) {
        lines_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        lines_ebo->bind_data(prim->lines.data(), lines_count * sizeof(prim->lines[0]));
        lines_prog = get_lines_program(path);
    }

    tris_count = prim->tris.size();
    if (tris_count) {
        tris_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        tris_ebo->bind_data(prim->tris.data(), tris_count * sizeof(prim->tris[0]));
        tris_prog = get_tris_program(path);
    }

    draw_all_points = !points_count && !lines_count && !tris_count;
    if (draw_all_points) {
        points_prog = get_points_program(path);
    }
  }

  virtual void draw() override {
    vbo->bind();
    vbo->attribute(/*index=*/0,
        /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 9,
        GL_FLOAT, /*count=*/3);
    vbo->attribute(/*index=*/1,
        /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 9,
        GL_FLOAT, /*count=*/3);
    vbo->attribute(/*index=*/2,
        /*offset=*/sizeof(float) * 6, /*stride=*/sizeof(float) * 9,
        GL_FLOAT, /*count=*/3);

    if (draw_all_points) {
        //printf("ALLPOINTS\n");
        points_prog->use();
        set_program_uniforms(points_prog);
        CHECK_GL(glDrawArrays(GL_POINTS, /*first=*/0, /*count=*/vertex_count));
    }

    if (points_count) {
        //printf("POINTS\n");
        points_prog->use();
        set_program_uniforms(points_prog);
        points_ebo->bind();
        CHECK_GL(glDrawElements(GL_POINTS, /*count=*/points_count * 1,
              GL_UNSIGNED_INT, /*first=*/0));
        points_ebo->unbind();
    }

    if (lines_count) {
        //printf("LINES\n");
        lines_prog->use();
        set_program_uniforms(lines_prog);
        lines_ebo->bind();
        CHECK_GL(glDrawElements(GL_LINES, /*count=*/lines_count * 2,
              GL_UNSIGNED_INT, /*first=*/0));
        lines_ebo->unbind();
    }

    if (tris_count) {
        //printf("TRIS\n");
        tris_prog->use();
        set_program_uniforms(tris_prog);
        tris_ebo->bind();
        CHECK_GL(glDrawElements(GL_TRIANGLES, /*count=*/tris_count * 3,
              GL_UNSIGNED_INT, /*first=*/0));
        tris_ebo->unbind();
    }

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
  }

  Program *get_points_program(std::string const &path) {
    auto vert = hg::file_get_content(path + ".points.vert");
    auto frag = hg::file_get_content(path + ".points.frag");

    if (vert.size() == 0) {
      vert =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"\n"
"attribute vec3 vPosition;\n"
"attribute vec3 vColor;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 color;\n"
"\n"
"void main()\n"
"{\n"
"  position = vPosition;\n"
"  color = vColor;\n"
"\n"
"  gl_Position = mVP * vec4(position, 1.0);\n"
"  gl_PointSize = 5.0;\n"
"}\n";
    }
    if (frag.size() == 0) {
      frag =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 color;\n"
"\n"
"void main()\n"
"{\n"
"  if (length(gl_PointCoord - vec2(0.5)) > 0.5)\n"
"    discard;\n"
"  gl_FragColor = vec4(color, 1.0);\n"
"}\n";
    }

    return compile_program(vert, frag);
  }

  Program *get_lines_program(std::string const &path) {
    auto vert = hg::file_get_content(path + ".lines.vert");
    auto frag = hg::file_get_content(path + ".lines.frag");

    if (vert.size() == 0) {
      vert =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"uniform mat4 mInvView;\n"
"uniform mat4 mInvProj;\n"
"\n"
"attribute vec3 vPosition;\n"
"attribute vec3 vColor;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 color;\n"
"\n"
"void main()\n"
"{\n"
"  position = vPosition;\n"
"  color = vColor;\n"
"\n"
"  gl_Position = mVP * vec4(position, 1.0);\n"
"}\n";
    }
    if (frag.size() == 0) {
      frag =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"uniform mat4 mInvView;\n"
"uniform mat4 mInvProj;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 color;\n"
"\n"
"void main()\n"
"{\n"
"  gl_FragColor = vec4(color, 1.0);\n"
"}\n";
    }

    return compile_program(vert, frag);
  }

  Program *get_tris_program(std::string const &path) {
    auto vert = hg::file_get_content(path + ".tris.vert");
    auto frag = hg::file_get_content(path + ".tris.frag");

    if (vert.size() == 0) {
      vert =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"uniform mat4 mInvView;\n"
"uniform mat4 mInvProj;\n"
"\n"
"attribute vec3 vPosition;\n"
"attribute vec3 vColor;\n"
"attribute vec3 vNormal;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 color;\n"
"varying vec3 iNormal;\n"
"\n"
"void main()\n"
"{\n"
"  position = vPosition;\n"
"  color = vColor;\n"
"  iNormal = vNormal;\n"
"\n"
"  gl_Position = mVP * vec4(position, 1.0);\n"
"}\n";
    }
    if (frag.size() == 0) {
      frag =
"#version 120\n"
"\n"
"uniform mat4 mVP;\n"
"uniform mat4 mInvVP;\n"
"uniform mat4 mView;\n"
"uniform mat4 mProj;\n"
"uniform mat4 mInvView;\n"
"uniform mat4 mInvProj;\n"
"\n"
"varying vec3 position;\n"
"varying vec3 color;\n"
"varying vec3 iNormal;\n"
"\n"
"void main()\n"
"{\n"
"  vec3 normal = normalize(iNormal);\n"
"  gl_FragColor = vec4(normal * 0.5 + 0.5, 1.0);\n"
"}\n";
    }

    return compile_program(vert, frag);
  }
};

std::unique_ptr<IGraphic> makeGraphicPrimitive
    ( zenbase::PrimitiveObject *prim
    , std::string const &path
    ) {
  return std::make_unique<GraphicPrimitive>(prim, path);
}

}
