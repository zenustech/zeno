#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <zeno/vec.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>
#include <zeno/PrimitiveObject.h>

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

  std::vector<std::unique_ptr<Texture>> textures;

  GraphicPrimitive
    ( zeno::PrimitiveObject *prim
    , std::string const &path
    ) {
    if (!prim->has_attr("pos")) {
        auto &pos = prim->add_attr<zeno::vec3f>("pos");
        for (size_t i = 0; i < pos.size(); i++) {
            pos[i] = zeno::vec3f(i * (1.0f / (pos.size() - 1)), 0, 0);
        }
    }
    if (!prim->has_attr("clr")) {
        auto &clr = prim->add_attr<zeno::vec3f>("clr");
        for (size_t i = 0; i < clr.size(); i++) {
            clr[i] = zeno::vec3f(0.8);
        }
    }
    if (!prim->has_attr("nrm")) {
        auto &nrm = prim->add_attr<zeno::vec3f>("nrm");

        for (size_t i = 0; i < nrm.size(); i++) {
            nrm[i] = zeno::vec3f(1 / zeno::sqrt(3.0f));
        }
    }
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    vertex_count = prim->size();

    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(vertex_count * 3);
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

    load_textures(path);
  }

  virtual void draw() override {
    for (int id = 0; id < textures.size(); id++) {
        textures[id]->bind_to(id);
    }

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
    vbo->disable_attribute(2);
    vbo->unbind();
  }

  void load_textures(std::string const &path) {
      for (int id = 0; id < 8; id++) {
          std::ostringstream ss;
          if (!(ss << path << "." << id << ".png"))
              break;
          auto texpath = ss.str();
          if (!hg::file_exists(texpath))
              continue;
          auto tex = std::make_unique<Texture>();
          tex->load(texpath.c_str());
          textures.push_back(std::move(tex));
      }
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
"varying vec3 iColor;\n"
"varying vec3 iNormal;\n"
"\n"
"void main()\n"
"{\n"
"  position = vPosition;\n"
"  iColor = vColor;\n"
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
"varying vec3 iColor;\n"
"varying vec3 iNormal;\n"
"\n"
"struct Light {\n"
"  vec3 dir;\n"
"  vec3 color;\n"
"};\n"
"\n"
"struct Material {\n"
"  vec3 albedo;\n"
"  float roughness;\n"
"  float metallic;\n"
"  float specular;\n"
"};\n"
"\n"
"vec3 pbr(Material material, vec3 nrm, vec3 idir, vec3 odir) {\n"
"  float roughness = material.roughness;\n"
"  float metallic = material.metallic;\n"
"  float specular = material.specular;\n"
"  vec3 albedo = material.albedo;\n"
"\n"
"  vec3 hdir = normalize(idir + odir);\n"
"  float NoH = max(0, dot(hdir, nrm));\n"
"  float NoL = max(0, dot(idir, nrm));\n"
"  float NoV = max(0, dot(odir, nrm));\n"
"  float VoH = clamp(dot(odir, hdir), 0, 1);\n"
"  float LoH = clamp(dot(idir, hdir), 0, 1);\n"
"\n"
"  vec3 f0 = metallic * albedo + (1 - metallic) * 0.16 * specular * specular;\n"
"  vec3 fdf = f0 + (1 - f0) * pow(1 - VoH, 5);\n"
"\n"
"  float k = (roughness + 1) * (roughness + 1) / 8;\n"
"  float vdf = 0.25 / ((NoV * k + 1 - k) * (NoL * k + 1 - k));\n"
"\n"
"  float alpha2 = max(0, roughness * roughness);\n"
"  float denom = 1 - NoH * NoH * (1 - alpha2);\n"
"  float ndf = alpha2 / (denom * denom);\n"
"\n"
"  vec3 brdf = fdf * vdf * ndf * f0 + (1 - f0) * albedo;\n"
"  return brdf * NoL;\n"
"}\n"
"\n"
"vec3 calcRayDir(vec3 pos)\n"
"{\n"
"  vec4 vpos = mVP * vec4(pos, 1);\n"
"  vec2 uv = vpos.xy / vpos.w;\n"
"  vec4 ro = mInvVP * vec4(uv, -1, 1);\n"
"  vec4 re = mInvVP * vec4(uv, +1, 1);\n"
"  vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);\n"
"  return rd;\n"
"}\n"
"\n"
"void main()\n"
"{\n"
"  vec3 normal = normalize(iNormal);\n"
"  vec3 viewdir = -calcRayDir(position);\n"
"\n"
"  Material material;\n"
//"  material.albedo = vec3(0.8);\n"
"  material.albedo = iColor;\n"
"  material.roughness = 0.4;\n"
"  material.metallic = 0.0;\n"
"  material.specular = 0.5;\n"
"\n"
"  Light light;\n"
"  light.dir = normalize((mVP * vec4(-1, -2, 5, 0)).xyz);\n"
"  light.dir = faceforward(light.dir, -light.dir, normal);\n"
"  light.color = vec3(1, 1, 1);\n"
"\n"
"  vec3 strength = pbr(material, normal, light.dir, viewdir);\n"
"  vec3 color = light.color * strength;\n"
"  gl_FragColor = vec4(color, 1.0);\n"
"}\n";
    }

    return compile_program(vert, frag);
  }
};

std::unique_ptr<IGraphic> makeGraphicPrimitive
    ( zeno::PrimitiveObject *prim
    , std::string const &path
    ) {
  return std::make_unique<GraphicPrimitive>(prim, path);
}

}
