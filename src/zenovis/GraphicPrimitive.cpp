#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>

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

  //std::vector<std::unique_ptr<Texture>> textures;

  GraphicPrimitive(std::shared_ptr<zeno::PrimitiveObject> prim) {
    if (!prim->has_attr("pos")) {
        auto &pos = prim->add_attr<zeno::vec3f>("pos");
        for (size_t i = 0; i < pos.size(); i++) {
            pos[i] = zeno::vec3f(i * (1.0f / (pos.size() - 1)), 0, 0);
        }
    }
    if (!prim->has_attr("clr")) {
        auto &clr = prim->add_attr<zeno::vec3f>("clr");
        for (size_t i = 0; i < clr.size(); i++) {
            clr[i] = zeno::vec3f(1.0f);
        }
    }
    if (!prim->has_attr("nrm")) {
        auto &nrm = prim->add_attr<zeno::vec3f>("nrm");

        if (prim->has_attr("rad")) {
            if (prim->has_attr("opa")) {
                auto &rad = prim->attr<float>("rad");
                auto &opa = prim->attr<float>("opa");
                for (size_t i = 0; i < nrm.size(); i++) {
                    nrm[i] = zeno::vec3f(rad[i], opa[i], 0.0f);
                }
            } else {
                auto &rad = prim->attr<float>("rad");
                for (size_t i = 0; i < nrm.size(); i++) {
                    nrm[i] = zeno::vec3f(rad[i], 0.0f, 0.0f);
                }
            }
        } else if (prim->tris.size()) {
            for (size_t i = 0; i < nrm.size(); i++) {
                nrm[i] = zeno::vec3f(1 / zeno::sqrt(3.0f));
            }
        } else {
            for (size_t i = 0; i < nrm.size(); i++) {
                nrm[i] = zeno::vec3f(1.5f, 0.0f, 0.0f);
            }
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
        points_prog = get_points_program();
    }

    lines_count = prim->lines.size();
    if (lines_count) {
        lines_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        lines_ebo->bind_data(prim->lines.data(), lines_count * sizeof(prim->lines[0]));
        lines_prog = get_lines_program();
    }

    tris_count = prim->tris.size();
    if (tris_count) {
        tris_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        tris_ebo->bind_data(prim->tris.data(), tris_count * sizeof(prim->tris[0]));
        tris_prog = get_tris_program();
    }

    draw_all_points = !points_count && !lines_count && !tris_count;
    if (draw_all_points) {
        points_prog = get_points_program();
    }

    //load_textures(path);
  }

  virtual void draw() override {
    /*for (int id = 0; id < textures.size(); id++) {
        textures[id]->bind_to(id);
    }*/

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
        tris_prog->set_uniform("mRenderWireframe", false);
        tris_ebo->bind();
        CHECK_GL(glDrawElements(GL_TRIANGLES, /*count=*/tris_count * 3,
              GL_UNSIGNED_INT, /*first=*/0));
        if (render_wireframe) {
          glEnable(GL_POLYGON_OFFSET_LINE);
          glPolygonOffset(-1, -1);
          tris_prog->set_uniform("mRenderWireframe", true);
          glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
          CHECK_GL(glDrawElements(GL_TRIANGLES, tris_count * 3, GL_UNSIGNED_INT, 0));
          glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
          glDisable(GL_POLYGON_OFFSET_LINE);
        }
        tris_ebo->unbind();
    }

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->disable_attribute(2);
    vbo->unbind();
  }

  /*void load_textures(std::string const &path) {
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
  }*/

  Program *get_points_program() {
    auto vert = R"(#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform float mPointScale;

attribute vec3 vPosition;
attribute vec3 vColor;
attribute vec3 vNormal;

varying vec3 position;
varying vec3 color;
varying float radius;
varying float opacity;
void main()
{
  position = vPosition;
  color = vColor;
  radius = vNormal.x;
  opacity = vNormal.y;

  vec3 posEye = vec3(mView * vec4(position, 1.0));
  float dist = length(posEye);
  if (radius != 0)
    gl_PointSize = max(1, radius * mPointScale / dist);
  else
    gl_PointSize = 1.5;
  gl_Position = mVP * vec4(position, 1.0);
}
)";
    auto frag = R"(#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;

varying vec3 position;
varying vec3 color;
varying float radius;
varying float opacity;
void main()
{
  const vec3 lightDir = vec3(0.577, 0.577, 0.577);
  vec2 coor = gl_PointCoord * 2 - 1;
  float len2 = dot(coor, coor);
  if (len2 > 1 && radius != 0)
    discard;
  vec3 oColor;
  if (radius != 0)
  {
    vec3 N;
    N.xy = gl_PointCoord*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N) * 0.6 + 0.4);
    oColor = color * diffuse;
  }
  else
    oColor = color;
  gl_FragColor = vec4(oColor, 1.0 - opacity);
}
)";

    return compile_program(vert, frag);
  }

  Program *get_lines_program() {
      auto vert = R"(#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

attribute vec3 vPosition;
attribute vec3 vColor;

varying vec3 position;
varying vec3 color;

void main()
{
  position = vPosition;
  color = vColor;

  gl_Position = mVP * vec4(position, 1.0);
}
)";
      auto frag = R"(#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

varying vec3 position;
varying vec3 color;

void main()
{
  gl_FragColor = vec4(color, 1.0);
}
)";

    return compile_program(vert, frag);
  }

  Program *get_tris_program() {
      auto vert = R"(#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

attribute vec3 vPosition;
attribute vec3 vColor;
attribute vec3 vNormal;

varying vec3 position;
varying vec3 iColor;
varying vec3 iNormal;

void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;

  gl_Position = mVP * vec4(position, 1.0);
}
)";
      auto frag = R"(
#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform bool mSmoothShading;
uniform bool mRenderWireframe;

varying vec3 position;
varying vec3 iColor;
varying vec3 iNormal;

vec3 pbr(vec3 albedo, float roughness, float metallic, float specular,
    vec3 nrm, vec3 idir, vec3 odir) {

  vec3 hdir = normalize(idir + odir);
  float NoH = max(0., dot(hdir, nrm));
  float NoL = max(0., dot(idir, nrm));
  float NoV = max(0., dot(odir, nrm));
  float VoH = clamp(dot(odir, hdir), 0., 1.);
  float LoH = clamp(dot(idir, hdir), 0., 1.);

  vec3 f0 = metallic * albedo + (1. - metallic) * 0.16 * specular * specular;
  vec3 fdf = f0 + (1. - f0) * pow(1. - VoH, 5.);

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
    color += vec3(0.45, 0.47, 0.5) * pbr(albedo, 0.19, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(-4., -2., 1., 0.)).xyz);
    color += vec3(0.3, 0.23, 0.18) * pbr(albedo, 0.14, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(3., -5., 2., 0.)).xyz);
    color += vec3(0.15, 0.2, 0.22) * pbr(albedo, 0.23, 0.0, 1.0, normal, light_dir, view_dir);

    color *= 1.2;
    //color = pow(clamp(color, 0., 1.), vec3(1./2.2));
    return color;
}

vec3 calcRayDir(vec3 pos)
{
  vec4 vpos = mVP * vec4(pos, 1);
  vec2 uv = vpos.xy / vpos.w;
  vec4 ro = mInvVP * vec4(uv, -1, 1);
  vec4 re = mInvVP * vec4(uv, +1, 1);
  vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);
  return rd;
}

void main()
{
  if (mRenderWireframe) {
    gl_FragColor = vec4(0.89, 0.57, 0.15, 1.0);
    return;
  }
  vec3 normal;
  if (mSmoothShading) {
    normal = normalize(iNormal);
  } else {
    normal = normalize(cross(dFdx(position), dFdy(position)));
  }
  vec3 viewdir = -calcRayDir(position);

  vec3 albedo = iColor;
  vec3 color = studioShading(albedo, viewdir, normal);
  gl_FragColor = vec4(color, 1.0);
}
)";

    return compile_program(vert, frag);
  }
};

std::unique_ptr<IGraphic> makeGraphicPrimitive(std::shared_ptr<zeno::IObject> obj) {
  if (auto prim = std::dynamic_pointer_cast<zeno::PrimitiveObject>(obj))
      return std::make_unique<GraphicPrimitive>(std::move(prim));
  return nullptr;
}

}
