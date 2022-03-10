#include "glad/glad.h"
#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <memory>
#include <vector>
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>
namespace zenvis {
extern unsigned int getGlobalEnvMap();
struct drawObject
{
    std::unique_ptr<Buffer> vbo;
    std::unique_ptr<Buffer> ebo;
    size_t count=0;
    Program *prog;
};
void parsePointsDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj)
{
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &uv = prim->attr<zeno::vec3f>("uv");
    obj.count = prim->size();

    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 4);
    for (int i = 0; i < obj.count; i++)
    {
      mem[4 * i + 0] = pos[i];
      mem[4 * i + 1] = clr[i];
      mem[4 * i + 2] = nrm[i];
      mem[4 * i + 3] = uv[i];
    }
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    size_t points_count = prim->points.size();
    if (points_count) {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(prim->points.data(), points_count * sizeof(prim->points[0]));
    }

}
void parseLinesDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj)
{
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &lines = prim->lines;
    bool has_uv = lines.has_attr("uv0")&&lines.has_attr("uv1");
    obj.count = prim->lines.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 2 * 4);
    std::vector<zeno::vec2i> linesdata(obj.count);
    #pragma omp parallel for
    for(int i=0; i<obj.count;i++)
    {
        mem[8 * i + 0] = pos[lines[i][0]];
        mem[8 * i + 1] = clr[lines[i][0]];
        mem[8 * i + 2] = nrm[lines[i][0]];
        mem[8 * i + 3] = has_uv? lines.attr<zeno::vec3f>("uv0")[i]:zeno::vec3f(0,0,0);
        mem[8 * i + 4] = pos[lines[i][1]];
        mem[8 * i + 5] = clr[lines[i][1]];
        mem[8 * i + 6] = nrm[lines[i][1]];
        mem[8 * i + 7] = has_uv? lines.attr<zeno::vec3f>("uv1")[i]:zeno::vec3f(0,0,0);
        linesdata[i] = zeno::vec2i(i*2, i*2+1);

    }
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    if(obj.count)
    {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(linesdata[0]), obj.count * sizeof(linesdata[0]));
    }
}

void computeTrianglesTangent(zeno::PrimitiveObject *prim, drawObject &obj)
{
    const auto &tris = prim->tris;
    const auto &pos = prim->attr<zeno::vec3f>("pos");
    auto &tang = prim->add_attr<zeno::vec3f>("tang");
#pragma omp parallel for
    for (size_t i = 0; i < obj.count; ++i)
    {
        const auto &pos0 = pos[tris[i][0]];
        const auto &pos1 = pos[tris[i][1]];
        const auto &pos2 = pos[tris[i][2]];
        const auto &uv0 = tris.has_attr("uv0") ? tris.attr<zeno::vec3f>("uv0")[i] : zeno::vec3f(0.0f, 0.0f, 0.0f);
        const auto &uv1 = tris.has_attr("uv1") ? tris.attr<zeno::vec3f>("uv1")[i] : zeno::vec3f(0.0f, 0.0f, 0.0f);
        const auto &uv2 = tris.has_attr("uv2") ? tris.attr<zeno::vec3f>("uv2")[i] : zeno::vec3f(0.0f, 0.0f, 0.0f);

        auto edge0 = pos1 - pos0;
        auto edge1 = pos2 - pos0;
        auto deltaUV0 = uv1 - uv0;
        auto deltaUV1 = uv2 - uv0;

        auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[1] * deltaUV0[0]);

        zeno::vec3f tangent;
        tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
        tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
        tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
        tang[i] = zeno::normalize(tangent);
    }
}

void parseTrianglesDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj)
{
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tris = prim->tris;
    bool has_uv = tris.has_attr("uv0")&&tris.has_attr("uv1")&&tris.has_attr("uv2");
    const auto &tang = prim->attr<zeno::vec3f>("tang");
    obj.count = prim->tris.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 3 * 5);
    std::vector<zeno::vec3i> trisdata(obj.count);
    #pragma omp parallel for
    for(int i=0; i<obj.count;i++)
    {
        mem[15 * i + 0] = pos[tris[i][0]];
        mem[15 * i + 1] = clr[tris[i][0]];
        mem[15 * i + 2] = nrm[tris[i][0]];
        mem[15 * i + 3] = has_uv ? tris.attr<zeno::vec3f>("uv0")[i] : zeno::vec3f(0, 0, 0);
        mem[15 * i + 4] = tang[i];
        mem[15 * i + 5] = pos[tris[i][1]];
        mem[15 * i + 6] = clr[tris[i][1]];
        mem[15 * i + 7] = nrm[tris[i][1]];
        mem[15 * i + 8] = has_uv ? tris.attr<zeno::vec3f>("uv1")[i] : zeno::vec3f(0, 0, 0);
        mem[15 * i + 9] = tang[i];
        mem[15 * i + 10] = pos[tris[i][2]];
        mem[15 * i + 11] = clr[tris[i][2]];
        mem[15 * i + 12] = nrm[tris[i][2]];
        mem[15 * i + 13] = has_uv ? tris.attr<zeno::vec3f>("uv2")[i] : zeno::vec3f(0, 0, 0);
        mem[15 * i + 14] = tang[i];

        trisdata[i] = zeno::vec3i(i*3, i*3+1, i*3+2);

    }
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    if(obj.count)
    {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(trisdata[0]), obj.count * sizeof(trisdata[0]));
    }
}
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

  drawObject lineObj;
  drawObject triObj;
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
            clr[i] = zeno::vec3f(1.0f);
        }
    }
    if(prim->tris.size())
    {
        zeno::getNormal(prim);
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
            // for (size_t i = 0; i < nrm.size(); i++) {
            //     nrm[i] = zeno::vec3f(1 / zeno::sqrt(3.0f));
            // }
            

        } else {
            for (size_t i = 0; i < nrm.size(); i++) {
                nrm[i] = zeno::vec3f(1.5f, 0.0f, 0.0f);
            }
        }
    }
    if (!prim->has_attr("uv"))
    {
        auto &uv = prim->add_attr<zeno::vec3f>("uv");
        for (size_t i = 0; i < uv.size(); i++) {
            uv[i] = zeno::vec3f(0.0f);
        }
    }
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &uv = prim->attr<zeno::vec3f>("uv");
    vertex_count = prim->size();

    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(vertex_count * 4);
    for (int i = 0; i < vertex_count; i++)
    {
      mem[4 * i + 0] = pos[i];
      mem[4 * i + 1] = clr[i];
      mem[4 * i + 2] = nrm[i];
      mem[4 * i + 3] = uv[i];
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
        // lines_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        // lines_ebo->bind_data(prim->lines.data(), lines_count * sizeof(prim->lines[0]));
        // lines_prog = get_lines_program(path);
        parseLinesDrawBuffer(prim, lineObj);
        lineObj.prog = get_lines_program(path);
    }

    tris_count = prim->tris.size();
    if (tris_count) {
        // tris_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        // tris_ebo->bind_data(prim->tris.data(), tris_count * sizeof(prim->tris[0]));
        // tris_prog = get_tris_program(path, prim->mtl);
        // if (!tris_prog)
        //     tris_prog = get_tris_program(path, nullptr);
        computeTrianglesTangent(prim, triObj);
        parseTrianglesDrawBuffer(prim, triObj);
        triObj.prog = get_tris_program(path, prim->mtl);
        if(!triObj.prog)
            triObj.prog = get_tris_program(path,nullptr);
    }

    draw_all_points = !points_count && !lines_count && !tris_count;
    if (draw_all_points) {
        points_prog = get_points_program(path);
    }

    if ((prim->mtl != nullptr) && !prim->mtl->tex2Ds.empty())
    {
      load_texture2Ds(prim->mtl->tex2Ds);
    }
    //load_textures(path);
  }

  virtual void draw() override {
    int id = 0;
    for (id = 0; id < textures.size(); id++) {
        textures[id]->bind_to(id);
    }

    vbo->bind();
    vbo->attribute(/*index=*/0,
        /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 12,
        GL_FLOAT, /*count=*/3);
    vbo->attribute(/*index=*/1,
        /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 12,
        GL_FLOAT, /*count=*/3);
    vbo->attribute(/*index=*/2,
        /*offset=*/sizeof(float) * 6, /*stride=*/sizeof(float) * 12,
        GL_FLOAT, /*count=*/3);
    vbo->attribute(/*index=*/3,
        /*offset=*/sizeof(float) * 9, /*stride=*/sizeof(float) * 12,
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
    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->disable_attribute(2);
    vbo->disable_attribute(3);
    vbo->unbind();


    if (lines_count) {
        //printf("LINES\n");
        lineObj.vbo->bind();
        lineObj.vbo->attribute(/*index=*/0,
        /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 12,
        GL_FLOAT, /*count=*/3);
        lineObj.vbo->attribute(/*index=*/1,
        /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 12,
        GL_FLOAT, /*count=*/3);
        lineObj.vbo->attribute(/*index=*/2,
        /*offset=*/sizeof(float) * 6, /*stride=*/sizeof(float) * 12,
        GL_FLOAT, /*count=*/3);
        lineObj.vbo->attribute(/*index=*/3,
        /*offset=*/sizeof(float) * 9, /*stride=*/sizeof(float) * 12,
        GL_FLOAT, /*count=*/3);
        lineObj.prog->use();
        set_program_uniforms(lineObj.prog);
        lineObj.ebo->bind();
        CHECK_GL(glDrawElements(GL_LINES, /*count=*/lineObj.count * 2,
              GL_UNSIGNED_INT, /*first=*/0));
        lineObj.ebo->unbind();
        lineObj.vbo->unbind();
    }

    if (tris_count) {
        //printf("TRIS\n");
        triObj.vbo->bind();
        triObj.vbo->attribute(/*index=*/0,
        /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 15,
        GL_FLOAT, /*count=*/3);
        triObj.vbo->attribute(/*index=*/1,
        /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 15,
        GL_FLOAT, /*count=*/3);
        triObj.vbo->attribute(/*index=*/2,
        /*offset=*/sizeof(float) * 6, /*stride=*/sizeof(float) * 15,
        GL_FLOAT, /*count=*/3);
        triObj.vbo->attribute(/*index=*/3,
        /*offset=*/sizeof(float) * 9, /*stride=*/sizeof(float) * 15,
        GL_FLOAT, /*count=*/3);
        triObj.vbo->attribute(/*index=*/4,
        /*offset=*/sizeof(float) * 9, /*stride=*/sizeof(float) * 15,
        GL_FLOAT, /*count=*/3);
        triObj.prog->use();
        set_program_uniforms(triObj.prog);
        triObj.prog->set_uniform("mRenderWireframe", false);
        triObj.prog->set_uniformi("skybox",id);
        CHECK_GL(glActiveTexture(GL_TEXTURE0+id));
        if (auto envmap = getGlobalEnvMap(); envmap != (unsigned int)-1)
            CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, envmap));

        triObj.ebo->bind();
        CHECK_GL(glDrawElements(GL_TRIANGLES, /*count=*/triObj.count * 3,
              GL_UNSIGNED_INT, /*first=*/0));
        if (render_wireframe) {
          glEnable(GL_POLYGON_OFFSET_LINE);
          glPolygonOffset(-1, -1);
          triObj.prog->set_uniform("mRenderWireframe", true);
          glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
          CHECK_GL(glDrawElements(GL_TRIANGLES, triObj.count * 3, GL_UNSIGNED_INT, 0));
          glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
          glDisable(GL_POLYGON_OFFSET_LINE);
        }
        triObj.ebo->unbind();
        triObj.vbo->unbind();
    }

    
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

  void load_texture2Ds(const std::vector<std::shared_ptr<zeno::Texture2DObject>> &tex2Ds)
  {
    for (const auto &tex2D : tex2Ds)
    {
      auto tex = std::make_unique<Texture>();
      tex->load(tex2D->path.c_str());

#define SET_TEX_WRAP(TEX_WRAP, TEX_2D_WRAP)                                    \
  if (TEX_2D_WRAP == zeno::Texture2DObject::TexWrapEnum::REPEAT)               \
    TEX_WRAP = GL_REPEAT;                                                      \
  else if (TEX_2D_WRAP == zeno::Texture2DObject::TexWrapEnum::MIRRORED_REPEAT) \
    TEX_WRAP = GL_MIRRORED_REPEAT;                                             \
  else if (TEX_2D_WRAP == zeno::Texture2DObject::TexWrapEnum::CLAMP_TO_EDGE)   \
    TEX_WRAP = GL_CLAMP_TO_EDGE;                                               \
  else if (TEX_2D_WRAP == zeno::Texture2DObject::TexWrapEnum::CLAMP_TO_BORDER) \
    TEX_WRAP = GL_CLAMP_TO_BORDER;

      SET_TEX_WRAP(tex->wrap_s, tex2D->wrapS)
      SET_TEX_WRAP(tex->wrap_t, tex2D->wrapT)

#undef SET_TEX_WRAP

#define SET_TEX_FILTER(TEX_FILTER, TEX_2D_FILTER)                                         \
  if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::NEAREST)                     \
    TEX_FILTER = GL_NEAREST;                                                              \
  else if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::LINEAR)                 \
    TEX_FILTER = GL_LINEAR;                                                               \
  else if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_NEAREST) \
    TEX_FILTER = GL_NEAREST_MIPMAP_NEAREST;                                               \
  else if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_NEAREST)  \
    TEX_FILTER = GL_LINEAR_MIPMAP_NEAREST;                                                \
  else if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_LINEAR)  \
    TEX_FILTER = GL_NEAREST_MIPMAP_LINEAR;                                                \
  else if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_LINEAR)   \
    TEX_FILTER = GL_LINEAR_MIPMAP_LINEAR;

      SET_TEX_FILTER(tex->min_filter, tex2D->minFilter)
      SET_TEX_FILTER(tex->mag_filter, tex2D->magFilter)

#undef SET_TEX_FILTER
      textures.push_back(std::move(tex));
    }
  }

  Program *get_points_program(std::string const &path) {
    auto vert = hg::file_get_content(path + ".points.vert");
    auto frag = hg::file_get_content(path + ".points.frag");

    if (vert.size() == 0) {
      vert = R"(
#version 120

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
    }
    if (frag.size() == 0) {
      frag = R"(
#version 120

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
    }

    return compile_program(vert, frag);
  }

  Program *get_lines_program(std::string const &path) {
    auto vert = hg::file_get_content(path + ".lines.vert");
    auto frag = hg::file_get_content(path + ".lines.frag");

    if (vert.size() == 0) {
      vert = R"(
#version 120

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
    }
    if (frag.size() == 0) {
      frag = R"(
#version 120

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
    }

    return compile_program(vert, frag);
  }

  Program *get_tris_program(std::string const &path, std::shared_ptr<zeno::MaterialObject> mtl) {
    auto vert = hg::file_get_content(path + ".tris.vert");
    auto frag = hg::file_get_content(path + ".tris.frag");

    if (vert.size() == 0) {
      vert = R"(
#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

attribute vec3 vPosition;
attribute vec3 vColor;
attribute vec3 vNormal;
attribute vec3 vTexCoord;
attribute vec3 vTangent;

varying vec3 position;
varying vec3 iColor;
varying vec3 iNormal;
varying vec3 iTexCoord;

void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;
  iTexCoord = vTexCoord;

  gl_Position = mVP * vec4(position, 1.0);
}
)";
    }
    if (frag.size() == 0) {
      frag = R"(
#version 120
)" + (mtl ? mtl->extensions : "") + R"(
const float minDot = 1e-5;

// Clamped dot product
float dot_c(vec3 a, vec3 b){
	return max(dot(a, b), minDot);
}

// Get orthonormal basis from surface normal
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
void pixarONB(vec3 n, out vec3 b1, out vec3 b2){
	vec3 up        = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    b1   = normalize(cross(up, n));
    b2 = cross(n, b1);
}

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform bool mSmoothShading;
uniform bool mNormalCheck;
uniform bool mRenderWireframe;

varying vec3 position;
varying vec3 iColor;
varying vec3 iNormal;
varying vec3 iTexCoord;
uniform samplerCube skybox;
vec3 pbr(vec3 albedo, float roughness, float metallic, float specular,
    vec3 nrm, vec3 idir, vec3 odir) {

  vec3 hdir = normalize(idir + odir);
  float NoH = max(0., dot(hdir, nrm));
  float NoL = max(0., dot(idir, nrm));
  float NoV = max(0., dot(odir, nrm));
  float VoH = clamp(dot(odir, hdir), 0., 1.);
  float LoH = clamp(dot(idir, hdir), 0., 1.);

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

)" + (
!mtl ?
R"(
vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal) {
    vec3 color = vec3(0.0);
    vec3 light_dir;

    light_dir = normalize((mInvView * vec4(1., 2., 5., 0.)).xyz);
    color += vec3(0.45, 0.47, 0.5) * pbr(albedo, 0.44, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(-4., -2., 1., 0.)).xyz);
    color += vec3(0.3, 0.23, 0.18) * pbr(albedo, 0.37, 0.0, 1.0, normal, light_dir, view_dir);

    light_dir = normalize((mInvView * vec4(3., -5., 2., 0.)).xyz);
    color += vec3(0.15, 0.2, 0.22) * pbr(albedo, 0.48, 0.0, 1.0, normal, light_dir, view_dir);

    color *= 1.2;
    //color = pow(clamp(color, 0., 1.), vec3(1./2.2));
    return color;
}
)" : "\n/* common_funcs_begin */\n" + mtl->common + "\n/* common_funcs_end */\n"
R"(
  
vec3 CalculateDiffuse(
    in vec3 albedo){                              
    return (albedo / 3.1415926);
}


vec3 CalculateHalfVector(
    in vec3 toLight, in vec3 toView){
    return normalize(toLight + toView);
}

// Specular D -  Normal distribution function (NDF)
float CalculateNDF( // GGX/Trowbridge-Reitz NDF
    in vec3  surfNorm,
    in vec3  halfVector,
    in float roughness){
    float a2 = (roughness * roughness * roughness * roughness);
    float halfAngle = dot(surfNorm, halfVector);
    float d = (halfAngle * a2 - halfAngle) * halfAngle + 1;
    return (a2 / (3.1415926 *  d * d));
}

// Specular G - Microfacet geometric attenuation
float CalculateAttenuation( // GGX/Schlick-Beckmann
    in vec3  surfNorm,
    in vec3  vector,
    in float k)
{
    float d = max(dot(surfNorm, vector), 0.0);
 	return (d / ((d * (1.0 - k)) + k));
}
float CalculateAttenuationAnalytical(// Smith for analytical light
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in float roughness)
{
    float k = pow((roughness*roughness + 1.0), 2.0) * 0.125;

    // G(l) and G(v)
    float lightAtten = CalculateAttenuation(surfNorm, toLight, k);
    float viewAtten  = CalculateAttenuation(surfNorm, toView, k);

    // Smith
    return (lightAtten * viewAtten);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness){
    return F0 + (max(vec3(1.0-roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}
// Specular F - Fresnel reflectivity
vec3 CalculateFresnel(
    in vec3 surfNorm,
    in vec3 toView,
    in vec3 fresnel0)
{
	float d = max(dot(surfNorm, toView), 0.0); 
    float p = ((-5.55473 * d) - 6.98316) * d;
        
    return fresnel0 + ((1.0 - fresnel0) * pow(1.0 - d, 5.0));
}

// Specular Term - put together
vec3 CalculateSpecularAnalytical(
    in    vec3  surfNorm,            // Surface normal
    in    vec3  toLight,             // Normalized vector pointing to light source
    in    vec3  toView,              // Normalized vector point to the view/camera
    in    vec3  fresnel0,            // Fresnel incidence value
    inout vec3  sfresnel,            // Final fresnel value used a kS
    in    float roughness)           // Roughness parameter (microfacet contribution)
{
    vec3 halfVector = CalculateHalfVector(toLight, toView);

    float ndf      = CalculateNDF(surfNorm, halfVector, roughness);
    float geoAtten = CalculateAttenuationAnalytical(surfNorm, toLight, toView, roughness);

    sfresnel = CalculateFresnel(surfNorm, toView, fresnel0);

    vec3  numerator   = (sfresnel * ndf * geoAtten); // FDG
    float denominator = 4.0 * dot(surfNorm, toLight) * dot(surfNorm, toView);

    return (numerator / denominator);
}
float D_GGX( float a2, float NoH )
{
	float d = ( NoH * a2 - NoH ) * NoH + 1;	// 2 mad
	return a2 / ( 3.1415926*d*d );					// 4 mul, 1 rcp
}
float Vis_SmithJointApprox( float a2, float NoV, float NoL )
{
	float a = sqrt(a2);
	float Vis_SmithV = NoL * ( NoV * ( 1 - a ) + a );
	float Vis_SmithL = NoV * ( NoL * ( 1 - a ) + a );
	return 0.5 / ( Vis_SmithV + Vis_SmithL );
}
vec3 F_Schlick( vec3 SpecularColor, float VoH )
{
	float Fc = pow( 1 - VoH , 5.0 );					// 1 sub, 3 mul
	//return Fc + (1 - Fc) * SpecularColor;		// 1 add, 3 mad
	
	
	return clamp( 50.0 * SpecularColor.g, 0, 1 ) * Fc + (1 - Fc) * SpecularColor;
	
}
vec3 SpecularGGX( float Roughness, vec3 SpecularColor, float NoL, float NoH, float NoV, float VoH)
{
	float a2 = pow( Roughness, 4);
	
	// Generalized microfacet specular
    float D = D_GGX( a2,  NoH);
	float Vis = Vis_SmithJointApprox( a2, NoV, NoL );
	vec3 F = F_Schlick( SpecularColor, VoH );

	return (D * Vis) * F;
}
vec3 UELighting(
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    vec3 ks       = vec3(0.0);
    vec3 diffuse  = CalculateDiffuse(albedo);
    vec3 half = normalize(toLight + toView);
    float NoL = dot(surfNorm, toLight);
    float NoH = dot(surfNorm, half);
    float NoV = dot(surfNorm, toView);
    float VoH = dot(toView, half);
    float angle = clamp(dot(surfNorm, toLight), 0.0, 1.0);
    return (diffuse * (1-metallic) + SpecularGGX(roughness, vec3(0,0,0), NoL, NoH, NoV, NoH))*angle;

}
// Solve Rendering Integral - Final
vec3 CalculateLightingAnalytical(
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    vec3 fresnel0 = mix(vec3(0.04), albedo, metallic);
    vec3 ks       = vec3(0.0);
    vec3 diffuse  = CalculateDiffuse(albedo);
    vec3 specular = CalculateSpecularAnalytical(surfNorm, toLight, toView, fresnel0, ks, roughness);
    vec3 kd       = (1.0 - ks);

    float angle = clamp(dot(surfNorm, toLight), 0.0, 1.0);

    return ((kd * diffuse) + specular) * angle;
}
float VanDerCorpus(int n, int base) {
    float invBase = 1.0 / float(base);
    float denom   = 1.0;
    float result  = 0.0;

    for(int i = 0; i < 32; ++i)
    {
        if(n > 0)
        {
            denom   = mod(float(n), 2.0);
            result += denom * invBase;
            invBase = invBase / 2.0;
            n       = int(float(n) / 2.0);
        }
    }

    return result;
}

vec2 Hammersley(int i, int N) {
    return vec2(float(i)/float(N), VanDerCorpus(i, 2));
}  
float CalculateAttenuationIBL(
    in float roughness,
    in float normDotLight,          // Clamped to [0.0, 1.0]
    in float normDotView)           // Clamped to [0.0, 1.0]
{
    float k = pow(roughness*roughness, 2.0) * 0.5;
    
    float lightAtten = (normDotLight / ((normDotLight * (1.0 - k)) + k));
    float viewAtten  = (normDotView / ((normDotView * (1.0 - k)) + k));
    
    return (lightAtten * viewAtten);
}

vec3 ImportanceSample(vec2 Xi, vec3 N, float roughness) {
    float a = roughness*roughness;
	
    float phi = 2.0 * 3.1415926 * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    //vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent;//   = normalize(cross(up, N));
    vec3 bitangent;// = cross(N, tangent);
	pixarONB(N, tangent, bitangent);
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

#define time 0
float hash2(in vec2 n){ return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453); }

mat2 mm2(in float a){float c = cos(a), s = sin(a);return mat2(c,-s,s,c);}

vec2 field(in vec2 x)
{
    vec2 n = floor(x);
	vec2 f = fract(x);
	vec2 m = vec2(5.,1.);
	for(int j=0; j<=1; j++)
	for(int i=0; i<=1; i++)
    {
		vec2 g = vec2( float(i),float(j) );
		vec2 r = g - f;
        float d = length(r)*(sin(time*0.12)*0.5+1.5); //any metric can be used
        d = sin(d*5.+abs(fract(time*0.1)-0.5)*1.8+0.2);
		m.x *= d;
		m.y += d*1.2;
    }
	return abs(m);
}

vec3 tex(in vec2 p, in float ofst)
{    
    vec2 rz = field(p*ofst*0.5);
	vec3 col = sin(vec3(2.,1.,.1)*rz.y*.2+3.+ofst*2.)+.9*(rz.x+1.);
	col = col*col*.5;
    col *= sin(length(p)*9.+time*5.)*0.35+0.65;
	return col;
}

vec3 cubem(in vec3 p, in float ofst)
{
    p = abs(p);
    if (p.x > p.y && p.x > p.z) return tex( vec2(p.z,p.y)/p.x,ofst );
    else if (p.y > p.x && p.y > p.z) return tex( vec2(p.z,p.x)/p.y,ofst );
    else return tex( vec2(p.y,p.x)/p.z,ofst );
}



//important to do: load env texture here
vec3 SampleEnvironment(in vec3 reflVec)
{
    //if(reflVec.y>-0.5) return vec3(0,0,0);
    //else return vec3(1,1,1);//cubem(reflVec, 0);//texture(TextureEnv, reflVec).rgb;
    return textureCube(skybox, reflVec).rgb;
}

/**
 * Performs the Riemann Sum approximation of the IBL lighting integral.
 *
 * The ambient IBL source hits the surface from all angles. We average
 * the lighting contribution from a number of random light directional
 * vectors to approximate the total specular lighting.
 *
 * The number of steps is controlled by the 'IBL Steps' global.
 */
 //Geometry for IBL uses a different k than direct lighting
 //GGX and Schlick-Beckmann
float geometry(float cosTheta, float k){
	return (cosTheta)/(cosTheta*(1.0-k)+k);
}
float smithsIBL(float NdotV, float NdotL, float roughness){
    float k = (roughness * roughness);
    k = k*k; 
	return geometry(NdotV, k) * geometry(NdotL, k);
}
vec3 CalculateSpecularIBL(
    in    vec3  surfNorm,
    in    vec3  toView,
    in    vec3  fresnel0,
    inout vec3  sfresnel,
    in    float roughness)
{
    vec3 totalSpec = vec3(0.0);
    vec3 toSurfaceCenter = reflect(-toView, surfNorm);
    int IBLSteps = 64;
    for(int i = 0; i < IBLSteps; ++i)
    {
        // The 2D hemispherical sampling vector
    	vec2 xi = Hammersley(i, IBLSteps);
        
        // Bias the Hammersley vector towards the specular lobe of the surface roughness
        vec3 H = ImportanceSample(xi, surfNorm, roughness);
        
        // The light sample vector
        vec3 L = normalize((2.0 * dot(toView, H) * H) - toView);
        
        float NoV = clamp(dot(surfNorm, toView), 0.0, 1.0);
        float NoL = clamp(dot(surfNorm, L), 0.0, 1.0);
        float NoH = clamp(dot(surfNorm, H), 0.0, 1.0);
        float VoH = clamp(dot(toView, H), 0.0, 1.0);
        
        if(NoL > 0.0)
        {
            vec3 color = SampleEnvironment(L);
            
            float geoAtten = smithsIBL(NoV, NoL, roughness);
            vec3  fresnel = CalculateFresnel(surfNorm, toView, fresnel0);
            
            sfresnel += fresnel;
            totalSpec += (color * fresnel * geoAtten * VoH) / (NoH * NoV);
        }
    }
    
    sfresnel /= float(IBLSteps);
    
    return (totalSpec / float(IBLSteps));
}

vec3 CalculateLightingIBL(
    in vec3  surfNorm,
    in vec3  toView,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    vec3 fresnel0 = mix(vec3(0.04), albedo, metallic);
    //vec3 F = fresnelSchlickRoughness(dot_c(surfNorm, toView), 0.04);
    vec3 ks       = vec3(0);
    vec3 diffuse  = CalculateDiffuse(albedo) * CalculateSpecularIBL(surfNorm, toView, fresnel0, ks, 1.0);
    vec3 specular = CalculateSpecularIBL(surfNorm, toView, fresnel0, ks, roughness);
    vec3 kd       = (1.0 - ks);
    

    return ((kd * diffuse) + specular);

}

vec3 ACESToneMapping(vec3 color, float adapted_lum)
{
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	color *= adapted_lum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
}

const float PI = 3.14159265358979323846;

float sqr(float x) { return x*x; }

float SchlickFresnel(float u)
{
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

float GTR1(float NdotH, float a)
{
    if (a >= 1) return 1/PI;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (PI*log(a2)*t);
}

float GTR2(float NdotH, float a)
{
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return a2 / (PI * t*t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1 / (PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ));
}

float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ));
}

vec3 mon2lin(vec3 x)
{
    return vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
}


vec3 BRDF(vec3 baseColor, float metallic, float subsurface, 
float specular, 
float roughness,
float specularTint,
float anisotropic,
float sheen,
float sheenTint,
float clearcoat,
float clearcoatGloss,
vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y )
{
    float NdotL = dot(N,L);
    float NdotV = dot(N,V);
    //if (NdotL < 0 || NdotV < 0) return vec3(0);

    vec3 H = normalize(L+V);
    float NdotH = dot(N,H);
    float LdotH = dot(L,H);

    vec3 Cdlin = mon2lin(baseColor);
    float Cdlum = .3*Cdlin[0] + .6*Cdlin[1]  + .1*Cdlin[2]; // luminance approx.

    vec3 Ctint = Cdlum > 0 ? Cdlin/Cdlum : vec3(1); // normalize lum. to isolate hue+sat
    vec3 Cspec0 = mix(specular*.08*mix(vec3(1), Ctint, specularTint), Cdlin, metallic);
    vec3 Csheen = mix(vec3(1), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5 + 2 * LdotH*LdotH * roughness;
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH*LdotH*roughness;
    float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
    float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);

    // specular
    float aspect = sqrt(1-anisotropic*.9);
    float ax = max(.001, sqr(roughness)/aspect);
    float ay = max(.001, sqr(roughness)*aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0, vec3(1), FH);
    float Gs;
    Gs  = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    // sheen
    vec3 Fsheen = FH * sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, mix(.1,.001,clearcoatGloss));
    float Fr = mix(.04, 1.0, FH);
    float Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25);
    float angle = clamp(dot(N, L), 0.0, 1.0);
    return (((1/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen)
        * (1-metallic)
        + Gs*Fs*Ds + .25*clearcoat*Gr*Fr*Dr)*angle;
}

const mat3x3 ACESInputMat = mat3x3
(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3x3 ACESOutputMat = mat3x3
(
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

vec3 ACESFitted(vec3 color, float gamma)
{
    color = color * ACESInputMat;

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = color * ACESOutputMat;

    // Clamp to [0, 1]
  	color = clamp(color, 0.0, 1.0);
    
    color = pow(color, vec3(1. / gamma));

    return color;
}

vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal) {
    vec3 att_pos = position;
    vec3 att_clr = iColor;
    vec3 att_nrm = normal;
    vec3 att_uv = iTexCoord;

    /* custom_shader_begin */
)" + mtl->frag + R"(
    /* custom_shader_end */
    mat_metallic = clamp(mat_metallic, 0, 1);
    vec3 new_normal = normal; /* TODO: use mat_normal to transform this */
    vec3 color = vec3(0,0,0);
    vec3 light_dir;
    vec3 albedo2 = mat_basecolor;
    float roughness = mat_roughness;

    new_normal = normalize(gl_NormalMatrix * new_normal);
    //vec3 up        = abs(new_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent;//   = normalize(cross(up, new_normal));
    vec3 bitangent;// = cross(new_normal, tangent);
    pixarONB(new_normal, tangent, bitangent);
    light_dir = vec3(1,1,0);
    color += BRDF(mat_basecolor, mat_metallic,mat_subsurface,mat_specular,mat_roughness,mat_specularTint,mat_anisotropic,mat_sheen,mat_sheenTint,mat_clearcoat,mat_clearcoatGloss,normalize(light_dir), normalize(view_dir), normalize(new_normal),normalize(tangent), normalize(bitangent)) * vec3(1, 1, 1) * mat_zenxposure;
 //   color +=  
 //       CalculateLightingAnalytical(
 //           new_normal,
 //           normalize(light_dir),
 //           normalize(view_dir),
 //           albedo2,
 //           roughness,
 //           mat_metallic) * vec3(1, 1, 1) * mat_zenxposure;
//    color += vec3(0.45, 0.47, 0.5) * pbr(mat_basecolor, mat_roughness,
//             mat_metallic, mat_specular, new_normal, light_dir, view_dir);

    light_dir = vec3(0,1,-1);
//    color += vec3(0.3, 0.23, 0.18) * pbr(mat_basecolor, mat_roughness,
//             mat_metallic, mat_specular, new_normal, light_dir, view_dir);
//    color +=  
//        CalculateLightingAnalytical(
//            new_normal,
//            light_dir,
//            view_dir,
//            albedo2,
//            roughness,
//            mat_metallic) * vec3(0.3, 0.23, 0.18)*5;
//    light_dir = vec3(0,-0.2,-1);
//    color +=  
//        CalculateLightingAnalytical(
//            new_normal,
//            light_dir,
//            view_dir,
//            albedo2,
//            roughness,
//            mat_metallic) * vec3(0.15, 0.2, 0.22)*6;
//    color += vec3(0.15, 0.2, 0.22) * pbr(mat_basecolor, mat_roughness,
//             mat_metallic, mat_specular, new_normal, light_dir, view_dir);

    color +=  
        CalculateLightingIBL(
            new_normal,
            view_dir,
            albedo2,
            roughness,
            mat_metallic);
    color = ACESFitted(color.rgb, 2.2);
    return color;

}
)"
) + R"(

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
  normal = faceforward(normal, -viewdir, normal);

  vec3 albedo = iColor;
  vec3 color = studioShading(albedo, viewdir, normal);
  
  gl_FragColor = vec4(color, 1.0);
  if (mNormalCheck) {
      float intensity = clamp((mView * vec4(normal, 0)).z, 0, 1) * 0.4 + 0.6;
      if (gl_FrontFacing) {
        gl_FragColor = vec4(0.42 * intensity, 0.42 * intensity, 0.93 * intensity, 1.0);
      } else {
        gl_FragColor = vec4(0.87 * intensity, 0.22 * intensity, 0.22 * intensity, 1.0);
      }
  }
}
)";
    }

//printf("!!!!%s!!!!\n", frag.c_str());
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
