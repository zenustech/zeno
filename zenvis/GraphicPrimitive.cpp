#include "glad/glad.h"
#include "stdafx.hpp"
#include "IGraphic.hpp"
#include "MyShader.hpp"
#include "main.hpp"
#include <memory>
#include <string>
#include <vector>
#include <zeno/utils/vec.h>
#include <zeno/utils/ticktock.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/types/InstancingObject.h>
#include <Hg/IOUtils.h>
#include <Hg/IterUtils.h>
#include <Scene.hpp>
#include "voxelizeProgram.h"
#include "shaders.h"
#include <map>
namespace zenvis {
extern float getCamFar();
extern void ensureGlobalMapExist();
extern unsigned int getGlobalEnvMap();
extern unsigned int getIrradianceMap();
extern unsigned int getPrefilterMap();
extern unsigned int getBRDFLut();
extern glm::mat4 getReflectMVP(int i);
extern std::vector<unsigned int> getReflectMaps();
extern void setReflectivePlane(int i, glm::vec3 n, glm::vec3 c);
extern bool renderReflect(int i);
extern int getReflectionViewID();
extern void setCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, double _fov, double fnear, double ffar, double _dof, int set);
extern unsigned int getDepthTexture();
extern glm::vec3 getReflectiveNormal(int i);
extern glm::vec3 getReflectiveCenter(int i);

struct drawObject
{
    std::unique_ptr<Buffer> vbo;
    std::unique_ptr<Buffer> ebo;
    std::unique_ptr<Buffer> instvbo;
    size_t count=0;
    Program *prog;
    Program *shadowprog;
    Program *voxelprog;
};
void parsePointsDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj)
{
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &uv = prim->attr<zeno::vec3f>("uv");
    auto const &tang = prim->attr<zeno::vec3f>("tang");
    obj.count = prim->size();

    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 5);
    for (int i = 0; i < obj.count; i++)
    {
      mem[5 * i + 0] = pos[i];
      mem[5 * i + 1] = clr[i];
      mem[5 * i + 2] = nrm[i];
      mem[5 * i + 3] = uv[i];
      mem[5 * i + 4] = tang[i];
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
    auto const &tang = prim->attr<zeno::vec3f>("tang");
    auto const &lines = prim->lines;
    bool has_uv = lines.has_attr("uv0")&&lines.has_attr("uv1");
    obj.count = prim->lines.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 2 * 5);
    std::vector<zeno::vec2i> linesdata(obj.count);
    #pragma omp parallel for
    for(int i=0; i<obj.count;i++)
    {
        mem[10 * i + 0] = pos[lines[i][0]];
        mem[10 * i + 1] = clr[lines[i][0]];
        mem[10 * i + 2] = nrm[lines[i][0]];
        mem[10 * i + 3] = has_uv? lines.attr<zeno::vec3f>("uv0")[i]:zeno::vec3f(0,0,0);
        mem[10 * i + 4] = tang[lines[i][0]];
        mem[10 * i + 5] = pos[lines[i][1]];
        mem[10 * i + 6] = clr[lines[i][1]];
        mem[10 * i + 7] = nrm[lines[i][1]];
        mem[10 * i + 8] = has_uv? lines.attr<zeno::vec3f>("uv1")[i]:zeno::vec3f(0,0,0);
        mem[10 * i + 9] = tang[lines[i][1]];
        linesdata[i] = zeno::vec2i(i*2, i*2+1);

    }
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    if(obj.count)
    {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(linesdata[0]), obj.count * sizeof(linesdata[0]));
    }
}

void computeTrianglesTangent(zeno::PrimitiveObject *prim)
{
    const auto &tris = prim->tris;
    const auto &pos = prim->attr<zeno::vec3f>("pos");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto &tang = prim->tris.add_attr<zeno::vec3f>("tang");
    bool has_uv = tris.has_attr("uv0")&&tris.has_attr("uv1")&&tris.has_attr("uv2");
    //printf("!!has_uv = %d\n", has_uv);
#pragma omp parallel for
    for (size_t i = 0; i < prim->tris.size(); ++i)
    {
        if(has_uv) {
            const auto &pos0 = pos[tris[i][0]];
            const auto &pos1 = pos[tris[i][1]];
            const auto &pos2 = pos[tris[i][2]];
            auto uv0 = tris.attr<zeno::vec3f>("uv0")[i];
            auto uv1 = tris.attr<zeno::vec3f>("uv1")[i];
            auto uv2 = tris.attr<zeno::vec3f>("uv2")[i];

            auto edge0 = pos1 - pos0;
            auto edge1 = pos2 - pos0;
            auto deltaUV0 = uv1 - uv0;
            auto deltaUV1 = uv2 - uv0;

            auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

            zeno::vec3f tangent;
            tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
            tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
            tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
            //printf("%f %f %f\n", tangent[0], tangent[1], tangent[3]);
            auto tanlen = zeno::length(tangent);
            tangent * (1.f / (tanlen + 1e-8));
            /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
                zeno::vec3f n = nrm[tris[i][0]], unused;
                zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
            } else {
                tang[i] = tangent * (1.f / tanlen);
            }*/
            tang[i] = tangent;
        } else {
            tang[i] = zeno::vec3f(0);
            //zeno::vec3f n = nrm[tris[i][0]], unused;
            //zeno::pixarONB(n, tang[i], unused);
        }
    }
}
void parseTrianglesDrawBufferCompress(zeno::PrimitiveObject *prim, drawObject &obj)
{
    //TICK(parse);
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tris = prim->tris;
    bool has_uv = tris.has_attr("uv0")&&tris.has_attr("uv1")&&tris.has_attr("uv2");
    auto &tang = prim->tris.attr<zeno::vec3f>("tang");
    std::vector<zeno::vec3f> pos1(pos.size());
    std::vector<zeno::vec3f> clr1(pos.size());
    std::vector<zeno::vec3f> nrm1(pos.size());
    std::vector<zeno::vec3f> uv1(pos.size());
    std::vector<zeno::vec3f> tang1(pos.size());
    std::vector<int> vertVisited(pos.size());
    std::vector<zeno::vec3i> tris1(tris.size());
    vertVisited.assign(pos.size(),0);
    for(int i=0; i<tris.size();i++)
    {
        float area = zeno::length(zeno::cross(pos[tris[i][1]]-pos[tris[i][0]], pos[tris[i][2]]-pos[tris[i][0]]));
        for(int j=0;j<3;j++)
        {
            tang1[tris[i][j]]+=area*tang[i];
        }
    }
    std::cout<<"1111111111111111\n";
    #pragma omp parallel for
    for(int i=0; i<tang1.size();i++)
    {
        tang1[i] = tang[i]/(zeno::length(tang[i])+0.000001);
    }
    std::cout<<"2222222222222222\n";
    std::vector<int> issueTris(0);
    for(int i=0; i<tris.size();i++)
    {
        //if all verts not visited
        for(int j=0;j<3;j++)
        {
            //just add verts id
        }

        //else
        {
            //if no uv confliction
            //simply add verts id
            //else
            {
                //add this tri to issueTris
            }
        }
    }
    //for issueTris
    {
        //emit new verts
    }
    std::cout<<"3333333333333333333\n";

    //end compressed tri assign
    obj.count = tris1.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(pos1.size()  * 5);
    std::vector<zeno::vec3i> trisdata(obj.count);
#pragma omp parallel for
    for(int i=0; i<pos1.size(); i++)
    {
        mem[5 * i + 0]  = pos1[i];
        mem[5 * i + 1]  = clr1[i];
        mem[5 * i + 2]  = nrm1[i];
        mem[5 * i + 3]  = uv1[i];
        mem[5 * i + 4]  = tang1[i];
    }
#pragma omp parallel for
    for(int i=0; i<tris1.size();i++)
    {
        trisdata[i] = tris1[i];
    }

    TICK(bindvbo);
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    TOCK(bindvbo);
    TICK(bindebo);
    if(obj.count)
    {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(trisdata[0]), tris1.size() * sizeof(trisdata[0]));
    }
    TOCK(bindebo);
}
void parseTrianglesDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj)
{
    TICK(parse);
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tris = prim->tris;
    bool has_uv = tris.has_attr("uv0")&&tris.has_attr("uv1")&&tris.has_attr("uv2");
    obj.count = tris.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 3 * 5);
    std::vector<zeno::vec3i> trisdata(obj.count);
    auto &tang = prim->tris.attr<zeno::vec3f>("tang");
#pragma omp parallel for
    for(int i=0; i<obj.count;i++)
    {
        mem[15 * i + 0]  = pos[tris[i][0]];
        mem[15 * i + 1]  = clr[tris[i][0]];
        mem[15 * i + 2]  = nrm[tris[i][0]];
        mem[15 * i + 3]  = has_uv ? tris.attr<zeno::vec3f>("uv0")[i] : zeno::vec3f(0.0f, 0.0f, 0.0f);
        mem[15 * i + 4]  = tang[i];
        mem[15 * i + 5]  = pos[tris[i][1]];
        mem[15 * i + 6]  = clr[tris[i][1]];
        mem[15 * i + 7]  = nrm[tris[i][1]];
        mem[15 * i + 8]  = has_uv ? tris.attr<zeno::vec3f>("uv1")[i] : zeno::vec3f(0.0f, 0.0f, 0.0f);
        mem[15 * i + 9]  = tang[i];
        mem[15 * i + 10] = pos[tris[i][2]];
        mem[15 * i + 11] = clr[tris[i][2]];
        mem[15 * i + 12] = nrm[tris[i][2]];
        mem[15 * i + 13] = has_uv ? tris.attr<zeno::vec3f>("uv2")[i] : zeno::vec3f(0.0f, 0.0f, 0.0f);
        mem[15 * i + 14] = tang[i];
        //std::cout<<tang[i][0]<<" "<<tang[i][1]<<" "<<tang[i][2]<<std::endl;
        trisdata[i] = zeno::vec3i(i*3, i*3+1, i*3+2);

    }
    TOCK(parse);

    TICK(bindvbo);
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    TOCK(bindvbo);
    TICK(bindebo);
    if(obj.count)
    {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(trisdata[0]), obj.count * sizeof(trisdata[0]));
    }
    TOCK(bindebo);
}

struct InstVboData
{
    glm::mat4 modelMatrix;
    float time;
};
std::map<std::string, std::shared_ptr<Texture>> g_Textures;
struct GraphicPrimitive : IGraphic {
  std::unique_ptr<Buffer> vbo;
  std::unique_ptr<Buffer> instvbo;
  size_t vertex_count;
  bool draw_all_points;

  //Program *points_prog;
  //std::unique_ptr<Buffer> points_ebo;
  size_t points_count;

  //Program *lines_prog;
  //std::unique_ptr<Buffer> lines_ebo;
  size_t lines_count;

  //Program *tris_prog;
  //std::unique_ptr<Buffer> tris_ebo;
  size_t tris_count;

  drawObject pointObj;
  drawObject lineObj;
  drawObject triObj;
  std::map<int, std::string> textures;

  bool prim_has_mtl = false;
  bool prim_has_inst = false;
  int prim_inst_amount = 0;
  float prim_inst_delta_time = 0.0f;
  int prim_inst_frame_amount = 0;
  std::unique_ptr<Texture> prim_inst_vertex_frame_sampler;
  
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
    bool primNormalCorrect = prim->has_attr("nrm") && length(prim->attr<zeno::vec3f>("nrm")[0])>1e-5;
    bool need_computeNormal = !primNormalCorrect || !(prim->has_attr("nrm"));
    if(prim->tris.size() && need_computeNormal)
    {
        std::cout<<"computing normal\n";
        zeno::primCalcNormal(prim, 1);
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
    if (!prim->has_attr("tang"))
    {
        auto &tang = prim->add_attr<zeno::vec3f>("tang");
        for (size_t i = 0; i < tang.size(); i++) {
            tang[i] = zeno::vec3f(0.0f);
        }
    }
    bool enable_uv = false;
    
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &uv = prim->attr<zeno::vec3f>("uv");
    auto const &tang = prim->attr<zeno::vec3f>("tang");
    vertex_count = prim->size();

    vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(vertex_count * 5);
    for (int i = 0; i < vertex_count; i++)
    {
      mem[5 * i + 0] = pos[i];
      mem[5 * i + 1] = clr[i];
      mem[5 * i + 2] = nrm[i];
      mem[5 * i + 3] = uv[i];
      mem[5 * i + 4] = tang[i];
    }
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    if (prim->inst != nullptr)
    {
        prim_has_inst = true;

        const auto &inst = prim->inst;

        const auto amount = inst->amount;    
        prim_inst_amount = amount;    

        prim_inst_delta_time = inst->deltaTime;

        const auto &vertexFrameBuffer = inst->vertexFrameBuffer;    
        const auto frameAmount = vertexFrameBuffer.size();    
        prim_inst_frame_amount = frameAmount;

        std::size_t vertexAmount = 0;
        if (frameAmount > 0)
        {
            vertexAmount = vertexFrameBuffer[0].size();            
        }
        std::vector<float> samplerData(3 * vertexAmount * frameAmount);
#pragma omp parallel for
        for (int i = 0; i < frameAmount; ++i)
        {
#pragma omp parallel for
            for (int j = 0; j < vertexAmount; ++j)
            {
                int k = (i * vertexAmount + j) * 3;
                const auto &data = vertexFrameBuffer[i][j];
                samplerData[k + 0] = data[0];
                samplerData[k + 1] = data[1];
                samplerData[k + 2] = data[2];
            }
        }
        prim_inst_vertex_frame_sampler = std::make_unique<Texture>();
        prim_inst_vertex_frame_sampler->target = GL_TEXTURE_2D;
        prim_inst_vertex_frame_sampler->wrap_s = GL_CLAMP_TO_EDGE;
        prim_inst_vertex_frame_sampler->wrap_t = GL_CLAMP_TO_EDGE;
        prim_inst_vertex_frame_sampler->min_filter = GL_LINEAR;
        prim_inst_vertex_frame_sampler->mag_filter = GL_LINEAR;
        prim_inst_vertex_frame_sampler->internal_fmt = GL_RGB32F;
        prim_inst_vertex_frame_sampler->format = GL_RGB;
        prim_inst_vertex_frame_sampler->dtype = GL_FLOAT;
        prim_inst_vertex_frame_sampler->bind_image(samplerData.data(), vertexAmount, frameAmount);

        instvbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        std::vector<InstVboData> vboData(amount);
        const auto &modelMatrices = inst->modelMatrices;
        const auto &timeList = inst->timeList;
#pragma omp parallel for
        for (int i = 0; i < amount; ++i)
        {
            auto &instVboData = vboData[i];
            instVboData.modelMatrix = modelMatrices[i];
            instVboData.time = timeList[i];
        }
        instvbo->bind_data(vboData.data(), amount * sizeof(InstVboData));
    }

    points_count = prim->points.size();
    if (points_count) {
        pointObj.count = points_count;
        pointObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        pointObj.ebo->bind_data(prim->points.data(), points_count * sizeof(prim->points[0]));
        pointObj.prog = get_points_program(path);
    }

    lines_count = prim->lines.size();
    if (lines_count) {
        // lines_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        // lines_ebo->bind_data(prim->lines.data(), lines_count * sizeof(prim->lines[0]));
        // lines_prog = get_lines_program(path);
        if (!(prim->lines.has_attr("uv0")&&prim->lines.has_attr("uv1"))) {
            lineObj.count = lines_count;
            lineObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
            lineObj.ebo->bind_data(prim->lines.data(), lines_count * sizeof(prim->lines[0]));
            lineObj.vbo = nullptr;
        } else {
            parseLinesDrawBuffer(prim, lineObj);
        }
        lineObj.prog = get_lines_program(path);
    }

    tris_count = prim->tris.size();
    if (tris_count) {
        // tris_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        // tris_ebo->bind_data(prim->tris.data(), tris_count * sizeof(prim->tris[0]));
        // tris_prog = get_tris_program(path, prim->mtl);
        // if (!tris_prog)
        //     tris_prog = get_tris_program(path, nullptr);
        
        if (!(prim->tris.has_attr("uv0")&&prim->tris.has_attr("uv1")&&prim->tris.has_attr("uv2"))) {
            triObj.count = tris_count;
            triObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
            triObj.ebo->bind_data(prim->tris.data(), tris_count * sizeof(prim->tris[0]));
            triObj.vbo = nullptr;
            triObj.instvbo = nullptr;
        } else {
            computeTrianglesTangent(prim);
            parseTrianglesDrawBuffer(prim, triObj);

            if (prim->inst != nullptr)
            {
                const auto &inst = prim->inst;

                const auto amount = inst->amount;    

                const auto &vertexFrameBuffer = inst->vertexFrameBuffer;    
                const auto frameAmount = vertexFrameBuffer.size();    

                const auto &tris = prim->tris;
                const auto trisAmount = tris.size();
                std::vector<float> samplerData(3 * 3 * trisAmount * frameAmount);
#pragma omp parallel for
                for (int i = 0; i < frameAmount; ++i)
                {
#pragma omp parallel for
                    for (int j = 0; j < trisAmount; ++j)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            int l = ((i * trisAmount + j) * 3 + k) * 3;
                            const auto &data = vertexFrameBuffer[i][tris[j][k]];
                            samplerData[l + 0] = data[0];
                            samplerData[l + 1] = data[1];
                            samplerData[l + 2] = data[2];
                        }
                    }
                }

                prim_inst_vertex_frame_sampler = std::make_unique<Texture>();
                prim_inst_vertex_frame_sampler->target = GL_TEXTURE_2D;
                prim_inst_vertex_frame_sampler->wrap_s = GL_CLAMP_TO_EDGE;
                prim_inst_vertex_frame_sampler->wrap_t = GL_CLAMP_TO_EDGE;
                prim_inst_vertex_frame_sampler->min_filter = GL_LINEAR;
                prim_inst_vertex_frame_sampler->mag_filter = GL_LINEAR;
                prim_inst_vertex_frame_sampler->internal_fmt = GL_RGB32F;
                prim_inst_vertex_frame_sampler->format = GL_RGB;
                prim_inst_vertex_frame_sampler->dtype = GL_FLOAT;
                prim_inst_vertex_frame_sampler->bind_image(samplerData.data(), 3 * trisAmount, frameAmount);

                triObj.instvbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
                std::vector<InstVboData> vboData(amount);
                const auto &modelMatrices = inst->modelMatrices;
                const auto &timeList = inst->timeList;
#pragma omp parallel for
                for (int i = 0; i < amount; ++i)
                {
                    auto &instVboData = vboData[i];
                    instVboData.modelMatrix = modelMatrices[i];
                    instVboData.time = timeList[i];
                }
                triObj.instvbo->bind_data(vboData.data(), amount * sizeof(InstVboData));
            }
        }

        bool findCamera=false;
        triObj.prog = get_tris_program(path, prim->mtl, prim->inst);
        if(prim->mtl!=nullptr){
            triObj.voxelprog = get_voxelize_program(prim->mtl, prim->inst);
            triObj.shadowprog = get_shadow_program(prim->mtl, prim->inst);
            auto code = prim->mtl->frag;
            if(code.find("mat_reflection = float(float(1))")!=std::string::npos)
            {
                glm::vec3 c=glm::vec3(0);
                for(auto v:prim->verts)
                {
                    c+=glm::vec3(v[0], v[1], v[2]);
                }
                c = c/(prim->verts.size() + 0.000001f);
                auto n=prim->attr<zeno::vec3f>("nrm")[0];
                auto idpos = code.find("mat_reflectID");
                auto idstr0 = code.substr(idpos + 28, 1);
                auto idstr1 = code.substr(idpos + 29, 1);
                std::string num;
                if(idstr1!=")")
                    num = idstr0+idstr1;
                else
                    num = idstr0;
                auto refID = std::atoi(num.c_str());
                setReflectivePlane(refID, glm::vec3(n[0], n[1], n[2]), c);
            }
            if(code.find("mat_isCamera = float(float(1))")!=std::string::npos)
            {
                auto pos = prim->attr<zeno::vec3f>("pos")[0];
                auto up = prim->attr<zeno::vec3f>("nrm")[0];
                auto view = prim->attr<zeno::vec3f>("clr")[0];
                auto fov = prim->attr<zeno::vec3f>("uv")[0][0];
                auto dof = prim->attr<zeno::vec3f>("uv")[0][1];
                auto ffar = prim->attr<zeno::vec3f>("uv")[0][2];
                setCamera(glm::vec3(pos[0],pos[1],pos[2]),
                          glm::vec3(view[0],view[1],view[2]),
                          glm::vec3(up[0],up[1],up[2]),
                          fov, 0.1,ffar, dof, 1);
                
            }
            if(code.find("mat_isVoxelDomain = float(float(1))")!=std::string::npos)
            {
                auto origin = prim->attr<zeno::vec3f>("pos")[0];
                auto right = prim->attr<zeno::vec3f>("pos")[1] - prim->attr<zeno::vec3f>("pos")[0];
                auto up = prim->attr<zeno::vec3f>("pos")[3] - prim->attr<zeno::vec3f>("pos")[0];

                voxelizer::setVoxelizeView(glm::vec3(origin[0],origin[1],origin[2]), 
                                           glm::vec3(right[0], right[1], right[2]), 
                                           glm::vec3(up[0], up[1], up[2]));
                
            }
            
        }
        if(!triObj.prog){
            triObj.prog = get_tris_program(path,nullptr,nullptr);
        }
        
    }

    draw_all_points = !points_count && !lines_count && !tris_count;
    if (draw_all_points) {
        pointObj.prog = get_points_program(path);
    }

    if ((prim->mtl != nullptr) && !prim->mtl->tex2Ds.empty())
    {
      load_texture2Ds(prim->mtl->tex2Ds);
    }
    //load_textures(path);
    prim_has_mtl = (prim->mtl != nullptr) && triObj.prog && triObj.shadowprog;
  }
  
  virtual void drawShadow(Light *light) override 
  {
    if(!prim_has_mtl)
        return;
    int id = 0;
    int idx = 0;
    for (id = 0; id < 64; id++) {
        if(textures.find(id)!=textures.end())
        {
            if(g_Textures.find(textures[id])!=g_Textures.end())
            {
                g_Textures[textures[id]]->bind_to(idx);
                idx++;
            }
        }
    }
    auto vbobind = [&] (auto &vbo) {
        vbo->bind();
        vbo->attribute(/*index=*/0,
            /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/1,
            /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/2,
            /*offset=*/sizeof(float) * 6, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/3,
            /*offset=*/sizeof(float) * 9, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/4,
            /*offset=*/sizeof(float) * 12, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
    };
    auto vbounbind = [&] (auto &vbo) {
        vbo->disable_attribute(0);
        vbo->disable_attribute(1);
        vbo->disable_attribute(2);
        vbo->disable_attribute(3);
        vbo->disable_attribute(4);
        vbo->unbind();
    };

    auto instvbobind = [&] (auto &instvbo) {
        instvbo->bind();
        instvbo->attribute(5, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 0, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(6, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 1, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(7, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 2, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(8, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 3, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(9, offsetof(InstVboData, time), sizeof(InstVboData), GL_FLOAT, 1);
        instvbo->attrib_divisor(5, 1);
        instvbo->attrib_divisor(6, 1);
        instvbo->attrib_divisor(7, 1);
        instvbo->attrib_divisor(8, 1);
        instvbo->attrib_divisor(9, 1);
    };
    auto instvbounbind = [&] (auto &instvbo) {
        instvbo->disable_attribute(5);
        instvbo->disable_attribute(6);
        instvbo->disable_attribute(7);
        instvbo->disable_attribute(8);
        instvbo->disable_attribute(9);
        instvbo->unbind();
    };

    if (tris_count) {
        //printf("TRIS\n");
        if (triObj.vbo) {
            vbobind(triObj.vbo);
        } else {
            vbobind(vbo);
        }

        if (prim_has_inst) {
            if (triObj.instvbo) {
                instvbobind(triObj.instvbo);
            } else {
                instvbobind(instvbo);
            }
        }

        triObj.shadowprog->use();
        int texOcp = 0;
        light->setShadowMV(triObj.shadowprog);

        if (prim_has_inst)
        {
            triObj.shadowprog->set_uniform("fInstDeltaTime", prim_inst_delta_time);
            triObj.shadowprog->set_uniformi("iInstFrameAmount", prim_inst_frame_amount);
            triObj.shadowprog->set_uniformi("iInstVertexFrameSampler",texOcp);
            prim_inst_vertex_frame_sampler->bind_to(texOcp);
            texOcp++;
        }

        if (prim_has_mtl) {
            const int &texsSize = textures.size();
            for (int texId=0; texId < texsSize; ++ texId)
            {
                std::string texName = "zenotex" + std::to_string(texId);
                triObj.shadowprog->set_uniformi(texName.c_str(), texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
                CHECK_GL(glBindTexture(g_Textures[textures[texId]]->target, g_Textures[textures[texId]]->tex));
                texOcp++;
            }
        }
        triObj.ebo->bind();

        if (prim_has_inst)
        {
            CHECK_GL(glDrawElementsInstancedARB(GL_TRIANGLES, /*count=*/triObj.count * 3,
                GL_UNSIGNED_INT, /*first=*/0, prim_inst_amount));
        }
        else
        {
            CHECK_GL(glDrawElements(GL_TRIANGLES, /*count=*/triObj.count * 3,
                GL_UNSIGNED_INT, /*first=*/0));
        }
        
        triObj.ebo->unbind();
        if (triObj.vbo) {
            vbounbind(triObj.vbo);
        } else {
            vbounbind(vbo);
        }

        if (prim_has_inst) {
            if (triObj.instvbo) {
                instvbounbind(triObj.instvbo);
            } else {
                instvbounbind(instvbo);
            }
        }
    }

  }
  virtual void drawVoxelize(float alphaPass) override {
    if (!prim_has_mtl) return;
    if (prim_has_mtl) ensureGlobalMapExist();
    int id = 0;
    int idx = 0;
    for (id = 0; id < 64; id++) {
        if(textures.find(id)!=textures.end())
        {
            if(g_Textures.find(textures[id])!=g_Textures.end())
            {
                g_Textures[textures[id]]->bind_to(idx);
                idx++;
            }
        }
    }

    auto vbobind = [&] (auto &vbo) {
        vbo->bind();
        vbo->attribute(/*index=*/0,
            /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/1,
            /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/2,
            /*offset=*/sizeof(float) * 6, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/3,
            /*offset=*/sizeof(float) * 9, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/4,
            /*offset=*/sizeof(float) * 12, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
    };
    auto vbounbind = [&] (auto &vbo) {
        vbo->disable_attribute(0);
        vbo->disable_attribute(1);
        vbo->disable_attribute(2);
        vbo->disable_attribute(3);
        vbo->disable_attribute(4);
        vbo->unbind();
    };

    auto instvbobind = [&] (auto &instvbo) {
        instvbo->bind();
        instvbo->attribute(5, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 0, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(6, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 1, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(7, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 2, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(8, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 3, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(9, offsetof(InstVboData, time), sizeof(InstVboData), GL_FLOAT, 1);
        instvbo->attrib_divisor(5, 1);
        instvbo->attrib_divisor(6, 1);
        instvbo->attrib_divisor(7, 1);
        instvbo->attrib_divisor(8, 1);
        instvbo->attrib_divisor(9, 1);
    };
    auto instvbounbind = [&] (auto &instvbo) {
        instvbo->disable_attribute(5);
        instvbo->disable_attribute(6);
        instvbo->disable_attribute(7);
        instvbo->disable_attribute(8);
        instvbo->disable_attribute(9);
        instvbo->unbind();
    };
    
    if (tris_count) {
        //printf("TRIS\n");
        if (triObj.vbo) {
            vbobind(triObj.vbo);
        } else {
            vbobind(vbo);
        }

        if (prim_has_inst) {
            if (triObj.instvbo) {
                instvbobind(triObj.instvbo);
            } else {
                instvbobind(instvbo);
            }
        }
        auto &scene = Scene::getInstance();
        auto &lights = scene.lights;
        if (LightMatrixUBO == 0)
        {
            CHECK_GL(glGenBuffers(1, &LightMatrixUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, LightMatrixUBO));
            CHECK_GL(glBufferData(GL_UNIFORM_BUFFER, sizeof(glm::mat4x4) * 128, nullptr, GL_STATIC_DRAW));
            CHECK_GL(glBindBufferBase(GL_UNIFORM_BUFFER, 0, LightMatrixUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
        }
        glBindBuffer(GL_UNIFORM_BUFFER, LightMatrixUBO);
        //std::cout<<"                        "<<LightMatrixUBO<<std::endl;
        for (int lightNo = 0; lightNo < lights.size(); ++lightNo)
        {
            auto &light = lights[lightNo];
            
            auto matrices = light->lightSpaceMatrices;
            for (size_t i = 0; i < matrices.size(); ++i)
            {
                glBufferSubData(GL_UNIFORM_BUFFER, (lightNo * (Light::cascadeCount + 1) + i) * sizeof(glm::mat4x4), sizeof(glm::mat4x4), &matrices[i]);
            }
            
        }
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        triObj.voxelprog->use();
        set_program_uniforms(triObj.voxelprog);
        triObj.voxelprog->set_uniform("u_scene_voxel_scale", glm::vec3(1.0/voxelizer::getDomainLength()));
        triObj.voxelprog->set_uniform("m_gi_emission_base", get_gi_emission_base());
        triObj.voxelprog->set_uniform("voxelgrid_resolution", voxelizer::getVoxelResolution());
        triObj.voxelprog->set_uniformi("lightNum", lights.size());
        triObj.voxelprog->set_uniform("alphaPass", alphaPass);
        triObj.voxelprog->set_uniform("vxView", voxelizer::getView());
        triObj.voxelprog->set_uniform("vxMaterialPass", voxelizer::isMaterialPass);
        
        for (int lightNo = 0; lightNo < lights.size(); ++lightNo)
        {
            auto &light = lights[lightNo];
            auto name = "light[" + std::to_string(lightNo) + "]";
            triObj.voxelprog->set_uniform(name.c_str(), light->lightDir);
        }

        

        if (prim_has_mtl) {
            const int &texsSize = textures.size();
            int texOcp=0;
            for (int texId=0; texId < texsSize; ++ texId)
            {
                std::string texName = "zenotex" + std::to_string(texId);
                triObj.voxelprog->set_uniformi(texName.c_str(), texId);
                CHECK_GL(glActiveTexture(GL_TEXTURE0+texId));
                CHECK_GL(glBindTexture(g_Textures[textures[texId]]->target, g_Textures[textures[texId]]->tex));
                texOcp++;
            }

            triObj.voxelprog->set_uniformi("irradianceMap",texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            if (auto irradianceMap = getIrradianceMap(); irradianceMap != (unsigned int)-1)
                CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap));
            texOcp++;

            triObj.voxelprog->set_uniformi("prefilterMap",texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            if (auto prefilterMap = getPrefilterMap(); prefilterMap != (unsigned int)-1)
                CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap));
            texOcp++;

            triObj.voxelprog->set_uniformi("brdfLUT",texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            if (auto brdfLUT = getBRDFLut(); brdfLUT != (unsigned int)-1)
                CHECK_GL(glBindTexture(GL_TEXTURE_2D, brdfLUT));
            texOcp++;
            
            triObj.voxelprog->set_uniformi("vxNormal", texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            CHECK_GL(glBindTexture(GL_TEXTURE_3D, voxelizer::vxNormal.id));
            texOcp++;
            


            
            triObj.voxelprog->set_uniform("farPlane", getCamFar());
            triObj.voxelprog->set_uniformi("cascadeCount", Light::cascadeCount);
            for (int lightNo = 0; lightNo < lights.size(); ++lightNo)
            {
                auto &light = lights[lightNo];
                auto name = "lightDir[" + std::to_string(lightNo) + "]";
                triObj.voxelprog->set_uniform(name.c_str(), light->lightDir);
                name = "shadowTint[" + std::to_string(lightNo) + "]";
                triObj.voxelprog->set_uniform(name.c_str(), light->getShadowTint());
                name = "shadowSoftness[" + std::to_string(lightNo) + "]";
                triObj.voxelprog->set_uniform(name.c_str(), light->shadowSoftness);
                name = "lightIntensity[" + std::to_string(lightNo) + "]";
                triObj.voxelprog->set_uniform(name.c_str(), light->getIntensity());
                for (size_t i = 0; i < Light::cascadeCount + 1; i++)
                {
                    auto name1 = "near[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                    triObj.voxelprog->set_uniform(name1.c_str(), light->m_nearPlane[i]);

                    auto name2 = "far[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                    triObj.voxelprog->set_uniform(name2.c_str(), light->m_farPlane[i]);

                    auto name = "shadowMap[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                    triObj.voxelprog->set_uniformi(name.c_str(), texOcp);
                    CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                    if (auto shadowMap = light->DepthMaps[i]; shadowMap != (unsigned int)-1)
                        CHECK_GL(glBindTexture(GL_TEXTURE_2D, shadowMap));
                    texOcp++;
                }
                for (size_t i = 0; i < Light::cascadeCount; ++i)
                {
                    auto name = "cascadePlaneDistances[" + std::to_string(lightNo * Light::cascadeCount + i) + "]";
                    triObj.voxelprog->set_uniform(name.c_str(), light->shadowCascadeLevels[i]);
                }
                name = "lview[" + std::to_string(lightNo) + "]";
                triObj.voxelprog->set_uniform(name.c_str(), light->lightMV);

                // auto matrices = light->lightSpaceMatrices;
                // for (size_t i = 0; i < matrices.size(); i++)
                // {
                //     auto name = "lightSpaceMatrices[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                //     triObj.voxelprog->set_uniform(name.c_str(), matrices[i]);
                // }
            }

            
            
    
        }
        
        
        triObj.ebo->bind();

        if (prim_has_inst)
        {
            CHECK_GL(glDrawElementsInstancedARB(GL_TRIANGLES, /*count=*/triObj.count * 3,
                GL_UNSIGNED_INT, /*first=*/0, prim_inst_amount));
        }
        else
        {
            CHECK_GL(glDrawElements(GL_TRIANGLES, /*count=*/triObj.count * 3,
                GL_UNSIGNED_INT, /*first=*/0));
        }

        triObj.ebo->unbind();
        if (triObj.vbo) {
            vbounbind(triObj.vbo);
        } else {
            vbounbind(vbo);
        }

        if (prim_has_inst) {
            if (triObj.instvbo) {
                instvbounbind(triObj.instvbo);
            } else {
                instvbounbind(instvbo);
            }
        }       

    }
  }
  virtual void draw(bool reflect, float depthPass) override {
      if (prim_has_mtl) ensureGlobalMapExist();

    int id = 0;
    int idx = 0;
    for (id = 0; id < 64; id++) {
        if(textures.find(id)!=textures.end())
        {
            if(g_Textures.find(textures[id])!=g_Textures.end())
            {
                g_Textures[textures[id]]->bind_to(idx);
                idx++;
            }
        }
    }

    auto vbobind = [&] (auto &vbo) {
        vbo->bind();
        vbo->attribute(/*index=*/0,
            /*offset=*/sizeof(float) * 0, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/1,
            /*offset=*/sizeof(float) * 3, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/2,
            /*offset=*/sizeof(float) * 6, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/3,
            /*offset=*/sizeof(float) * 9, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
        vbo->attribute(/*index=*/4,
            /*offset=*/sizeof(float) * 12, /*stride=*/sizeof(float) * 15,
            GL_FLOAT, /*count=*/3);
    };
    auto vbounbind = [&] (auto &vbo) {
        vbo->disable_attribute(0);
        vbo->disable_attribute(1);
        vbo->disable_attribute(2);
        vbo->disable_attribute(3);
        vbo->disable_attribute(4);
        vbo->unbind();
    };

    auto instvbobind = [&] (auto &instvbo) {
        instvbo->bind();
        instvbo->attribute(5, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 0, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(6, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 1, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(7, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 2, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(8, offsetof(InstVboData, modelMatrix) + sizeof(glm::vec4) * 3, sizeof(InstVboData), GL_FLOAT, 4);
        instvbo->attribute(9, offsetof(InstVboData, time), sizeof(InstVboData), GL_FLOAT, 1);
        instvbo->attrib_divisor(5, 1);
        instvbo->attrib_divisor(6, 1);
        instvbo->attrib_divisor(7, 1);
        instvbo->attrib_divisor(8, 1);
        instvbo->attrib_divisor(9, 1);
    };
    auto instvbounbind = [&] (auto &instvbo) {
        instvbo->disable_attribute(5);
        instvbo->disable_attribute(6);
        instvbo->disable_attribute(7);
        instvbo->disable_attribute(8);
        instvbo->disable_attribute(9);
        instvbo->unbind();
    };

    if (draw_all_points || points_count)
        vbobind(vbo);

    if (draw_all_points) {
        //printf("ALLPOINTS\n");
        pointObj.prog->use();
        set_program_uniforms(pointObj.prog);
        CHECK_GL(glDrawArrays(GL_POINTS, /*first=*/0, /*count=*/vertex_count));
    }

    if (points_count) {
        //printf("POINTS\n");
        pointObj.prog->use();
        set_program_uniforms(pointObj.prog);
        pointObj.ebo->bind();
        CHECK_GL(glDrawElements(GL_POINTS, /*count=*/pointObj.count * 1,
              GL_UNSIGNED_INT, /*first=*/0));
        pointObj.ebo->unbind();
    }

    if (draw_all_points || points_count)
        vbounbind(vbo);


    if (lines_count) {
        //printf("LINES\n");
        if (lineObj.vbo) {
            vbobind(lineObj.vbo);
        } else {
            vbobind(vbo);
        }
        lineObj.prog->use();
        set_program_uniforms(lineObj.prog);
        lineObj.ebo->bind();
        CHECK_GL(glDrawElements(GL_LINES, /*count=*/lineObj.count * 2,
              GL_UNSIGNED_INT, /*first=*/0));
        lineObj.ebo->unbind();
        if (lineObj.vbo) {
            vbounbind(lineObj.vbo);
        } else {
            vbounbind(vbo);
        }
    }

    if (tris_count) {
        //printf("TRIS\n");
        if (triObj.vbo) {
            vbobind(triObj.vbo);
        } else {
            vbobind(vbo);
        }

        if (prim_has_inst) {
            if (triObj.instvbo) {
                instvbobind(triObj.instvbo);
            } else {
                instvbobind(instvbo);
            }
        }
        auto &scene = Scene::getInstance();
        auto &lights = scene.lights;
        if (LightMatrixUBO == 0)
        {
            CHECK_GL(glGenBuffers(1, &LightMatrixUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, LightMatrixUBO));
            CHECK_GL(glBufferData(GL_UNIFORM_BUFFER, sizeof(glm::mat4x4) * 128, nullptr, GL_STATIC_DRAW));
            CHECK_GL(glBindBufferBase(GL_UNIFORM_BUFFER, 0, LightMatrixUBO));
            CHECK_GL(glBindBuffer(GL_UNIFORM_BUFFER, 0));
        }
        glBindBuffer(GL_UNIFORM_BUFFER, LightMatrixUBO);
        //std::cout<<"                        "<<LightMatrixUBO<<std::endl;
        for (int lightNo = 0; lightNo < lights.size(); ++lightNo)
        {
            auto &light = lights[lightNo];
            
            auto matrices = light->lightSpaceMatrices;
            for (size_t i = 0; i < matrices.size(); ++i)
            {
                glBufferSubData(GL_UNIFORM_BUFFER, (lightNo * (Light::cascadeCount + 1) + i) * sizeof(glm::mat4x4), sizeof(glm::mat4x4), &matrices[i]);
            }
            
        }
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        triObj.prog->use();
        int texOcp=0;
        set_program_uniforms(triObj.prog);

        if (prim_has_inst)
        {
            triObj.prog->set_uniform("fInstDeltaTime", prim_inst_delta_time);
            triObj.prog->set_uniformi("iInstFrameAmount", prim_inst_frame_amount);
            triObj.prog->set_uniformi("iInstVertexFrameSampler",texOcp);
            prim_inst_vertex_frame_sampler->bind_to(texOcp);
            texOcp++;
        }

        
        triObj.prog->set_uniformi("lightNum", lights.size());
        for (int lightNo = 0; lightNo < lights.size(); ++lightNo)
        {
            auto &light = lights[lightNo];
            auto name = "light[" + std::to_string(lightNo) + "]";
            triObj.prog->set_uniform(name.c_str(), light->lightDir);
        }

        triObj.prog->set_uniformi("mRenderWireframe", false);

        if (prim_has_mtl) {
            const int &texsSize = textures.size();
            for (int texId=0; texId < texsSize; ++ texId)
            {
                std::string texName = "zenotex" + std::to_string(texId);
                triObj.prog->set_uniformi(texName.c_str(), texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
                CHECK_GL(glBindTexture(g_Textures[textures[texId]]->target, g_Textures[textures[texId]]->tex));
                texOcp++;
            }
            triObj.prog->set_uniformi("skybox",texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            if (auto envmap = getGlobalEnvMap(); envmap != (unsigned int)-1)
                CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, envmap));
            texOcp++;

            triObj.prog->set_uniformi("irradianceMap",texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            if (auto irradianceMap = getIrradianceMap(); irradianceMap != (unsigned int)-1)
                CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap));
            texOcp++;

            triObj.prog->set_uniformi("prefilterMap",texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            if (auto prefilterMap = getPrefilterMap(); prefilterMap != (unsigned int)-1)
                CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap));
            texOcp++;

            triObj.prog->set_uniformi("brdfLUT",texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            if (auto brdfLUT = getBRDFLut(); brdfLUT != (unsigned int)-1)
                CHECK_GL(glBindTexture(GL_TEXTURE_2D, brdfLUT));
            texOcp++;

            


            
            triObj.prog->set_uniform("farPlane", getCamFar());
            triObj.prog->set_uniformi("cascadeCount", Light::cascadeCount);
            for (int lightNo = 0; lightNo < lights.size(); ++lightNo)
            {
                auto &light = lights[lightNo];
                auto name = "lightDir[" + std::to_string(lightNo) + "]";
                triObj.prog->set_uniform(name.c_str(), light->lightDir);
                name = "shadowTint[" + std::to_string(lightNo) + "]";
                triObj.prog->set_uniform(name.c_str(), light->getShadowTint());
                name = "shadowSoftness[" + std::to_string(lightNo) + "]";
                triObj.prog->set_uniform(name.c_str(), light->shadowSoftness);
                name = "lightIntensity[" + std::to_string(lightNo) + "]";
                triObj.prog->set_uniform(name.c_str(), light->getIntensity());
                for (size_t i = 0; i < Light::cascadeCount + 1; i++)
                {
                    auto name1 = "near[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                    triObj.prog->set_uniform(name1.c_str(), light->m_nearPlane[i]);

                    auto name2 = "far[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                    triObj.prog->set_uniform(name2.c_str(), light->m_farPlane[i]);

                    auto name = "shadowMap[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                    triObj.prog->set_uniformi(name.c_str(), texOcp);
                    CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                    if (auto shadowMap = light->DepthMaps[i]; shadowMap != (unsigned int)-1)
                        CHECK_GL(glBindTexture(GL_TEXTURE_2D, shadowMap));
                    texOcp++;
                }
                for (size_t i = 0; i < Light::cascadeCount; ++i)
                {
                    auto name = "cascadePlaneDistances[" + std::to_string(lightNo * Light::cascadeCount + i) + "]";
                    triObj.prog->set_uniform(name.c_str(), light->shadowCascadeLevels[i]);
                }
                name = "lview[" + std::to_string(lightNo) + "]";
                triObj.prog->set_uniform(name.c_str(), light->lightMV);

                // auto matrices = light->lightSpaceMatrices;
                // for (size_t i = 0; i < matrices.size(); i++)
                // {
                //     auto name = "lightSpaceMatrices[" + std::to_string(lightNo * (Light::cascadeCount + 1) + i) + "]";
                //     triObj.prog->set_uniform(name.c_str(), matrices[i]);
                // }
                
            }

            if(reflect)
            {
                triObj.prog->set_uniform("reflectPass", 1.0f);
            }
            else {
                triObj.prog->set_uniform("reflectPass",0.0f);
            }
            triObj.prog->set_uniform("reflectionViewID", (float)getReflectionViewID());
            for(int i=0;i<16;i++)
            {
                if(!renderReflect(i))
                    continue;
                auto name = "reflectMVP[" + std::to_string(i) + "]";
                triObj.prog->set_uniform(name.c_str(), getReflectMVP(i));
                name = "reflect_normals[" + std::to_string(i) + "]";
                triObj.prog->set_uniform(name.c_str(), getReflectiveNormal(i));
                name = "reflect_centers[" + std::to_string(i) + "]";
                triObj.prog->set_uniform(name.c_str(), getReflectiveCenter(i));
                auto name2 = "reflectionMap"+std::to_string(i);
                triObj.prog->set_uniformi(name2.c_str(),texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
                CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, getReflectMaps()[i]));
                texOcp++;
            }
            triObj.prog->set_uniform("depthPass", depthPass);
            triObj.prog->set_uniformi("depthBuffer", texOcp);
            CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, getDepthTexture()));
            texOcp++;
    
        }
        
        triObj.prog->set_uniformi("vxgibuffer", texOcp);
        CHECK_GL(glActiveTexture(GL_TEXTURE0+texOcp));
        CHECK_GL(glBindTexture(GL_TEXTURE_3D, voxelizer::vxTexture.id));
        texOcp++;
        triObj.prog->set_uniform("vxSize",voxelizer::getDomainLength());
        triObj.prog->set_uniform("vxView", voxelizer::getView());
        triObj.prog->set_uniform("enable_gi_flag", zenvis::get_enable_gi());
        triObj.prog->set_uniform("m_gi_base", zenvis::get_gi_base());
        
        
        triObj.prog->set_uniform("msweight", m_weight);
        triObj.ebo->bind();

        if (prim_has_inst)
        {
            CHECK_GL(glDrawElementsInstancedARB(GL_TRIANGLES, /*count=*/triObj.count * 3,
                GL_UNSIGNED_INT, /*first=*/0, prim_inst_amount));
        }
        else
        {
            CHECK_GL(glDrawElements(GL_TRIANGLES, /*count=*/triObj.count * 3,
                GL_UNSIGNED_INT, /*first=*/0));
        }

        if (render_wireframe) {
          glEnable(GL_POLYGON_OFFSET_LINE);
          glPolygonOffset(1, 1);
          glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
          triObj.prog->set_uniformi("mRenderWireframe", true);
          if (prim_has_inst){
                CHECK_GL(glDrawElementsInstancedARB(GL_TRIANGLES, /*count=*/triObj.count * 3,
                    GL_UNSIGNED_INT, /*first=*/0, prim_inst_amount));
          }else
          {
                CHECK_GL(glDrawElements(GL_TRIANGLES, /*count=*/triObj.count * 3,
                    GL_UNSIGNED_INT, /*first=*/0));
          }
          glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
          glDisable(GL_POLYGON_OFFSET_LINE);
        }
        triObj.ebo->unbind();
        if (triObj.vbo) {
            vbounbind(triObj.vbo);
        } else {
            vbounbind(vbo);
        }

        if (prim_has_inst) {
            if (triObj.instvbo) {
                instvbounbind(triObj.instvbo);
            } else {
                instvbounbind(instvbo);
            }
        }       

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
      int t_idx = 0;
    for (const auto &tex2D : tex2Ds)
    {
        if(g_Textures.find(tex2D->path)!=g_Textures.end()){
            textures[t_idx] = tex2D->path;
            t_idx++;
            continue;
        }

      auto tex = std::make_shared<Texture>();

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

      tex->load(tex2D->path.c_str());
      g_Textures[tex2D->path] = tex;
      textures[t_idx] = tex2D->path;
      t_idx++;
    }
  }

};

std::unique_ptr<IGraphic> makeGraphicPrimitive
    ( zeno::PrimitiveObject *prim
    , std::string const &path
    ) {
  return std::make_unique<GraphicPrimitive>(prim, path);
}

}
