#include <memory>
#include <string>
#include <vector>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/InstancingObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/ticktock.h>
#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/texture.h>

namespace zenovis {
namespace {

using namespace opengl;

struct ZhxxDrawObject {
    std::unique_ptr<Buffer> vbo;
    std::unique_ptr<Buffer> ebo;
    std::unique_ptr<Buffer> instvbo;
    size_t count = 0;
    Program *prog{};
};

static void parsePointsDrawBuffer(zeno::PrimitiveObject *prim, ZhxxDrawObject &obj) {
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &uv = prim->attr<zeno::vec3f>("uv");
    auto const &tang = prim->attr<zeno::vec3f>("tang");
    obj.count = prim->size();

    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 5);
    for (int i = 0; i < obj.count; i++) {
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
        obj.ebo->bind_data(prim->points.data(),
                           points_count * sizeof(prim->points[0]));
    }
}

static void parseLinesDrawBuffer(zeno::PrimitiveObject *prim, ZhxxDrawObject &obj) {
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tang = prim->attr<zeno::vec3f>("tang");
    auto const &lines = prim->lines;
    bool has_uv = lines.has_attr("uv0") && lines.has_attr("uv1");
    obj.count = prim->lines.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 2 * 5);
    std::vector<zeno::vec2i> linesdata(obj.count);
#pragma omp parallel for
    for (int i = 0; i < obj.count; i++) {
        mem[10 * i + 0] = pos[lines[i][0]];
        mem[10 * i + 1] = clr[lines[i][0]];
        mem[10 * i + 2] = nrm[lines[i][0]];
        mem[10 * i + 3] =
            has_uv ? lines.attr<zeno::vec3f>("uv0")[i] : zeno::vec3f(0, 0, 0);
        mem[10 * i + 4] = tang[lines[i][0]];
        mem[10 * i + 5] = pos[lines[i][1]];
        mem[10 * i + 6] = clr[lines[i][1]];
        mem[10 * i + 7] = nrm[lines[i][1]];
        mem[10 * i + 8] =
            has_uv ? lines.attr<zeno::vec3f>("uv1")[i] : zeno::vec3f(0, 0, 0);
        mem[10 * i + 9] = tang[lines[i][1]];
        linesdata[i] = zeno::vec2i(i * 2, i * 2 + 1);
    }
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    if (obj.count) {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(linesdata[0]), obj.count * sizeof(linesdata[0]));
    }
}

static void computeTrianglesTangent(zeno::PrimitiveObject *prim) {
    const auto &tris = prim->tris;
    const auto &pos = prim->attr<zeno::vec3f>("pos");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto &tang = prim->tris.add_attr<zeno::vec3f>("tang");
    bool has_uv =
        tris.has_attr("uv0") && tris.has_attr("uv1") && tris.has_attr("uv2");
    //printf("!!has_uv = %d\n", has_uv);
#pragma omp parallel for
    for (size_t i = 0; i < prim->tris.size(); ++i) {
        if (has_uv) {
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

            auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] -
                             deltaUV1[0] * deltaUV0[1] + 1e-5);

            zeno::vec3f tangent;
            tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
            tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
            tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
            //printf("%f %f %f\n", tangent[0], tangent[1], tangent[3]);
            auto tanlen = zeno::length(tangent);
            tangent *(1.f / (tanlen + 1e-8));
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

static void parseTrianglesDrawBufferCompress(zeno::PrimitiveObject *prim, ZhxxDrawObject &obj) {
    //TICK(parse);
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tris = prim->tris;
    bool has_uv =
        tris.has_attr("uv0") && tris.has_attr("uv1") && tris.has_attr("uv2");
    auto &tang = prim->tris.attr<zeno::vec3f>("tang");
    std::vector<zeno::vec3f> pos1(pos.size());
    std::vector<zeno::vec3f> clr1(pos.size());
    std::vector<zeno::vec3f> nrm1(pos.size());
    std::vector<zeno::vec3f> uv1(pos.size());
    std::vector<zeno::vec3f> tang1(pos.size());
    std::vector<int> vertVisited(pos.size());
    std::vector<zeno::vec3i> tris1(tris.size());
    vertVisited.assign(pos.size(), 0);
    for (int i = 0; i < tris.size(); i++) {
        float area =
            zeno::length(zeno::cross(pos[tris[i][1]] - pos[tris[i][0]],
                                     pos[tris[i][2]] - pos[tris[i][0]]));
        for (int j = 0; j < 3; j++) {
            tang1[tris[i][j]] += area * tang[i];
        }
    }
    /* std::cout << "1111111111111111\n"; */
#pragma omp parallel for
    for (int i = 0; i < tang1.size(); i++) {
        tang1[i] = tang[i] / (zeno::length(tang[i]) + 0.000001);
    }
    /* std::cout << "2222222222222222\n"; */
    std::vector<int> issueTris(0);
    for (int i = 0; i < tris.size(); i++) {
        //if all verts not visited
        for (int j = 0; j < 3; j++) {
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
    /* std::cout << "3333333333333333333\n"; */

    //end compressed tri assign
    obj.count = tris1.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(pos1.size() * 5);
    std::vector<zeno::vec3i> trisdata(obj.count);
#pragma omp parallel for
    for (int i = 0; i < pos1.size(); i++) {
        mem[5 * i + 0] = pos1[i];
        mem[5 * i + 1] = clr1[i];
        mem[5 * i + 2] = nrm1[i];
        mem[5 * i + 3] = uv1[i];
        mem[5 * i + 4] = tang1[i];
    }
#pragma omp parallel for
    for (int i = 0; i < tris1.size(); i++) {
        trisdata[i] = tris1[i];
    }

    /* TICK(bindvbo); */
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    /* TOCK(bindvbo); */
    /* TICK(bindebo); */
    if (obj.count) {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(trisdata[0]), tris1.size() * sizeof(trisdata[0]));
    }
    /* TOCK(bindebo); */
}

static void parseTrianglesDrawBuffer(zeno::PrimitiveObject *prim, ZhxxDrawObject &obj) {
    /* TICK(parse); */
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tris = prim->tris;
    bool has_uv =
        tris.has_attr("uv0") && tris.has_attr("uv1") && tris.has_attr("uv2");
    obj.count = tris.size();
    obj.vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 3 * 5);
    std::vector<zeno::vec3i> trisdata(obj.count);
    auto &tang = prim->tris.attr<zeno::vec3f>("tang");
#pragma omp parallel for
    for (int i = 0; i < obj.count; i++) {
        mem[15 * i + 0] = pos[tris[i][0]];
        mem[15 * i + 1] = clr[tris[i][0]];
        mem[15 * i + 2] = nrm[tris[i][0]];
        mem[15 * i + 3] = has_uv ? tris.attr<zeno::vec3f>("uv0")[i]
                                 : zeno::vec3f(0.0f, 0.0f, 0.0f);
        mem[15 * i + 4] = tang[i];
        mem[15 * i + 5] = pos[tris[i][1]];
        mem[15 * i + 6] = clr[tris[i][1]];
        mem[15 * i + 7] = nrm[tris[i][1]];
        mem[15 * i + 8] = has_uv ? tris.attr<zeno::vec3f>("uv1")[i]
                                 : zeno::vec3f(0.0f, 0.0f, 0.0f);
        mem[15 * i + 9] = tang[i];
        mem[15 * i + 10] = pos[tris[i][2]];
        mem[15 * i + 11] = clr[tris[i][2]];
        mem[15 * i + 12] = nrm[tris[i][2]];
        mem[15 * i + 13] = has_uv ? tris.attr<zeno::vec3f>("uv2")[i]
                                  : zeno::vec3f(0.0f, 0.0f, 0.0f);
        mem[15 * i + 14] = tang[i];
        //std::cout<<tang[i][0]<<" "<<tang[i][1]<<" "<<tang[i][2]<<std::endl;
        trisdata[i] = zeno::vec3i(i * 3, i * 3 + 1, i * 3 + 2);
    }
    /* TOCK(parse); */

    /* TICK(bindvbo); */
    obj.vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    /* TOCK(bindvbo); */
    /* TICK(bindebo); */
    if (obj.count) {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(trisdata[0]), obj.count * sizeof(trisdata[0]));
    }
    /* TOCK(bindebo); */
}

struct ZhxxGraphicPrimitive final : IGraphicDraw {
    Scene *scene;
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

    ZhxxDrawObject pointObj;
    ZhxxDrawObject lineObj;
    ZhxxDrawObject triObj;
    std::vector<std::unique_ptr<Texture>> textures;
    bool prim_has_inst = false;
    int prim_inst_amount = 0;

    explicit ZhxxGraphicPrimitive(Scene *scene_, zeno::PrimitiveObject *prim)
        : scene(scene_) {
        zeno::log_trace("rendering primitive size {}", prim->size());

        prim_has_inst = prim->inst != nullptr;
        prim_inst_amount = prim->inst ? prim->inst->amount : 0;

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
        bool primNormalCorrect =
            prim->has_attr("nrm") &&
            length(prim->attr<zeno::vec3f>("nrm")[0]) > 1e-5;
        bool need_computeNormal =
            !primNormalCorrect || !(prim->has_attr("nrm"));
        if (prim->tris.size() && need_computeNormal) {
            /* std::cout << "computing normal\n"; */
            zeno::log_trace("computing normal");
            zeno::primCalcNormal(&*prim, 1);
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
        if (!prim->has_attr("uv")) {
            auto &uv = prim->add_attr<zeno::vec3f>("uv");
            for (size_t i = 0; i < uv.size(); i++) {
                uv[i] = zeno::vec3f(0.0f);
            }
        }
        if (!prim->has_attr("tang")) {
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
        for (int i = 0; i < vertex_count; i++) {
            mem[5 * i + 0] = pos[i];
            mem[5 * i + 1] = clr[i];
            mem[5 * i + 2] = nrm[i];
            mem[5 * i + 3] = uv[i];
            mem[5 * i + 4] = tang[i];
        }
        vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

        if (prim->inst != nullptr) {
            instvbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
            instvbo->bind_data(prim->inst->modelMatrices.data(),
                               prim->inst->modelMatrices.size() *
                                   sizeof(prim->inst->modelMatrices[0]));
        }

        points_count = prim->points.size();
        if (points_count) {
            pointObj.count = points_count;
            pointObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
            pointObj.ebo->bind_data(prim->points.data(),
                                    points_count * sizeof(prim->points[0]));
            pointObj.prog = get_points_program();
        }

        lines_count = prim->lines.size();
        if (lines_count) {
            // lines_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
            // lines_ebo->bind_data(prim->lines.data(), lines_count * sizeof(prim->lines[0]));
            // lines_prog = get_lines_program();
            if (!(prim->lines.has_attr("uv0") && prim->lines.has_attr("uv1"))) {
                lineObj.count = lines_count;
                lineObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
                lineObj.ebo->bind_data(prim->lines.data(),
                                       lines_count * sizeof(prim->lines[0]));
                lineObj.vbo = nullptr;
            } else {
                parseLinesDrawBuffer(&*prim, lineObj);
            }
            lineObj.prog = get_lines_program();
        }

        tris_count = prim->tris.size();
        if (tris_count) {
            if (!(prim->tris.has_attr("uv0") && prim->tris.has_attr("uv1") &&
                  prim->tris.has_attr("uv2"))) {
                triObj.count = tris_count;
                triObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
                triObj.ebo->bind_data(prim->tris.data(),
                                      tris_count * sizeof(prim->tris[0]));
                triObj.vbo = nullptr;
            } else {
                computeTrianglesTangent(&*prim);
                parseTrianglesDrawBuffer(&*prim, triObj);
            }

            if (prim->inst != nullptr) {
                triObj.instvbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
                triObj.instvbo->bind_data(
                    prim->inst->modelMatrices.data(),
                    prim->inst->modelMatrices.size() *
                        sizeof(prim->inst->modelMatrices[0]));
            }

            bool findCamera = false;
            triObj.prog = get_tris_program();
        }

        draw_all_points = !points_count && !lines_count && !tris_count;
        if (draw_all_points) {
            pointObj.prog = get_points_program();
        }
    }

    virtual void draw() override {
        int id = 0;
        for (id = 0; id < textures.size(); id++) {
            textures[id]->bind_to(id);
        }

        auto vbobind = [&](auto &vbo) {
            vbo->bind();
            vbo->attribute(/*index=*/0,
                           /*offset=*/sizeof(float) * 0,
                           /*stride=*/sizeof(float) * 15, GL_FLOAT,
                           /*count=*/3);
            vbo->attribute(/*index=*/1,
                           /*offset=*/sizeof(float) * 3,
                           /*stride=*/sizeof(float) * 15, GL_FLOAT,
                           /*count=*/3);
            vbo->attribute(/*index=*/2,
                           /*offset=*/sizeof(float) * 6,
                           /*stride=*/sizeof(float) * 15, GL_FLOAT,
                           /*count=*/3);
            vbo->attribute(/*index=*/3,
                           /*offset=*/sizeof(float) * 9,
                           /*stride=*/sizeof(float) * 15, GL_FLOAT,
                           /*count=*/3);
            vbo->attribute(/*index=*/4,
                           /*offset=*/sizeof(float) * 12,
                           /*stride=*/sizeof(float) * 15, GL_FLOAT,
                           /*count=*/3);
        };
        auto vbounbind = [&](auto &vbo) {
            vbo->disable_attribute(0);
            vbo->disable_attribute(1);
            vbo->disable_attribute(2);
            vbo->disable_attribute(3);
            vbo->disable_attribute(4);
            vbo->unbind();
        };

        auto instvbobind = [&](auto &instvbo) {
            instvbo->bind();
            instvbo->attribute(5, sizeof(glm::vec4) * 0, sizeof(glm::vec4) * 4,
                               GL_FLOAT, 4);
            instvbo->attribute(6, sizeof(glm::vec4) * 1, sizeof(glm::vec4) * 4,
                               GL_FLOAT, 4);
            instvbo->attribute(7, sizeof(glm::vec4) * 2, sizeof(glm::vec4) * 4,
                               GL_FLOAT, 4);
            instvbo->attribute(8, sizeof(glm::vec4) * 3, sizeof(glm::vec4) * 4,
                               GL_FLOAT, 4);
            instvbo->attrib_divisor(5, 1);
            instvbo->attrib_divisor(6, 1);
            instvbo->attrib_divisor(7, 1);
            instvbo->attrib_divisor(8, 1);
        };
        auto instvbounbind = [&](auto &instvbo) {
            instvbo->disable_attribute(5);
            instvbo->disable_attribute(6);
            instvbo->disable_attribute(7);
            instvbo->disable_attribute(8);
            instvbo->unbind();
        };

        if (draw_all_points || points_count)
            vbobind(vbo);

        if (draw_all_points) {
            //printf("ALLPOINTS\n");
            pointObj.prog->use();
            float point_scale = 21.6f / std::tan(scene->camera->m_fov * 0.5f * 3.1415926f / 180.0f);
            pointObj.prog->set_uniform("mPointScale", point_scale);
            scene->camera->set_program_uniforms(pointObj.prog);
            CHECK_GL(glDrawArrays(GL_POINTS, /*first=*/0, /*count=*/vertex_count));
        }

        if (points_count) {
            //printf("POINTS\n");
            pointObj.prog->use();
            scene->camera->set_program_uniforms(pointObj.prog);
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
            scene->camera->set_program_uniforms(lineObj.prog);
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

            triObj.prog->use();
            scene->camera->set_program_uniforms(triObj.prog);

            triObj.prog->set_uniform("mSmoothShading", scene->drawOptions->smooth_shading);
            triObj.prog->set_uniform("mNormalCheck", scene->drawOptions->normal_check);

            triObj.prog->set_uniformi("mRenderWireframe", false);

            triObj.ebo->bind();

            if (prim_has_inst) {
                CHECK_GL(glDrawElementsInstanced(
                        GL_TRIANGLES, /*count=*/triObj.count * 3,
                        GL_UNSIGNED_INT, /*first=*/0, prim_inst_amount));
            } else {
                CHECK_GL(glDrawElements(GL_TRIANGLES,
                                        /*count=*/triObj.count * 3,
                                        GL_UNSIGNED_INT, /*first=*/0));
            }

            if (scene->drawOptions->render_wireframe) {
                CHECK_GL(glEnable(GL_POLYGON_OFFSET_LINE));
                CHECK_GL(glPolygonOffset(0, 0));
                CHECK_GL(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
                triObj.prog->set_uniformi("mRenderWireframe", true);
                if (prim_has_inst) {
                    CHECK_GL(glDrawElementsInstanced(
                        GL_TRIANGLES, /*count=*/triObj.count * 3,
                        GL_UNSIGNED_INT, /*first=*/0, prim_inst_amount));
                } else {
                    CHECK_GL(glDrawElements(GL_TRIANGLES,
                                            /*count=*/triObj.count * 3,
                                            GL_UNSIGNED_INT, /*first=*/0));
                }
                CHECK_GL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
                CHECK_GL(glDisable(GL_POLYGON_OFFSET_LINE));
            }
            triObj.ebo->unbind();
            if (triObj.vbo) {
                vbounbind(triObj.vbo);
            } else {
                vbounbind(vbo);
            }
        }
    }

    Program *get_points_program() {
        auto vert = 
#include "shader/points.vert"
            ;
        auto frag = 
#include "shader/points.frag"
        ;

        return scene->shaderMan->compile_program(vert, frag);
    }

    Program *get_lines_program() {
        auto vert = 
#include "shader/lines.vert"
            ;
        auto frag =
#include "shader/lines.frag"
            ;

        return scene->shaderMan->compile_program(vert, frag);
    }

    Program *get_tris_program() {
        auto vert = prim_has_inst ?
#include "shader/tris.vert"
        :
#include "shader/tris_inst.vert"
        ;

        auto frag =
#include "shader/tris.frag"
        ;

        return scene->shaderMan->compile_program(vert, frag);
    }
};

}

void MakeGraphicVisitor::visit(zeno::PrimitiveObject *obj) {
     this->out_result = std::make_unique<ZhxxGraphicPrimitive>(this->in_scene, obj);
}

} // namespace zenovis
