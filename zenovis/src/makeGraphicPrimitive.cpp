#include <memory>
#include <string>
#include <vector>
#include <zeno/types/InstancingObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/TextureObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/ticktock.h>
#include <zeno/utils/vec.h>
#include <zenovis/Camera.h>
#include <zenovis/DepthPass.h>
#include <zenovis/EnvmapManager.h>
#include <zenovis/IGraphic.h>
#include <zenovis/Light.h>
#include <zenovis/ReflectivePass.h>
#include <zenovis/Scene.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/texture.h>
namespace zenovis {
using namespace opengl;

/* extern float getCamFar(); */
/* extern void ensureGlobalMapExist(); */
/* extern unsigned int getGlobalEnvMap(); */
/* extern unsigned int getIrradianceMap(); */
/* extern unsigned int getPrefilterMap(); */
/* extern unsigned int getBRDFLut(); */
/* extern glm::mat4 getReflectMVP(int i); */
/* extern std::vector<unsigned int> getReflectMaps(); */
/* extern void setReflectivePlane(int i, glm::vec3 n, glm::vec3 c); */
/* extern bool renderReflect(int i); */
/* extern int getReflectionViewID(); */
/* extern void setCamera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, double _fov, double fnear, double ffar, double _dof, int set); */
/* extern unsigned int getDepthTexture(); */

struct drawObject {
    std::unique_ptr<Buffer> vbo;
    std::unique_ptr<Buffer> ebo;
    std::unique_ptr<Buffer> instvbo;
    size_t count = 0;
    Program *prog;
    Program *shadowprog;
};
void parsePointsDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj) {
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
void parseLinesDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj) {
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

void computeTrianglesTangent(zeno::PrimitiveObject *prim) {
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
void parseTrianglesDrawBufferCompress(zeno::PrimitiveObject *prim,
                                      drawObject &obj) {
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
    std::cout << "1111111111111111\n";
#pragma omp parallel for
    for (int i = 0; i < tang1.size(); i++) {
        tang1[i] = tang[i] / (zeno::length(tang[i]) + 0.000001);
    }
    std::cout << "2222222222222222\n";
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
    std::cout << "3333333333333333333\n";

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
void parseTrianglesDrawBuffer(zeno::PrimitiveObject *prim, drawObject &obj) {
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
struct GraphicPrimitive : IGraphic {
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

    drawObject pointObj;
    drawObject lineObj;
    drawObject triObj;
    std::vector<std::unique_ptr<Texture>> textures;
    bool prim_has_mtl = false;
    bool prim_has_inst = false;
    int prim_inst_amount = 0;

    explicit GraphicPrimitive(Scene *scene_,
                              std::shared_ptr<zeno::PrimitiveObject> prim)
        : scene(scene_) {
        zeno::log_trace("rendering primitive size {}", prim->size());

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
            std::cout << "computing normal\n";
            zeno::primCalcNormal(prim.get(), 1);
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
                parseLinesDrawBuffer(prim.get(), lineObj);
            }
            lineObj.prog = get_lines_program();
        }

        tris_count = prim->tris.size();
        if (tris_count) {
            // tris_ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
            // tris_ebo->bind_data(prim->tris.data(), tris_count * sizeof(prim->tris[0]));
            // tris_prog = get_tris_program(path, prim->mtl);
            // if (!tris_prog)
            //     tris_prog = get_tris_program(nullptr);

            if (!(prim->tris.has_attr("uv0") && prim->tris.has_attr("uv1") &&
                  prim->tris.has_attr("uv2"))) {
                triObj.count = tris_count;
                triObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
                triObj.ebo->bind_data(prim->tris.data(),
                                      tris_count * sizeof(prim->tris[0]));
                triObj.vbo = nullptr;
            } else {
                computeTrianglesTangent(prim.get());
                parseTrianglesDrawBuffer(prim.get(), triObj);
            }

            if (prim->inst != nullptr) {
                triObj.instvbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
                triObj.instvbo->bind_data(
                    prim->inst->modelMatrices.data(),
                    prim->inst->modelMatrices.size() *
                        sizeof(prim->inst->modelMatrices[0]));
            }

            bool findCamera = false;
            triObj.prog = get_tris_program(prim->mtl, prim->inst);
            if (!triObj.prog)
                triObj.prog = get_tris_program(nullptr, nullptr);
            if (prim->mtl != nullptr) {
                triObj.shadowprog = get_shadow_program(prim->mtl, prim->inst);
                auto code = prim->mtl->frag;
                if (code.find("mat_reflection = float(float(1))") !=
                    std::string::npos) {
                    glm::vec3 c = glm::vec3(0);
                    for (auto v : prim->verts) {
                        c += glm::vec3(v[0], v[1], v[2]);
                    }
                    c = c / (prim->verts.size() + 0.000001f);
                    auto n = prim->attr<zeno::vec3f>("nrm")[0];
                    auto idpos = code.find("mat_reflectID");
                    auto idstr0 = code.substr(idpos + 28, 1);
                    auto idstr1 = code.substr(idpos + 29, 1);
                    std::string num;
                    if (idstr1 != ")")
                        num = idstr0 + idstr1;
                    else
                        num = idstr0;
                    auto refID = std::atoi(num.c_str());
                    scene->mReflectivePass->setReflectivePlane(
                        refID, glm::vec3(n[0], n[1], n[2]), c);
                }
                if (code.find("mat_isCamera = float(float(1))") !=
                    std::string::npos) {
                    auto pos = prim->attr<zeno::vec3f>("pos")[0];
                    auto up = prim->attr<zeno::vec3f>("nrm")[0];
                    auto view = prim->attr<zeno::vec3f>("clr")[0];
                    auto fov = prim->attr<zeno::vec3f>("uv")[0][0];
                    auto dof = prim->attr<zeno::vec3f>("uv")[0][1];
                    auto ffar = prim->attr<zeno::vec3f>("uv")[0][2];
                    scene->camera->setCamera(
                        glm::vec3(pos[0], pos[1], pos[2]),
                        glm::vec3(view[0], view[1], view[2]),
                        glm::vec3(up[0], up[1], up[2]), fov, 0.1, ffar, dof, 1);
                }
            }
            if (!triObj.prog) {
                triObj.prog = get_tris_program(nullptr, nullptr);
            }
        }

        draw_all_points = !points_count && !lines_count && !tris_count;
        if (draw_all_points) {
            pointObj.prog = get_points_program();
        }

        if ((prim->mtl != nullptr) && !prim->mtl->tex2Ds.empty()) {
            load_texture2Ds(prim->mtl->tex2Ds);
        }
        //load_textures(path);
        prim_has_mtl =
            (prim->mtl != nullptr) && triObj.prog && triObj.shadowprog;

        if (prim->inst != nullptr) {
            prim_has_inst = true;
            prim_inst_amount = prim->inst->amount;
        }
    }

    virtual void drawShadow(Light *light) override {
        if (!prim_has_mtl)
            return;
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
            light->setShadowMV(triObj.shadowprog);
            if (prim_has_mtl) {
                const int &texsSize = textures.size();
                for (int texId = 0; texId < texsSize; ++texId) {
                    std::string texName = "zenotex" + std::to_string(texId);
                    triObj.shadowprog->set_uniformi(texName.c_str(), texId);
                    CHECK_GL(glActiveTexture(GL_TEXTURE0 + texId));
                    CHECK_GL(glBindTexture(textures[texId]->target,
                                           textures[texId]->tex));
                }
            }
            triObj.ebo->bind();

            if (prim_has_inst) {
                CHECK_GL(glDrawElementsInstancedARB(
                    GL_TRIANGLES, /*count=*/triObj.count * 3, GL_UNSIGNED_INT,
                    /*first=*/0, prim_inst_amount));
            } else {
                CHECK_GL(glDrawElements(GL_TRIANGLES,
                                        /*count=*/triObj.count * 3,
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
        if (prim_has_mtl)
            scene->envmapMan->ensureGlobalMapExist();

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
            scene->camera->set_program_uniforms(pointObj.prog);
            CHECK_GL(
                glDrawArrays(GL_POINTS, /*first=*/0, /*count=*/vertex_count));
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

            auto &lights = scene->lights;
            triObj.prog->set_uniformi("lightNum", lights.size());
            for (int lightNo = 0; lightNo < lights.size(); ++lightNo) {
                auto &light = lights[lightNo];
                auto name = "light[" + std::to_string(lightNo) + "]";
                triObj.prog->set_uniform(name.c_str(), light->lightDir);
            }

            triObj.prog->set_uniformi("mRenderWireframe", false);

            if (prim_has_mtl) {
                const int &texsSize = textures.size();
                int texOcp = 0;
                for (int texId = 0; texId < texsSize; ++texId) {
                    std::string texName = "zenotex" + std::to_string(texId);
                    triObj.prog->set_uniformi(texName.c_str(), texId);
                    CHECK_GL(glActiveTexture(GL_TEXTURE0 + texId));
                    CHECK_GL(glBindTexture(textures[texId]->target,
                                           textures[texId]->tex));
                    texOcp++;
                }
                triObj.prog->set_uniformi("skybox", texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                if (auto envmap = scene->envmapMan->getGlobalEnvMap();
                    envmap != (unsigned int)-1)
                    CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, envmap));
                texOcp++;

                triObj.prog->set_uniformi("irradianceMap", texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                if (auto irradianceMap =
                        scene->envmapMan->integrator->getIrradianceMap();
                    irradianceMap != (unsigned int)-1)
                    CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap));
                texOcp++;

                triObj.prog->set_uniformi("prefilterMap", texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                if (auto prefilterMap =
                        scene->envmapMan->integrator->getPrefilterMap();
                    prefilterMap != (unsigned int)-1)
                    CHECK_GL(glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap));
                texOcp++;

                triObj.prog->set_uniformi("brdfLUT", texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                if (auto brdfLUT = scene->envmapMan->integrator->getBRDFLut();
                    brdfLUT != (unsigned int)-1)
                    CHECK_GL(glBindTexture(GL_TEXTURE_2D, brdfLUT));
                texOcp++;

                triObj.prog->set_uniform("farPlane", scene->camera->g_far);
                triObj.prog->set_uniformi("cascadeCount", Light::cascadeCount);
                for (int lightNo = 0; lightNo < lights.size(); ++lightNo) {
                    auto &light = lights[lightNo];
                    auto name = "lightDir[" + std::to_string(lightNo) + "]";
                    triObj.prog->set_uniform(name.c_str(), light->lightDir);
                    name = "shadowTint[" + std::to_string(lightNo) + "]";
                    triObj.prog->set_uniform(name.c_str(),
                                             light->getShadowTint());
                    name = "shadowSoftness[" + std::to_string(lightNo) + "]";
                    triObj.prog->set_uniform(name.c_str(),
                                             light->shadowSoftness);
                    name = "lightIntensity[" + std::to_string(lightNo) + "]";
                    triObj.prog->set_uniform(name.c_str(),
                                             light->getIntensity());
                    for (size_t i = 0; i < Light::cascadeCount + 1; i++) {
                        auto name1 =
                            "near[" +
                            std::to_string(lightNo * (Light::cascadeCount + 1) +
                                           i) +
                            "]";
                        triObj.prog->set_uniform(name1.c_str(),
                                                 light->m_nearPlane[i]);

                        auto name2 =
                            "far[" +
                            std::to_string(lightNo * (Light::cascadeCount + 1) +
                                           i) +
                            "]";
                        triObj.prog->set_uniform(name2.c_str(),
                                                 light->m_farPlane[i]);

                        auto name =
                            "shadowMap[" +
                            std::to_string(lightNo * (Light::cascadeCount + 1) +
                                           i) +
                            "]";
                        triObj.prog->set_uniformi(name.c_str(), texOcp);
                        CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                        if (auto shadowMap = light->DepthMaps[i];
                            shadowMap != (unsigned int)-1)
                            CHECK_GL(glBindTexture(GL_TEXTURE_2D, shadowMap));
                        texOcp++;
                    }
                    for (size_t i = 0; i < Light::cascadeCount; ++i) {
                        auto name =
                            "cascadePlaneDistances[" +
                            std::to_string(lightNo * Light::cascadeCount + i) +
                            "]";
                        triObj.prog->set_uniform(name.c_str(),
                                                 light->shadowCascadeLevels[i]);
                    }
                    name = "lview[" + std::to_string(lightNo) + "]";
                    triObj.prog->set_uniform(name.c_str(), light->lightMV);

                    auto matrices = light->lightSpaceMatrices;
                    for (size_t i = 0; i < matrices.size(); i++) {
                        auto name =
                            "lightSpaceMatrices[" +
                            std::to_string(lightNo * (Light::cascadeCount + 1) +
                                           i) +
                            "]";
                        triObj.prog->set_uniform(name.c_str(), matrices[i]);
                    }
                }

                if (reflect) {
                    triObj.prog->set_uniform("reflectPass", 1.0f);
                } else {
                    triObj.prog->set_uniform("reflectPass", 0.0f);
                }
                triObj.prog->set_uniform(
                    "reflectionViewID",
                    (float)scene->mReflectivePass->getReflectionViewID());
                for (int i = 0; i < 16; i++) {
                    if (!scene->mReflectivePass->renderReflect(i))
                        continue;
                    auto name = "reflectMVP[" + std::to_string(i) + "]";
                    triObj.prog->set_uniform(
                        name.c_str(), scene->mReflectivePass->getReflectMVP(i));
                    name = "reflect_normals[" + std::to_string(i) + "]";
                    triObj.prog->set_uniform(
                        name.c_str(),
                        scene->mReflectivePass->getReflectiveNormal(i));
                    name = "reflect_centers[" + std::to_string(i) + "]";
                    triObj.prog->set_uniform(
                        name.c_str(),
                        scene->mReflectivePass->getReflectiveCenter(i));
                    auto name2 = "reflectionMap" + std::to_string(i);
                    triObj.prog->set_uniformi(name2.c_str(), texOcp);
                    CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                    CHECK_GL(glBindTexture(
                        GL_TEXTURE_RECTANGLE,
                        scene->mReflectivePass->getReflectMaps()[i]));
                    texOcp++;
                }
                triObj.prog->set_uniform("depthPass", depthPass);
                triObj.prog->set_uniformi("depthBuffer", texOcp);
                CHECK_GL(glActiveTexture(GL_TEXTURE0 + texOcp));
                CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE,
                                       scene->mDepthPass->getDepthTexture()));
                texOcp++;
            }

            triObj.ebo->bind();

            if (prim_has_inst) {
                CHECK_GL(glDrawElementsInstancedARB(
                    GL_TRIANGLES, /*count=*/triObj.count * 3, GL_UNSIGNED_INT,
                    /*first=*/0, prim_inst_amount));
            } else {
                CHECK_GL(glDrawElements(GL_TRIANGLES,
                                        /*count=*/triObj.count * 3,
                                        GL_UNSIGNED_INT, /*first=*/0));
            }

            if (scene->camera->render_wireframe) {
                glEnable(GL_POLYGON_OFFSET_LINE);
                glPolygonOffset(-1, -1);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                triObj.prog->set_uniformi("mRenderWireframe", true);
                if (prim_has_inst) {
                    CHECK_GL(glDrawElementsInstancedARB(
                        GL_TRIANGLES, /*count=*/triObj.count * 3,
                        GL_UNSIGNED_INT, /*first=*/0, prim_inst_amount));
                } else {
                    CHECK_GL(glDrawElements(GL_TRIANGLES,
                                            /*count=*/triObj.count * 3,
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

    void load_texture2Ds(
        const std::vector<std::shared_ptr<zeno::Texture2DObject>> &tex2Ds) {
        for (const auto &tex2D : tex2Ds) {
            auto tex = std::make_unique<Texture>();

#define SET_TEX_WRAP(TEX_WRAP, TEX_2D_WRAP)                                    \
    if (TEX_2D_WRAP == zeno::Texture2DObject::TexWrapEnum::REPEAT)             \
        TEX_WRAP = GL_REPEAT;                                                  \
    else if (TEX_2D_WRAP ==                                                    \
             zeno::Texture2DObject::TexWrapEnum::MIRRORED_REPEAT)              \
        TEX_WRAP = GL_MIRRORED_REPEAT;                                         \
    else if (TEX_2D_WRAP == zeno::Texture2DObject::TexWrapEnum::CLAMP_TO_EDGE) \
        TEX_WRAP = GL_CLAMP_TO_EDGE;                                           \
    else if (TEX_2D_WRAP ==                                                    \
             zeno::Texture2DObject::TexWrapEnum::CLAMP_TO_BORDER)              \
        TEX_WRAP = GL_CLAMP_TO_BORDER;

            SET_TEX_WRAP(tex->wrap_s, tex2D->wrapS)
            SET_TEX_WRAP(tex->wrap_t, tex2D->wrapT)

#undef SET_TEX_WRAP

#define SET_TEX_FILTER(TEX_FILTER, TEX_2D_FILTER)                           \
    if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::NEAREST)     \
        TEX_FILTER = GL_NEAREST;                                            \
    else if (TEX_2D_FILTER == zeno::Texture2DObject::TexFilterEnum::LINEAR) \
        TEX_FILTER = GL_LINEAR;                                             \
    else if (TEX_2D_FILTER ==                                               \
             zeno::Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_NEAREST)  \
        TEX_FILTER = GL_NEAREST_MIPMAP_NEAREST;                             \
    else if (TEX_2D_FILTER ==                                               \
             zeno::Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_NEAREST)   \
        TEX_FILTER = GL_LINEAR_MIPMAP_NEAREST;                              \
    else if (TEX_2D_FILTER ==                                               \
             zeno::Texture2DObject::TexFilterEnum::NEAREST_MIPMAP_LINEAR)   \
        TEX_FILTER = GL_NEAREST_MIPMAP_LINEAR;                              \
    else if (TEX_2D_FILTER ==                                               \
             zeno::Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_LINEAR)    \
        TEX_FILTER = GL_LINEAR_MIPMAP_LINEAR;

            SET_TEX_FILTER(tex->min_filter, tex2D->minFilter)
            SET_TEX_FILTER(tex->mag_filter, tex2D->magFilter)

#undef SET_TEX_FILTER

            tex->load(tex2D->path.c_str());
            textures.push_back(std::move(tex));
        }
    }
    Program *get_shadow_program(std::shared_ptr<zeno::MaterialObject> mtl,
                                std::shared_ptr<zeno::InstancingObject> inst) {
        std::string SMVS;
        if (inst != nullptr) {
            SMVS = R"(
#version 330 core

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;
in mat4 mInstModel;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;

void main()
{
  position = vec3(mInstModel * vec4(vPosition, 1.0));
  iColor = vColor;
  iNormal = transpose(inverse(mat3(mInstModel))) * vNormal;
  iTexCoord = vTexCoord;
  iTangent = mat3(mInstModel) * vTangent;
  gl_Position = mView * vec4(position, 1.0);
}
)";
        } else {
            SMVS = R"(
#version 330 core

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;

void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;
  iTexCoord = vTexCoord;
  iTangent = vTangent;
  gl_Position = mView * vec4(position, 1.0);
}
)";
        }

        auto SMFS = "#version 330 core\n/* common_funcs_begin */\n" +
                    mtl->common + "\n/* common_funcs_end */\n" + R"(
uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform bool mSmoothShading;
uniform bool mNormalCheck;
uniform bool mRenderWireframe;


in vec3 position;
in vec3 iColor;
in vec3 iNormal;
in vec3 iTexCoord;
in vec3 iTangent;
out vec4 fColor;

void main()
{   
    vec3 att_pos = position;
    vec3 att_clr = iColor;
    vec3 att_nrm = iNormal;
    vec3 att_uv = iTexCoord;
    vec3 att_tang = iTangent;
    float att_NoL = 0;
)" + mtl->frag + R"(
    if(mat_opacity>=0.99)
         discard;
    //fColor = vec4(gl_FragCoord.zzz,1);
}
)";

        auto SMGS = R"(
#version 330 core

layout(triangles, invocations = 8) in;
layout(triangle_strip, max_vertices = 3) out;

layout (std140, binding = 0) uniform LightSpaceMatrices
{
    mat4 lightSpaceMatrices[128];
};

void main()
{          
	for (int i = 0; i < 3; ++i)
	{
		gl_Position = lightSpaceMatrices[gl_InvocationID] * gl_in[i].gl_Position;
		gl_Layer = gl_InvocationID;
		EmitVertex();
	}
	EndPrimitive();
}  
)";
        return scene->shaderMan->compile_program(SMVS, SMFS);
    }

    Program *get_points_program() {
        auto vert = R"(#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform float mPointScale;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;

out vec3 position;
out vec3 color;
out float radius;
out float opacity;
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
        auto frag = R"(#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;

in vec3 position;
in vec3 color;
in float radius;
in float opacity;
out vec4 fColor;
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
  fColor = vec4(oColor, 1.0 - opacity);
}
)";

        return scene->shaderMan->compile_program(vert, frag);
    }

    Program *get_lines_program() {
        auto vert = R"(#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;

out vec3 position;
out vec3 color;

void main()
{
  position = vPosition;
  color = vColor;

  gl_Position = mVP * vec4(position, 1.0);
}
)";
        auto frag = R"(#version 130

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 position;
in vec3 color;
out vec4 fColor;
void main()
{
  fColor = vec4(color, 1.0);
}
)";

        return scene->shaderMan->compile_program(vert, frag);
    }

    Program *get_tris_program(std::shared_ptr<zeno::MaterialObject> mtl,
                              std::shared_ptr<zeno::InstancingObject> inst) {
        auto vert = inst != nullptr ? R"(#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;
in mat4 mInstModel;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;
void main()
{
  position = vec3(mInstModel * vec4(vPosition, 1.0));
  iColor = vColor;
  iNormal = transpose(inverse(mat3(mInstModel))) * vNormal;
  iTexCoord = vTexCoord;
  iTangent = mat3(mInstModel) * vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)"
                                    : R"(
#version 330

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

in vec3 vPosition;
in vec3 vColor;
in vec3 vNormal;
in vec3 vTexCoord;
in vec3 vTangent;

out vec3 position;
out vec3 iColor;
out vec3 iNormal;
out vec3 iTexCoord;
out vec3 iTangent;
void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;
  iTexCoord = vTexCoord;
  iTangent = vTangent;
  gl_Position = mVP * vec4(position, 1.0);
}
)";
        auto frag = R"(#version 330
)" + (mtl ? mtl->extensions : "") +
                    R"(
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


in vec3 position;
in vec3 iColor;
in vec3 iNormal;
in vec3 iTexCoord;
in vec3 iTangent;
out vec4 fColor;
uniform samplerCube skybox;

uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

vec3 pbr(vec3 albedo, float roughness, float metallic, float specular,
    vec3 nrm, vec3 idir, vec3 odir) {

  vec3 hdir = normalize(idir + odir);
  float NoH = max(0., dot_c(hdir, nrm));
  float NoL = max(0., dot_c(idir, nrm));
  float NoV = max(0., dot_c(odir, nrm));
  float VoH = clamp(dot_c(odir, hdir), 0., 1.);
  float LoH = clamp(dot_c(idir, hdir), 0., 1.);

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

)" +
                    (!mtl ?
                          R"(
vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal, vec3 tangent) {
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
)"
                          : "\n/* common_funcs_begin */\n" + mtl->common +
                                "\n/* common_funcs_end */\n"
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
    float d = max(dot_c(surfNorm, vector), 0.0);
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
	float d = max(dot_c(surfNorm, toView), 0.0); 
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
    float denominator = 4.0 * dot_c(surfNorm, toLight) * dot_c(surfNorm, toView);

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
    vec3 halfVec = normalize(toLight + toView);
    float NoL = dot(surfNorm, toLight);
    float NoH = dot(surfNorm, halfVec);
    float NoV = dot(surfNorm, toView);
    float VoH = dot(toView, halfVec);
    float angle = clamp(dot_c(surfNorm, toLight), 0.0, 1.0);
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

    float angle = clamp(dot_c(surfNorm, toLight), 0.0, 1.0);

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

const float PI = 3.14159265358979323846;

//important to do: load env texture here
vec3 SampleEnvironment(in vec3 reflVec)
{
    //if(reflVec.y>-0.5) return vec3(0,0,0);
    //else return vec3(1,1,1);//cubem(reflVec, 0);//texture(TextureEnv, reflVec).rgb;
    //here we have the problem reflVec is in eyespace but we need it in world space
    vec3 r = inverse(transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz))))*reflVec;
    return texture(skybox, r).rgb;
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
        
        float NoV = clamp(dot_c(surfNorm, toView), 0.0, 1.0);
        float NoL = clamp(dot_c(surfNorm, L), 0.0, 1.0);
        float NoH = clamp(dot_c(surfNorm, H), 0.0, 1.0);
        float VoH = clamp(dot_c(toView, H), 0.0, 1.0);
        
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
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.1415926 * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 CalculateLightingIBL(
    in vec3  N,
    in vec3  V,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    mat3 m = inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz));
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlickRoughness(dot_c(N, V), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    const float MAX_REFLECTION_LOD = 7.0;
    vec3 irradiance = textureLod(prefilterMap, m*N,  MAX_REFLECTION_LOD).rgb;
    vec3 diffuse      = irradiance * CalculateDiffuse(albedo);
    vec3 R = reflect(-V, N); 
    vec3 prefilteredColor = textureLod(prefilterMap, m*R,  roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    return (kD * diffuse + specular);

}

vec3 CalculateLightingIBLToon(
    in vec3  N,
    in vec3  V,
    in vec3  albedo,
    in float roughness,
    in float metallic)
{
    mat3 m = inverse(transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz))));
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlickRoughness(dot_c(N, V), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    const float MAX_REFLECTION_LOD = 7.0;
    vec3 irradiance = textureLod(prefilterMap, m*N,  MAX_REFLECTION_LOD).rgb;
    vec3 diffuse      = irradiance * CalculateDiffuse(albedo);
    
    vec3 R = reflect(-V, N); 
    vec3 prefilteredColor = textureLod(prefilterMap, m*R,  roughness * MAX_REFLECTION_LOD).rgb;
    vec3 prefilteredColor2 = textureLod(prefilterMap, m*R,  max(roughness, 0.5) * MAX_REFLECTION_LOD).rgb;
    prefilteredColor = clamp(smoothstep(0.5,0.5,length(prefilteredColor)), 0,1)*vec3(1,1,1);
    vec3 specularColor = mix(prefilteredColor+0.2, prefilteredColor2, prefilteredColor);
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    brdf.r = (floor(brdf.r/0.33)+0.165)*0.33;
    vec3 specular = specularColor * (F * brdf.r + smoothstep(0.7,0.7,brdf.y));

    return (kD * diffuse + specular);

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

float toonSpecular(vec3 V, vec3 L, vec3 N, float roughness)
{
    float NoV = dot(N,V);
    float _SpecularSize = pow((1-roughness),5);
    float specularFalloff = NoV;
    specularFalloff = pow(specularFalloff, 2);
    vec3 reflectionDirection = reflect(L, N);
    float towardsReflection = dot(V, -reflectionDirection);
    float specularChange = fwidth(towardsReflection);
    float specularIntensity = smoothstep(1.0 - _SpecularSize, 1.0 - _SpecularSize + specularChange, towardsReflection);
    return clamp(specularIntensity,0,1);
}
vec3 histThings(vec3 s)
{
    vec3 norms = s/(length(s)+0.00001);
    float ls = length(s);
    ls = ceil(ls/0.2)*0.2;
    return norms * ls;
}
)" + R"(
float V_Kelemen(float LoH) {
    return 0.25 / (LoH * LoH);
}
vec3 ToonBRDF(vec3 baseColor, float metallic, float subsurface, 
float specular, 
float roughness,
float specularTint,
float anisotropic,
float sheen,
float sheenTint,
float clearcoat,
float clearcoatGloss,
vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y)
{
    float NoL = dot(N,L);
    float shad1 = smoothstep(0.3, 0.31, NoL);
    float shad2 = smoothstep(0.0,0.01, NoL);
    vec3 diffuse = mon2lin(baseColor)/PI;
    vec3 shadowC1 = diffuse * 0.4;
    vec3 C1 = mix(shadowC1, diffuse, shad1);
    vec3 shadowC2 = shadowC1 * 0.4;
    vec3 C2 = mix(shadowC2, C1, shad2);

    vec3 H = normalize(L+V);
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, baseColor, metallic);
    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);   
    float G   = GeometrySmith(N, V, L, roughness);    
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);        
    
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 s = numerator / denominator;
    
    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals 
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;	                

    vec3 norms = s/(length(s)+0.00001);
    float ls = length(s);
    ls = ceil(ls/0.4)*0.4;


    return (kD*C2 + norms * ls * toonSpecular(V, L, N, roughness));
}
vec3 ToonDisneyBRDF(vec3 baseColor, float metallic, float subsurface, 
float specular, 
float roughness,
float specularTint,
float anisotropic,
float sheen,
float sheenTint,
float clearcoat,
float clearcoatGloss,
vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y)
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
    float FL = clamp(SchlickFresnel(NdotL),0,1);
    float FV = clamp(SchlickFresnel(NdotV),0,1);
    float Fd90 = 0.5 + 2 * LdotH*LdotH * roughness;
    float viewIndp = mix(1.0, Fd90, FL);
    float Fd = (floor(viewIndp/0.33)+0.165) * 0.33 * mix(1.0, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH*LdotH*roughness;
    float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
    float NDLV = (NdotL + NdotV)>0?clamp((NdotL + NdotV),0.0001, 2.0):clamp((NdotL + NdotV), -2.0, -0.0001);
    float ss = 1.25 * (Fss * (1 /NDLV  - .5) + .5);

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
    float c1 = (1/PI) * mix(Fd, ss, subsurface);

    float shad1 = smoothstep(0.3, 0.31, NdotL);
    float shad2 = smoothstep(0.0,0.01, NdotL);
    vec3 shadowC1 = vec3(1,1,1) * 0.4;
    vec3 C1 = mix(shadowC1, vec3(1,1,1), shad1);
    vec3 shadowC2 = shadowC1 * 0.4;
    vec3 C2 = mix(shadowC2, C1, shad2);
    //c1 *= C2.x;
    
    Fsheen = Fsheen/(length(Fsheen)+1e-5) * (floor(length(Fsheen)/0.2)+0.1)*0.2;
    vec3 fspecularTerm = (Gs*Fs*Ds);
    vec3 fcoatTerm =  vec3(.25*clearcoat*Gr*Fr*Dr);

    return ((c1 * Cdlin  + Fsheen)
        * (1-metallic)
        + (normalize(fspecularTerm) * ceil(length(fspecularTerm)/0.3) * 0.3 + fcoatTerm)* toonSpecular(V, L, N, roughness)) * C2 ;
        
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
vec3 L, vec3 V, vec3 N, vec3 X, vec3 Y)
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
    float FL = clamp(SchlickFresnel(NdotL),0,1);
    float FV = clamp(SchlickFresnel(NdotV),0,1);
    float Fd90 = 0.5 + 2 * LdotH*LdotH * roughness;
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH*LdotH*roughness;
    float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
    float NDLV = (NdotL + NdotV)>0?clamp((NdotL + NdotV),0.0001, 2.0):clamp((NdotL + NdotV), -2.0, -0.0001);
    float ss = 1.25 * (Fss * (1 /NDLV  - .5) + .5);

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
    float c1 = (1/PI) * mix(Fd, ss, subsurface);
    
    return ((c1 * Cdlin + Fsheen)
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
float softLight0(float a, float b)
{
float G;
float res;
if(b<=0.25)
    G = ((16*b-12)*b+4)*b;
else
   G = sqrt(b);
if(a<=0.5)
   res = b - (1-2*a)*b*(1-b);
else
   res = b+(2*a-1)*(G-b);
return res;
}
float linearLight0(float a, float b)
{
    if(a>0.5)
        return b + 2 * (a-0.5);
   else
       return b + a - 1;
}
float brightness(vec3 c)
{
    return sqrt(c.x * c.r * 0.241 + c.y * c.y * 0.691 + c.z * c.z * 0.068);
}
uniform int lightNum; 
uniform vec3 light[16];
uniform sampler2D shadowMap[128];
uniform vec3 lightIntensity[16];
uniform vec3 shadowTint[16];
uniform float shadowSoftness[16];
uniform vec3 lightDir[16];
uniform float farPlane;
uniform mat4 lview[16];
uniform float near[128];
uniform float far[128];
//layout (std140, binding = 0) uniform LightSpaceMatrices
//{
uniform mat4 lightSpaceMatrices[128];
//};
uniform float cascadePlaneDistances[112];
uniform int cascadeCount;   // number of frusta - 1
vec3 random3(vec3 c) {
	float j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
	vec3 r;
	r.z = fract(512.0*j);
	j *= .125;
	r.x = fract(512.0*j);
	j *= .125;
	r.y = fract(512.0*j);
	return r-0.5;
}
float sampleShadowArray(int lightNo, vec2 coord, int layer)
{
    vec4 res;
    
    res = texture(shadowMap[lightNo * (cascadeCount + 1) + layer], coord);

    return res.r;    
}
float PCFLayer(int lightNo, float currentDepth, float bias, vec3 pos, int layer, int k, float softness, vec2 coord)
{
    float shadow = 0.0;
    
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap[lightNo * (cascadeCount + 1) + 0], 0));
    for(int x = -k; x <= k; ++x)
    {
        for(int y = -k; y <= k; ++y)
        {
            vec3 noise = random3(pos+vec3(x, y,0)*0.01*softness);
            float pcfDepth = sampleShadowArray(lightNo, coord + (vec2(x, y) * softness + noise.xy) * texelSize, layer); 
            shadow += (currentDepth - bias) > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    float size = 2.0*float(k)+1.0;
    return shadow /= (size*size);
}
float ShadowCalculation(int lightNo, vec3 fragPosWorldSpace, float softness)
{
    // select cascade layer
    vec4 fragPosViewSpace = mView * vec4(fragPosWorldSpace, 1.0);
    float depthValue = abs(fragPosViewSpace.z);

    int layer = -1;
    for (int i = 0; i < cascadeCount; ++i)
    {
        vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + i] * vec4(fragPosWorldSpace, 1.0);
        // perform perspective divide
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        // transform to [0,1] range
        projCoords = projCoords * 0.5 + 0.5;
        if (projCoords.x>=0&&projCoords.x<=1&&projCoords.y>=0&&projCoords.y<=1)
        {
            layer = i;
            break;
        }
    }
    if (layer == -1)
    {
        layer = cascadeCount;
    }

    vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + layer] * vec4(fragPosWorldSpace, 1.0);
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;

    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;

    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if (currentDepth > 1.0)
    {
        return 0.0;
    }
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(iNormal);
    float bias = max(0.0001 * (1.0 - dot(normal, light[0])), 0.00001);
    // float bm = bias;
    // const float biasModifier = 0.5f;
    // if (layer == cascadeCount)
    // {
    //     bm *= 1 / (farPlane * biasModifier);
    // }
    // else
    // {
    //     bm *= 1 / (cascadePlaneDistances[lightNo * cascadeCount + layer] * biasModifier);
    // }

    // PCF
    float shadow1 = PCFLayer(lightNo, currentDepth, bias, fragPosWorldSpace, layer, 3, softness, projCoords.xy);
    float shadow2 = shadow1;
    float coef = 0.0;
    if(layer>=1){
        //bm = 1 / (cascadePlaneDistances[lightNo * cascadeCount + (layer-1)] * biasModifier);
        fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + (layer-1)] * vec4(fragPosWorldSpace, 1.0);
        projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        projCoords * 0.5 + 0.5;
        shadow2 = PCFLayer(lightNo, currentDepth, bias, fragPosWorldSpace, layer-1, 3, softness, projCoords.xy);
        float coef = (depthValue - cascadePlaneDistances[lightNo * cascadeCount + (layer-1)])/(cascadePlaneDistances[lightNo * cascadeCount + layer] - cascadePlaneDistances[lightNo * cascadeCount + (layer-1)]);
    }
    
        
    return mix(shadow1, shadow2, coef);
}
float PCFAttLayer(int lightNo, float currentDepth, float bias, vec3 pos, int layer, int k, float softness, vec2 coord, float near, float far)
{
    
    float length = 0.0;
    float res = far;
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap[lightNo * (cascadeCount + 1) + 0], 0));
    for(int x = -k; x <= k; ++x)
    {
        for(int y = -k; y <= k; ++y)
        {
            vec3 noise = random3(pos+vec3(x, y,0)*0.01*softness);
            float pcfDepth = sampleShadowArray(lightNo, coord + (vec2(x, y) * softness + noise.xy) * softness * texelSize, layer) * (far-near) + near; 
            res = min(res, abs(currentDepth - pcfDepth));
        }    
    }
    return res;
}

float lightAttenuation(int lightNo, vec3 fragPosWorldSpace, float softness)
{
    // select cascade layer
    vec4 fragPosViewSpace = mView * vec4(fragPosWorldSpace, 1.0);
    float depthValue = abs(fragPosViewSpace.z);

    int layer = -1;
    for (int i = 0; i < cascadeCount; ++i)
    {
        vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + i] * vec4(fragPosWorldSpace, 1.0);
        // perform perspective divide
        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
        // transform to [0,1] range
        projCoords = projCoords * 0.5 + 0.5;
        if (projCoords.x>=0&&projCoords.x<=1&&projCoords.y>=0&&projCoords.y<=1)
        {
            layer = i;
            break;
        }
    }
    if (layer == -1)
    {
        layer = cascadeCount;
    }

    vec4 fragPosLightSpace = lightSpaceMatrices[lightNo * (cascadeCount + 1) + layer] * vec4(fragPosWorldSpace, 1.0);
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;

    // get depth of current fragment from light's perspective
    float nearPlane = near[lightNo * (cascadeCount + 1) + layer];
    float farPlane = far[lightNo * (cascadeCount + 1) + layer];
    float currentDepth = projCoords.z * (farPlane - nearPlane) + nearPlane;

    float avgL = PCFAttLayer(lightNo, currentDepth, 0, fragPosWorldSpace, layer, 5, 1, projCoords.xy, nearPlane, farPlane);

    return avgL;
    
}

)" + R"(
uniform mat4 reflectMVP[16];
uniform sampler2DRect reflectionMap0;
uniform sampler2DRect reflectionMap1;
uniform sampler2DRect reflectionMap2;
uniform sampler2DRect reflectionMap3;
uniform sampler2DRect reflectionMap4;
uniform sampler2DRect reflectionMap5;
uniform sampler2DRect reflectionMap6;
uniform sampler2DRect reflectionMap7;
uniform sampler2DRect reflectionMap8;
uniform sampler2DRect reflectionMap9;
uniform sampler2DRect reflectionMap10;
uniform sampler2DRect reflectionMap11;
uniform sampler2DRect reflectionMap12;
uniform sampler2DRect reflectionMap13;
uniform sampler2DRect reflectionMap14;
uniform sampler2DRect reflectionMap15;
vec4 sampleReflectRectID(vec2 coord, int id)
{
    if(id==0) return texture2DRect(reflectionMap0, coord);
    if(id==1) return texture2DRect(reflectionMap1, coord);
    if(id==2) return texture2DRect(reflectionMap2, coord);
    if(id==3) return texture2DRect(reflectionMap3, coord);
    if(id==4) return texture2DRect(reflectionMap4, coord);
    if(id==5) return texture2DRect(reflectionMap5, coord);
    if(id==6) return texture2DRect(reflectionMap6, coord);
    if(id==7) return texture2DRect(reflectionMap7, coord);
    if(id==8) return texture2DRect(reflectionMap8, coord);
    if(id==9) return texture2DRect(reflectionMap9, coord);
    if(id==10) return texture2DRect(reflectionMap10, coord);
    if(id==11) return texture2DRect(reflectionMap11, coord);
    if(id==12) return texture2DRect(reflectionMap12, coord);
    if(id==13) return texture2DRect(reflectionMap13, coord);
    if(id==14) return texture2DRect(reflectionMap14, coord);
    if(id==15) return texture2DRect(reflectionMap15, coord);
}
float mad(float a, float b, float c)
{
    return c + a * b;
}
vec3 mad3(vec3 a, vec3 b, vec3 c)
{
    return c + a * b;
}
vec3 skinBRDF(vec3 normal, vec3 light, float curvature)
{
    float NdotL = dot(normal, light) * 0.5 + 0.5; // map to 0 to 1 range
    float curva = (1.0/mad(curvature, 0.5 - 0.0625, 0.0625) - 2.0) / (16.0 - 2.0); 
    float oneMinusCurva = 1.0 - curva;
    vec3 curve0;
    {
        vec3 rangeMin = vec3(0.0, 0.3, 0.3);
        vec3 rangeMax = vec3(1.0, 0.7, 0.7);
        vec3 offset = vec3(0.0, 0.06, 0.06);
        vec3 t = clamp( mad3(vec3(NdotL), 1.0 / (rangeMax - rangeMin), (offset + rangeMin) / (rangeMin - rangeMax)  ), vec3(0), vec3(1));
        vec3 lowerLine = (t * t) * vec3(0.65, 0.5, 0.9);
        lowerLine.r += 0.045;
        lowerLine.b *= t.b;
        vec3 m = vec3(1.75, 2.0, 1.97);
        vec3 upperLine = mad3(vec3(NdotL), m, vec3(0.99, 0.99, 0.99) -m );
        upperLine = clamp(upperLine, vec3(0), vec3(1));
        vec3 lerpMin = vec3(0.0, 0.35, 0.35);
        vec3 lerpMax = vec3(1.0, 0.7 , 0.6 );
        vec3 lerpT = clamp( mad3(vec3(NdotL), vec3(1.0)/(lerpMax-lerpMin), lerpMin/ (lerpMin - lerpMax) ), vec3(0), vec3(1));
        curve0 = mix(lowerLine, upperLine, lerpT * lerpT);
    }
    vec3 curve1;
    {
        vec3 m = vec3(1.95, 2.0, 2.0);
        vec3 upperLine = mad3( vec3(NdotL), m, vec3(0.99, 0.99, 1.0) - m);
        curve1 = clamp(upperLine,0,1);
    }
    float oneMinusCurva2 = oneMinusCurva * oneMinusCurva;
    vec3 brdf = mix(curve0, curve1, mad(oneMinusCurva2, -1.0 * oneMinusCurva2, 1.0) );
    return brdf;
}
vec3 reflectionCalculation(vec3 worldPos, int id)
{
    vec4 fragPosReflectSpace = reflectMVP[id] * vec4(worldPos, 1.0);
    // perform perspective divide
    vec3 projCoords = fragPosReflectSpace.xyz / fragPosReflectSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.x>=0&&projCoords.x<=1&&projCoords.y>=0&&projCoords.y<=1)
    {
        return sampleReflectRectID(projCoords.xy * vec2(textureSize(reflectionMap0,0)), id ).xyz;
    }
    return vec3(0,0,0);
}
uniform float reflectPass;
uniform float reflectionViewID;
uniform float depthPass;
uniform sampler2DRect depthBuffer;
uniform vec3 reflect_normals[16];
uniform vec3 reflect_centers[16];
vec3 studioShading(vec3 albedo, vec3 view_dir, vec3 normal, vec3 old_tangent) {
    vec4 projPos = mView * vec4(position.xyz, 1.0);
    //normal = normalize(normal);
    vec3 L1 = light[0];
    vec3 att_pos = position;
    vec3 att_clr = iColor;
    vec3 att_nrm = normal;
    vec3 att_uv = iTexCoord;
    vec3 att_tang = old_tangent;
    float att_NoL = dot(normal, L1);
    //if(depthPass<=0.01)
    //{
    //    
    //    float d = texture2DRect(depthBuffer, gl_FragCoord.xy).r;
    //    if(d==0 || abs(projPos.z)>abs(d) )
    //        discard;
    //}
    /* custom_shader_begin */
)" + mtl->frag + R"(
    /* custom_shader_end */
    if(reflectPass==1.0 && mat_reflection==1.0 )
        discard;
    if(reflectPass==1.0 && dot(reflect_normals[int(reflectionViewID)], position-reflect_centers[int(reflectionViewID)])<0)
        discard;
    if(mat_opacity>=0.99 && mat_reflection!=1.0)
        discard;
    
    //if(depthPass>=0.99)
    //{
    //    return abs(projPos.zzz);
    //}
    
    vec3 colorEmission = mat_emission;
    mat_metallic = clamp(mat_metallic, 0, 1);
    vec3 new_normal = normal; /* TODO: use mat_normal to transform this */
    vec3 color = vec3(0,0,0);
    vec3 light_dir;
    vec3 albedo2 = mat_basecolor;
    float roughness = mat_roughness;
    vec3 tan = normalize(old_tangent - dot(normal, old_tangent)*normal);
    mat3 TBN = mat3(tan, cross(normal, tan), normal);

    new_normal = TBN*mat_normal;
    mat3 eyeinvmat = transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz)));
    new_normal = eyeinvmat*new_normal;
    //vec3 up        = abs(new_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = eyeinvmat*tan;//   = normalize(cross(up, new_normal));
    vec3 bitangent = eyeinvmat*TBN[1];// = cross(new_normal, tangent);
    //pixarONB(new_normal, tangent, bitangent);
    color = vec3(0,0,0);
    vec3 realColor = vec3(0,0,0);
    for(int lightId=0; lightId<lightNum; lightId++){
        light_dir = mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz)*lightDir[lightId];
        vec3 photoReal = BRDF(mat_basecolor, mat_metallic,mat_subsurface,mat_specular,mat_roughness,mat_specularTint,mat_anisotropic,mat_sheen,mat_sheenTint,mat_clearcoat,mat_clearcoatGloss,normalize(light_dir), normalize(view_dir), normalize(new_normal),normalize(tangent), normalize(bitangent)) * lightIntensity[lightId];// * vec3(1, 1, 1) * mat_zenxposure;
        vec3 NPR = ToonDisneyBRDF(mat_basecolor, mat_metallic,0,mat_specular,mat_roughness,mat_specularTint,mat_anisotropic,mat_sheen,mat_sheenTint,mat_clearcoat,mat_clearcoatGloss,normalize(light_dir), normalize(view_dir), normalize(new_normal),normalize(tangent), normalize(bitangent)) * lightIntensity[lightId];// * vec3(1, 1, 1) * mat_zenxposure;

        vec3 sss =  vec3(0);
        if(mat_subsurface>0)
        {
            vec3 vl = light_dir + new_normal * mat_sssParam.x;

            float ltDot = pow(clamp(dot(normalize(view_dir), -vl),0,1), 12.0) * mat_sssParam.y;
            float lthick = lightAttenuation(lightId, position, shadowSoftness[lightId]);
            sss = mat_thickness * exp(-lthick * mat_sssParam.z) * ltDot * mat_sssColor * lightIntensity[lightId];
        }
        if(mat_foliage>0)
        {
            if(dot(new_normal, light_dir)<0)
            {
                sss += mat_foliage * clamp(dot(-new_normal, light_dir)*0.6+0.4, 0,1)*mon2lin(mat_basecolor)/PI;
            }
        }
        if(mat_skin>0)
        {
            sss += mat_skin * skinBRDF(new_normal, light_dir, mat_curvature) * lightIntensity[lightId] * mon2lin(mat_basecolor)/PI;
        }

        vec3 lcolor = mix(photoReal, NPR, mat_toon) + mat_subsurface * sss;
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

    //    light_dir = vec3(0,1,-1);
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


        
        
        float shadow = ShadowCalculation(lightId, position, shadowSoftness[lightId]);
        color += lcolor * clamp(vec3(1.0-shadow)+shadowTint[lightId],vec3(0),vec3(1));
        realColor += photoReal * clamp(vec3(1.0-shadow)+shadowTint[lightId],vec3(0),vec3(1));
    }

    
    vec3 iblPhotoReal =  CalculateLightingIBL(new_normal,view_dir,albedo2,roughness,mat_metallic);
    vec3 iblNPR = CalculateLightingIBLToon(new_normal,view_dir,albedo2,roughness,mat_metallic);
    vec3 ibl = mat_ao * mix(iblPhotoReal, iblNPR,mat_toon);
    color += ibl;
    realColor += iblPhotoReal;
    float brightness0 = brightness(realColor)/(brightness(mon2lin(mat_basecolor))+0.00001);
    float brightness1 = smoothstep(mat_shape.x, mat_shape.y, dot(new_normal, light_dir));
    float brightness = mix(brightness1, brightness0, mat_style);
    
    float brightnessNoise = softLight0(mat_strokeNoise, brightness);
    brightnessNoise = smoothstep(mat_shad.x, mat_shad.y, brightnessNoise);
    float stroke = linearLight0(brightnessNoise, mat_stroke);
    stroke = clamp(stroke + mat_stroke, 0,1);
    vec3 strokeColor = clamp(vec3(stroke) + mat_strokeTint, vec3(0), vec3(1));

    //strokeColor = pow(strokeColor,vec3(2.0));
    color = mix(color, mix(color*strokeColor, color, strokeColor), mat_toon);

    color = ACESFitted(color.rgb, 2.2);
    if(mat_reflection==1.0 && mat_reflectID>-1 && mat_reflectID<16)
        color = mix(color, reflectionCalculation(position, int(mat_reflectID)), mat_reflection);
    return color + colorEmission;


}
)") + R"(

vec3 calcRayDir(vec3 pos)
{
  vec4 vpos = mView * vec4(pos, 1);
//   vec2 uv = vpos.xy / vpos.w;
//   vec4 ro = mInvVP * vec4(uv, -1, 1);
//   vec4 re = mInvVP * vec4(uv, +1, 1);
//   vec3 rd = normalize(re.xyz / re.w - ro.xyz / ro.w);
  return normalize(vpos.xyz);
}
uniform float mSampleWeight;
void main()
{
  
  if (mRenderWireframe) {
    fColor = vec4(0.89, 0.57, 0.15, 1.0);
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
  vec3 normalInView = transpose(inverse(mat3(mView[0].xyz, mView[1].xyz, mView[2].xyz)))*normal;
  if(dot(-viewdir, normalInView)>0)
    normal = - normal;

  //normal = faceforward(normal, -viewdir, normal);
  vec3 tangent = iTangent;
  if (tangent == vec3(0)) {
   vec3 unusedbitan;
   pixarONB(normal, tangent, unusedbitan);
  }

  vec3 color = studioShading(albedo, viewdir, normal, tangent);
  
  fColor = vec4(color*mSampleWeight, 1);
  
  if (mNormalCheck) {
      float intensity = clamp((mView * vec4(normal, 0)).z, 0, 1) * 0.4 + 0.6;
      if (gl_FrontFacing) {
        fColor = vec4(0.42 * intensity*mSampleWeight, 0.42 * intensity*mSampleWeight, 0.93 * intensity*mSampleWeight, 1);
      } else {
        fColor = vec4(0.87 * intensity*mSampleWeight, 0.22 * intensity*mSampleWeight, 0.22 * intensity*mSampleWeight, 1);
      }
  }
}
)";

        //printf("!!!!%s!!!!\n", frag.c_str());
        return scene->shaderMan->compile_program(vert, frag);
    }
};

std::unique_ptr<IGraphic>
makeGraphicPrimitive(Scene *scene, std::shared_ptr<zeno::IObject> obj) {
    if (auto prim = std::dynamic_pointer_cast<zeno::PrimitiveObject>(obj))
        return std::make_unique<GraphicPrimitive>(scene, std::move(prim));
    return nullptr;
}

} // namespace zenovis
