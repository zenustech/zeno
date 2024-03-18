#include <memory>
#include <string>
#include <vector>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/InstancingObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/ticktock.h>
#include <zeno/utils/vec.h>
#include <zeno/extra/TempNode.h>
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
    std::vector<std::unique_ptr<Buffer>> vbos;
    std::unique_ptr<Buffer> ebo;
    size_t count = 0;
    Program *prog{};
};
#if 0
static void parsePointsDrawBuffer(zeno::PrimitiveObject *prim, ZhxxDrawObject &obj) {
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &uv = prim->attr<zeno::vec3f>("uv");
    auto const &tang = prim->attr<zeno::vec3f>("tang");
    obj.count = prim->size();
    obj.vbos.resize(1);
    obj.vbos[0] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    std::vector<zeno::vec3f> mem(obj.count * 5);
    for (int i = 0; i < obj.count; i++) {
        mem[5 * i + 0] = pos[i];
        mem[5 * i + 1] = clr[i];
        mem[5 * i + 2] = nrm[i];
        mem[5 * i + 3] = uv[i];
        mem[5 * i + 4] = tang[i];
    }
    obj.vbos[0]->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    size_t points_count = prim->points.size();
    if (points_count) {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(prim->points.data(),
                           points_count * sizeof(prim->points[0]));
    }
}
#endif

static void parseLinesDrawBuffer(zeno::PrimitiveObject *prim, ZhxxDrawObject &obj) {
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tang = prim->attr<zeno::vec3f>("tang");
    auto const &lines = prim->lines;
    obj.count = prim->lines.size();
    obj.vbos.resize(5);
    std::vector<zeno::vec2i> linesdata(obj.count);
    std::vector<zeno::vec3f> _pos(obj.count * 2);
    std::vector<zeno::vec3f> _clr(obj.count * 2);
    std::vector<zeno::vec3f> _nrm(obj.count * 2);
    std::vector<zeno::vec3f> _uv(obj.count * 2);
    std::vector<zeno::vec3f> _tang(obj.count * 2);
#pragma omp parallel for
    for (auto i = 0; i < obj.count; i++) {
        _pos[i * 2 + 0] = pos[lines[i][0]];
        _pos[i * 2 + 1] = pos[lines[i][1]];

        _clr[i * 2 + 0] = clr[lines[i][0]];
        _clr[i * 2 + 1] = clr[lines[i][1]];

        _nrm[i * 2 + 0] = nrm[lines[i][0]];
        _nrm[i * 2 + 1] = nrm[lines[i][1]];

        _tang[i * 2 + 0] = tang[lines[i][0]];
        _tang[i * 2 + 1] = tang[lines[i][1]];

        linesdata[i] = zeno::vec2i(i * 2, i * 2 + 1);
    }
    bool has_uv = lines.has_attr("uv0") && lines.has_attr("uv1");
    if (has_uv) {
        auto &uv0 = lines.attr<zeno::vec3f>("uv0");
        auto &uv1 = lines.attr<zeno::vec3f>("uv1");
        for (auto i = 0; i < obj.count; i++) {
            _uv[i * 2 + 0] = uv0[i];
            _uv[i * 2 + 1] = uv1[i];
        }
    }

    obj.vbos[0] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[0]->bind_data(_pos.data(), _pos.size() * sizeof(_pos[0]));
    obj.vbos[1] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[1]->bind_data(_clr.data(), _clr.size() * sizeof(_clr[0]));
    obj.vbos[2] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[2]->bind_data(_nrm.data(), _nrm.size() * sizeof(_nrm[0]));
    obj.vbos[3] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[3]->bind_data(_uv.data(), _uv.size() * sizeof(_uv[0]));
    obj.vbos[4] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[4]->bind_data(_tang.data(), _tang.size() * sizeof(_tang[0]));

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
    const zeno::vec3f *uv0_data = nullptr;
    const zeno::vec3f *uv1_data = nullptr;
    const zeno::vec3f *uv2_data = nullptr;
    if(has_uv)
    {
        uv0_data = tris.attr<zeno::vec3f>("uv0").data();
        uv1_data = tris.attr<zeno::vec3f>("uv1").data();
        uv2_data = tris.attr<zeno::vec3f>("uv2").data();
    }
#pragma omp parallel for
    for (auto i = 0; i < prim->tris.size(); ++i) {
        if (has_uv) {
            const auto &pos0 = pos[tris[i][0]];
            const auto &pos1 = pos[tris[i][1]];
            const auto &pos2 = pos[tris[i][2]];
            auto uv0 = uv0_data[i];
            auto uv1 = uv1_data[i];
            auto uv2 = uv2_data[i];

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
#if 0
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
    obj.vbos.resize(1);
    obj.vbos[0] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
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
    obj.vbos[0]->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    /* TOCK(bindvbo); */
    /* TICK(bindebo); */
    if (obj.count) {
        obj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
        obj.ebo->bind_data(&(trisdata[0]), tris1.size() * sizeof(trisdata[0]));
    }
    /* TOCK(bindebo); */
}
#endif
static void parseTrianglesDrawBuffer(zeno::PrimitiveObject *prim, ZhxxDrawObject &obj) {
    /* TICK(parse); */
    auto const &pos = prim->attr<zeno::vec3f>("pos");
    auto const &clr = prim->attr<zeno::vec3f>("clr");
    auto const &nrm = prim->attr<zeno::vec3f>("nrm");
    auto const &tris = prim->tris;

    obj.count = tris.size();
    obj.vbos.resize(5);
    std::vector<zeno::vec3i> trisdata(obj.count);
    auto &tang = prim->tris.attr<zeno::vec3f>("tang");
    std::vector<zeno::vec3f> _pos(obj.count * 3);
    std::vector<zeno::vec3f> _clr(obj.count * 3);
    std::vector<zeno::vec3f> _nrm(obj.count * 3);
    std::vector<zeno::vec3f> _uv(obj.count * 3);
    std::vector<zeno::vec3f> _tang(obj.count * 3);

#pragma omp parallel for
    for (auto i = 0; i < obj.count; i++) {
        _pos[i * 3 + 0] = pos[tris[i][0]];
        _pos[i * 3 + 1] = pos[tris[i][1]];
        _pos[i * 3 + 2] = pos[tris[i][2]];

        _clr[i * 3 + 0] = clr[tris[i][0]];
        _clr[i * 3 + 1] = clr[tris[i][1]];
        _clr[i * 3 + 2] = clr[tris[i][2]];

        _nrm[i * 3 + 0] = nrm[tris[i][0]];
        _nrm[i * 3 + 1] = nrm[tris[i][1]];
        _nrm[i * 3 + 2] = nrm[tris[i][2]];

        _tang[i * 3 + 0] = tang[i];
        _tang[i * 3 + 1] = tang[i];
        _tang[i * 3 + 2] = tang[i];

        trisdata[i] = zeno::vec3i(i * 3, i * 3 + 1, i * 3 + 2);
    }
    bool has_uv =
            tris.has_attr("uv0") && tris.has_attr("uv1") && tris.has_attr("uv2");
    if (has_uv) {
        auto &uv0 = tris.attr<zeno::vec3f>("uv0");
        auto &uv1 = tris.attr<zeno::vec3f>("uv1");
        auto &uv2 = tris.attr<zeno::vec3f>("uv2");
        for (auto i = 0; i < obj.count; i++) {
            _uv[i * 3 + 0] = uv0[i];
            _uv[i * 3 + 1] = uv1[i];
            _uv[i * 3 + 2] = uv2[i];
        }
    }

    obj.vbos[0] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[0]->bind_data(_pos.data(), _pos.size() * sizeof(_pos[0]));
    obj.vbos[1] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[1]->bind_data(_clr.data(), _clr.size() * sizeof(_clr[0]));
    obj.vbos[2] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[2]->bind_data(_nrm.data(), _nrm.size() * sizeof(_nrm[0]));
    obj.vbos[3] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[3]->bind_data(_uv.data(), _uv.size() * sizeof(_uv[0]));
    obj.vbos[4] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
    obj.vbos[4]->bind_data(_tang.data(), _tang.size() * sizeof(_tang[0]));
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
    std::vector<std::unique_ptr<Buffer>> vbos = std::vector<std::unique_ptr<Buffer>>(5);
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

    bool invisible;
    bool custom_color;

    ZhxxDrawObject pointObj;
    ZhxxDrawObject lineObj;
    ZhxxDrawObject triObj;
    std::vector<std::unique_ptr<Texture>> textures;
    std::shared_ptr<zeno::PrimitiveObject> primUnique;
    zeno::PrimitiveObject *prim;

    ZhxxDrawObject polyEdgeObj = {};
    ZhxxDrawObject polyUvObj = {};

    explicit ZhxxGraphicPrimitive(Scene *scene_, zeno::PrimitiveObject *primArg)
        : scene(scene_), primUnique(std::make_shared<zeno::PrimitiveObject>(*primArg)) {
        prim = primUnique.get();
        invisible = prim->userData().get2<bool>("invisible", 0);
        zeno::log_trace("rendering primitive size {}", prim->size());

        {
            bool any_not_triangle = false;
            for (const auto &[b, c]: prim->polys) {
                if (c > 3) {
                    any_not_triangle = true;
                }
            }
            if (any_not_triangle) {
                std::vector<int> edge_list;
                auto add_edge = [&](int a, int b) {
                    int p0 = prim->loops[a];
                    int p1 = prim->loops[b];
                    edge_list.push_back(p0);
                    edge_list.push_back(p1);
                };
                for (const auto &[b, c]: prim->polys) {
                    for (auto i = 2; i < c; i++) {
                        if (i == 2) {
                            add_edge(b, b + 1);
                        }
                        add_edge(b + i - 1, b + i);
                        if (i == c - 1) {
                            add_edge(b, b + i);
                        }
                    }
                }
                polyEdgeObj.count = edge_list.size();
                polyEdgeObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
                polyEdgeObj.ebo->bind_data(edge_list.data(), edge_list.size() * sizeof(edge_list[0]));
                auto vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
                vbo->bind_data(prim->verts.data(), prim->verts.size() * sizeof(prim->verts[0]));
                polyEdgeObj.vbos.push_back(std::move(vbo));
                polyEdgeObj.prog = get_edge_program();
            }
            if (any_not_triangle && prim->loops.attr_is<int>("uvs")) {
                std::vector<zeno::vec3f> uv_data;
                std::vector<int> uv_list;
                auto &uvs = prim->loops.attr<int>("uvs");
                auto add_uv = [&](int a, int b) {
                    int p0 = uvs[a];
                    int p1 = uvs[b];
                    uv_list.push_back(p0);
                    uv_list.push_back(p1);
                };
                for (const auto &[b, c]: prim->polys) {
                    for (auto i = 2; i < c; i++) {
                        if (i == 2) {
                            add_uv(b, b + 1);
                        }
                        add_uv(b + i - 1, b + i);
                        if (i == c - 1) {
                            add_uv(b, b + i);
                        }
                    }
                }
                polyUvObj.count = uv_list.size();
                polyUvObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
                polyUvObj.ebo->bind_data(uv_list.data(), uv_list.size() * sizeof(uv_list[0]));
                auto vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
                for (const auto &uv: prim->uvs) {
                    uv_data.emplace_back(uv[0], uv[1], 0);
                }
                vbo->bind_data(uv_data.data(), uv_data.size() * sizeof(uv_data[0]));
                polyUvObj.vbos.push_back(std::move(vbo));
                polyUvObj.prog = get_edge_program();
            }
        }

        if (!prim->attr_is<zeno::vec3f>("pos")) {
            auto &pos = prim->add_attr<zeno::vec3f>("pos");
            for (size_t i = 0; i < pos.size(); i++) {
                pos[i] = zeno::vec3f(i * (1.0f / (pos.size() - 1)), 0, 0);
            }
        }
        custom_color = prim->attr_is<zeno::vec3f>("clr");
        if (!prim->attr_is<zeno::vec3f>("clr")) {
            auto &clr = prim->add_attr<zeno::vec3f>("clr");
            zeno::vec3f clr0(1.0f);
            if (!prim->tris.size() && !prim->quads.size() && !prim->polys.size()) {
                if (prim->lines.size())
                    clr0 = {1.0f, 0.6f, 0.2f};
                else
                    clr0 = {0.2f, 0.6f, 1.0f};
            }
            std::fill(clr.begin(), clr.end(), clr0);
        }
#if 1
        bool primNormalCorrect =
            prim->attr_is<zeno::vec3f>("nrm") &&
            (!prim->attr<zeno::vec3f>("nrm").size() ||
             length(prim->attr<zeno::vec3f>("nrm")[0]) > 1e-5);
        bool need_computeNormal =
            !primNormalCorrect || !(prim->attr_is<zeno::vec3f>("nrm"));
        bool thePrmHasFaces = !(!prim->tris.size() && !prim->quads.size() && !prim->polys.size());
        if (thePrmHasFaces && need_computeNormal) {
            /* std::cout << "computing normal\n"; */
            zeno::log_trace("computing normal");
            zeno::primCalcNormal(&*prim, 1);
        }
        if (int subdlevs = prim->userData().get2<int>("delayedSubdivLevels", 0)) {
            // todo: zhxx, should comp normal after subd or before?
            zeno::log_trace("computing subdiv {}", subdlevs);
            (void)zeno::TempNodeSimpleCaller("OSDPrimSubdiv")
                .set("prim", primUnique)
                .set2<int>("levels", subdlevs)
                .set2<std::string>("edgeCreaseAttr", "")
                .set2<bool>("triangulate", false)
                .set2<bool>("asQuadFaces", true)
                .set2<bool>("hasLoopUVs", true)
                .set2<bool>("delayTillIpc", false)
                .call();  // will inplace subdiv prim
            prim->userData().del("delayedSubdivLevels");
        }
        if (thePrmHasFaces) {
            zeno::log_trace("demoting faces");
            zeno::primTriangulateQuads(&*prim);
            zeno::primTriangulate(&*prim);//will further loop.attr("uv") to tris.attr("uv0")...
        }
#else
        zeno::primSepTriangles(&*prim, true, true);//TODO: rm keepTriFaces
#endif
        /* BEGIN TODO */
        //if (!prim->has_attr("nrm")) {
        if (!thePrmHasFaces) {
            if (prim->attr_is<float>("rad")) {
                if (prim->attr_is<float>("opa")) {
                    auto &rad = prim->attr<float>("rad");
                    auto &opa = prim->attr<float>("opa");
                    auto &radopa = prim->add_attr<zeno::vec3f>("nrm");
                    for (size_t i = 0; i < radopa.size(); i++) {
                        radopa[i] = zeno::vec3f(rad[i], opa[i], 0.0f);
                    }
                } else {
                    auto &rad = prim->attr<float>("rad");
                    auto &radopa = prim->add_attr<zeno::vec3f>("nrm");
                    for (size_t i = 0; i < radopa.size(); i++) {
                        radopa[i] = zeno::vec3f(rad[i], 0.0f, 0.0f);
                    }
                }
            } else {
                if (prim->attr_is<float>("opa")) {
                    auto &opa = prim->attr<float>("opa");
                    auto &radopa = prim->add_attr<zeno::vec3f>("nrm");
                    for (size_t i = 0; i < radopa.size(); i++) {
                        radopa[i] = zeno::vec3f(1.0f, opa[i], 0.0f);
                    }
                } else {
                    auto &radopa = prim->add_attr<zeno::vec3f>("nrm");
                    for (size_t i = 0; i < radopa.size(); i++) {
                        radopa[i] = zeno::vec3f(1.0f, 0.0f, 0.0f);
                    }
                }
            }
        } else {
        }
            //} else if (prim->tris.size()) {
                //// for (size_t i = 0; i < radopa.size(); i++) {
                ////     radopa[i] = zeno::vec3f(1 / zeno::sqrt(3.0f));
                //// }
                //for (size_t i = 0; i < radopa.size(); i++) {
                    //radopa[i] = zeno::vec3f(0.0f);
                //}

            //} else {
                //for (size_t i = 0; i < radopa.size(); i++) {
                    //radopa[i] = zeno::vec3f(1.5f, 0.0f, 0.0f);
                //}
            //}
        //}
        /* END TODO */
        if (!prim->attr_is<zeno::vec3f>("nrm")) {
            auto &nrm = prim->add_attr<zeno::vec3f>("nrm");
            std::fill(nrm.begin(), nrm.end(), zeno::vec3f(1.0f, 0.0f, 0.0f));
        }
        if (!prim->attr_is<zeno::vec3f>("uv")) {
            auto &uv = prim->add_attr<zeno::vec3f>("uv");
            std::fill(uv.begin(), uv.end(), zeno::vec3f(0.0f));
        }
        if (!prim->attr_is<zeno::vec3f>("tang")) {
            auto &tang = prim->add_attr<zeno::vec3f>("tang");
            std::fill(tang.begin(), tang.end(), zeno::vec3f(0.0f));
        }
        bool enable_uv = false;

        auto const &pos = prim->attr<zeno::vec3f>("pos");
        auto const &clr = prim->attr<zeno::vec3f>("clr");
        auto const &nrm = prim->attr<zeno::vec3f>("nrm");
        auto const &uv = prim->attr<zeno::vec3f>("uv");
        auto const &tang = prim->attr<zeno::vec3f>("tang");
        vertex_count = prim->size();

        vbos[0] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[0]->bind_data(pos.data(), pos.size() * sizeof(pos[0]));
        vbos[1] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[1]->bind_data(clr.data(), clr.size() * sizeof(clr[0]));
        vbos[2] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[2]->bind_data(nrm.data(), nrm.size() * sizeof(nrm[0]));
        vbos[3] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[3]->bind_data(uv.data(), uv.size() * sizeof(uv[0]));
        vbos[4] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[4]->bind_data(tang.data(), tang.size() * sizeof(tang[0]));

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

            } else {
                computeTrianglesTangent(&*prim);
                parseTrianglesDrawBuffer(&*prim, triObj);
            }

            bool findCamera = false;
            triObj.prog = get_tris_program();
        }

        draw_all_points = !points_count && !lines_count && !tris_count;
        auto& ud = prim->userData();
        if (ud.get2<int>("isImage", 0)) {
            draw_all_points = false;
        }
        if (draw_all_points) {
            pointObj.prog = get_points_program();
        }
    }

    virtual void draw() override {
        bool selected = scene->selected.count(nameid) > 0;
        if (scene->drawOptions->uv_mode && !selected) {
            return;
        }
        if (scene->drawOptions->show_grid == false && invisible) {
            return;
        }
        int id = 0;
        for (id = 0; id < textures.size(); id++) {
            textures[id]->bind_to(id);
        }

        auto vbobind = [](auto &vbos) {
            for (auto i = 0; i < 5; i++) {
                vbos[i]->bind();
                vbos[i]->attribute(/*index=*/i,
                        /*offset=*/sizeof(float) * 0,
                        /*stride=*/sizeof(float) * 3, GL_FLOAT,
                        /*count=*/3);
            }
        };
        auto vbounbind = [](auto &vbos) {
            for (auto i = 0; i < 5; i++) {
                vbos[i]->disable_attribute(i);
                vbos[i]->unbind();
            }
        };

        if (draw_all_points || points_count)
            vbobind(vbos);

        if (draw_all_points) {
            //printf("ALLPOINTS\n");
            pointObj.prog->use();
            float point_scale = 21.6f / std::tan(scene->camera->m_fov * 0.5f * 3.1415926f / 180.0f);
            point_scale *= scene->drawOptions->viewportPointSizeScale;
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
            vbounbind(vbos);

        if (lines_count) {
            //printf("LINES\n");
            if (lineObj.vbos.size()) {
                vbobind(lineObj.vbos);
            } else {
                vbobind(vbos);
            }
            lineObj.prog->use();
            scene->camera->set_program_uniforms(lineObj.prog);
            lineObj.ebo->bind();
            CHECK_GL(glDrawElements(GL_LINES, /*count=*/lineObj.count * 2,
                                    GL_UNSIGNED_INT, /*first=*/0));
            lineObj.ebo->unbind();
            if (lineObj.vbos.size()) {
                vbounbind(lineObj.vbos);
            } else {
                vbounbind(vbos);
            }
        }

        if (tris_count) {
            //printf("TRIS\n");
            if (triObj.vbos.size()) {
                vbobind(triObj.vbos);
            } else {
                vbobind(vbos);
            }

            triObj.prog->use();
            scene->camera->set_program_uniforms(triObj.prog);

            triObj.prog->set_uniform("mSmoothShading", scene->drawOptions->smooth_shading);
            triObj.prog->set_uniform("mNormalCheck", scene->drawOptions->normal_check);
            triObj.prog->set_uniform("mUvMode", scene->drawOptions->uv_mode);

            triObj.prog->set_uniformi("mRenderWireframe", false);
            triObj.prog->set_uniformi("mCustomColor", custom_color);

            triObj.ebo->bind();
            if (!scene->drawOptions->render_wireframe) {
                CHECK_GL(glDrawElements(GL_TRIANGLES,
                        /*count=*/triObj.count * 3,
                                        GL_UNSIGNED_INT, /*first=*/0));
            }

            if (scene->drawOptions->render_wireframe || selected || scene->drawOptions->uv_mode) {
                CHECK_GL(glDepthFunc(GL_GEQUAL));
                if (polyEdgeObj.count) {
                    if (scene->drawOptions->uv_mode) {
                        if (polyUvObj.count) {
                            polyUvObj.prog->use();
                            scene->camera->set_program_uniforms(polyUvObj.prog);

                            polyUvObj.vbos[0]->bind();
                            polyUvObj.vbos[0]->attribute(0, 0, 0, GL_FLOAT, 3);
                            polyUvObj.ebo->bind();

                            CHECK_GL(glDrawElements(GL_LINES, polyUvObj.count, GL_UNSIGNED_INT, 0));

                            polyUvObj.ebo->unbind();
                            polyUvObj.vbos[0]->disable_attribute(0);
                            polyUvObj.vbos[0]->unbind();
                        }
                    }
                    else {
                        polyEdgeObj.prog->use();
                        scene->camera->set_program_uniforms(polyEdgeObj.prog);

                        polyEdgeObj.vbos[0]->bind();
                        polyEdgeObj.vbos[0]->attribute(0, 0, 0, GL_FLOAT, 3);
                        polyEdgeObj.ebo->bind();

                        CHECK_GL(glDrawElements(GL_LINES, polyEdgeObj.count, GL_UNSIGNED_INT, 0));

                        polyEdgeObj.ebo->unbind();
                        polyEdgeObj.vbos[0]->disable_attribute(0);
                        polyEdgeObj.vbos[0]->unbind();
                    }
                }
                else {
                    CHECK_GL(glEnable(GL_POLYGON_OFFSET_LINE));
                    CHECK_GL(glPolygonOffset(0, 0));
                    CHECK_GL(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
                    triObj.prog->set_uniformi("mRenderWireframe", true);
                    CHECK_GL(glDrawElements(GL_TRIANGLES,
                                            /*count=*/triObj.count * 3,
                                            GL_UNSIGNED_INT, /*first=*/0));
                    CHECK_GL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
                    CHECK_GL(glDisable(GL_POLYGON_OFFSET_LINE));
                }
                CHECK_GL(glDepthFunc(GL_GREATER));
            }
            triObj.ebo->unbind();
            if (triObj.vbos.size()) {
                vbounbind(triObj.vbos);
            } else {
                vbounbind(vbos);
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
        auto vert =
#include "shader/tris.vert"
        ;

        auto frag =
#include "shader/tris.frag"
        ;

        return scene->shaderMan->compile_program(vert, frag);
    }

    Program *get_edge_program() {
        auto vert =
#include "shader/edge.vert"
        ;

        auto frag =
#include "shader/edge.frag"
        ;
        return scene->shaderMan->compile_program(vert, frag);
    }
};

}

void MakeGraphicVisitor::visit(zeno::PrimitiveObject *obj) {
     this->out_result = std::make_unique<ZhxxGraphicPrimitive>(this->in_scene, obj);
}

} // namespace zenovis
