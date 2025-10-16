#include <memory>
#include <string>
#include <vector>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/types/InstancingObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/geo/commonutil.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/ticktock.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/helper.h>
#include <zenovis/Camera.h>
#include <zenovis/DrawOptions.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zenovis/opengl/texture.h>
#include <GL/freeglut.h>

namespace zenovis {
namespace {

using namespace opengl;

struct ZhxxDrawObject {
    std::vector<std::unique_ptr<Buffer>> vbos;
    std::unique_ptr<Buffer> ebo;
    size_t count = 0;
    Program *prog{};
};

using CHAR_VBO_DATA = std::vector<glm::vec3>;

struct CHAR_VBO_INFO {
    glm::vec3 pos;
    CHAR_VBO_DATA m_data;
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
    Scene *scene = nullptr;
    std::vector<std::unique_ptr<Buffer>> vbos = std::vector<std::unique_ptr<Buffer>>(5);
    size_t vertex_count = 0;
    bool draw_all_points = false;

    std::unique_ptr<Buffer> ptnum_vbo;
    std::vector<CHAR_VBO_INFO> m_ptnum_data;
    Program* ptnums_prog = nullptr;

    std::unique_ptr<Buffer> facenum_vbo;
    std::vector<CHAR_VBO_INFO> m_facenum_data;
    Program* facenums_prog = nullptr;

    //Program *points_prog;
    //std::unique_ptr<Buffer> points_ebo;
    size_t points_count = 0;

    //Program *lines_prog;
    //std::unique_ptr<Buffer> lines_ebo;
    size_t lines_count = 0;

    //Program *tris_prog;
    //std::unique_ptr<Buffer> tris_ebo;
    size_t tris_count = 0;

    bool invisible = false;
    bool custom_color = false;

    ZhxxDrawObject pointObj;
    ZhxxDrawObject lineObj;
    ZhxxDrawObject triObj;
    std::vector<std::unique_ptr<Texture>> textures;

    ZhxxDrawObject polyEdgeObj = {};
    ZhxxDrawObject polyUvObj = {};

    explicit ZhxxGraphicPrimitive(Scene *scene_, zeno::PrimitiveObject *prim)
        : scene(scene_)
    {
        invisible = prim->userData()->get_bool("invisible", 0);
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

        if (scene->is_show_ptnum()) {
            init_facenum_data(prim);
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
        if (int subdlevs = prim->userData()->get_int("delayedSubdivLevels", 0)) {
            // todo: zhxx, should comp normal after subd or before?
#if 0
            zeno::log_trace("computing subdiv {}", subdlevs);
            (void)zeno::TempNodeSimpleCaller("OSDPrimSubdiv")
                .set("prim", std::make_shared<zeno::PrimitiveObject>(*prim))
                .set2<int>("levels", subdlevs)
                .set2<std::string>("edgeCreaseAttr", "")
                .set2<bool>("triangulate", false)
                .set2<bool>("asQuadFaces", true)
                .set2<bool>("hasLoopUVs", true)
                .set2<bool>("delayTillIpc", false)
                .call();  // will inplace subdiv prim
            prim->userData()->del("delayedSubdivLevels");
#endif
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
        auto ud = prim->userData();
        if (ud->get_int("isImage", 0)) {
            draw_all_points = false;
        }
        if (draw_all_points) {
            pointObj.prog = get_points_program();
        }
        if (scene->is_show_ptnum()) {
            init_ptnum_data(pos);
        }
    }

    explicit ZhxxGraphicPrimitive(Scene* scene_, zeno::GeometryObject* geo)
        : scene(scene_) {
#if 0
        zeno::log_trace("rendering primitive size {}", geo->npoints());

        const std::vector<zeno::vec3f>& points = geo->points_pos();
        std::vector<zeno::vec3f> clr, nrms, uv, tang;
        std::vector<zeno::vec3i> tris;

        bool any_not_triangle = !geo->is_base_triangle();
        if (any_not_triangle) {
            std::vector<int> edges = geo->edge_list();

            polyEdgeObj.count = edges.size();
            polyEdgeObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
            polyEdgeObj.ebo->bind_data(edges.data(), edges.size() * sizeof(edges[0]));
            auto vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);

            vbo->bind_data(points.data(), points.size() * sizeof(points[0]));
            polyEdgeObj.vbos.push_back(std::move(vbo));
            polyEdgeObj.prog = get_edge_program();
        }
        if (geo->nfaces() > 0) {
            //还是需要展开成三角形
            tris = geo->tri_indice();
        }

        if (any_not_triangle && geo->has_attr(zeno::ATTR_POINT, "uvs")) {
            //TODO: uvs
        }
        else {
            std::fill(uv.begin(), uv.end(), zeno::vec3f(0.0f));
        }

        if (geo->has_attr(zeno::ATTR_POINT, "clr")) {
            clr = geo->get_attr<zeno::vec3f>(zeno::ATTR_POINT, "clr");
        }
        else {
            zeno::vec3f clr0(1.0f);
            std::fill(clr.begin(), clr.end(), clr0);
        }

        if (geo->has_attr(zeno::ATTR_POINT, "nrm")) {
            nrms = geo->get_attr<zeno::vec3f>(zeno::ATTR_POINT, "nrm");
        }
        else {
            //TODO: calculate normals by util function.
            std::fill(nrms.begin(), nrms.end(), zeno::vec3f(1.0f, 0.0f, 0.0f));
        }

        if (geo->has_attr(zeno::ATTR_POINT, "tang")) {
            tang = geo->get_attr<zeno::vec3f>(zeno::ATTR_POINT, "tang");
        }
        else {
            std::fill(tang.begin(), tang.end(), zeno::vec3f(0.0f));
        }

        vbos[0] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[0]->bind_data(points.data(), points.size() * sizeof(points[0]));
        vbos[1] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[1]->bind_data(clr.data(), clr.size() * sizeof(clr[0]));
        vbos[2] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[2]->bind_data(nrms.data(), nrms.size() * sizeof(nrms[0]));
        vbos[3] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[3]->bind_data(uv.data(), uv.size() * sizeof(uv[0]));
        vbos[4] = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        vbos[4]->bind_data(tang.data(), tang.size() * sizeof(tang[0]));

        //TODO: case of points.

        //TODO: case of lines.

        tris_count = tris.size();
        if (tris_count > 0) {
            triObj.count = tris_count;
            triObj.ebo = std::make_unique<Buffer>(GL_ELEMENT_ARRAY_BUFFER);
            triObj.ebo->bind_data(tris.data(), tris_count * sizeof(tris[0]));
            triObj.prog = get_tris_program();
        }

        draw_all_points = false;
#endif
    }

    std::vector<CHAR_VBO_INFO> gen_vbo_info(const std::string& drawNum, const zeno::vec3f& basepos) {
        std::vector<CHAR_VBO_INFO> arrs;
        GLfloat arr[1024];
        int size = 0;
        int arr_split[64];
        int split_size = 0;
        glutGetStrokeString(GLUT_STROKE_MONO_ROMAN, drawNum.c_str(), arr, &size, arr_split, &split_size);

        //找出整个字符串数字顶点的包围盒
        GLfloat xmin = 100000, ymin = 100000, xmax = -10000, ymax = -10000;
        for (int i = 0; i < size; i++) {
            if (i % 2 == 0) {
                xmin = std::min(arr[i], xmin);
                xmax = std::max(arr[i], xmax);
            }
            else {
                ymin = std::min(arr[i], ymin);
                ymax = std::max(arr[i], ymax);
            }
        }
        //TODO:用于缩放glut导出的字体大小，经验值设定。
        float xscale, yscale;
        if (drawNum.length() > 1) {
            xscale = 0.01;
            yscale = 0.02;
        }
        else {
            xscale = 0.004;
            yscale = 0.02;
        }
        GLfloat width = xmax - xmin, height = ymax - ymin;
        for (int i = 0; i < size; i++) {
            if (i % 2 == 0) {
                GLfloat xp = arr[i];
                xp -= xmin;
                xp /= (width / 2);
                xp *= xscale;
                arr[i] = xp;
            }
            else {
                GLfloat yp = arr[i];
                yp -= ymin;
                yp /= (height * 1.5);
                yp *= yscale;
                arr[i] = yp;
            }
        }

        for (int j = 0; j < split_size; j++) {
            int i_start = (j == 0) ? 0 : arr_split[j - 1];
            int i_end = arr_split[j];
            int mem_size = i_end - i_start;
            assert(mem_size % 2 == 0);
            int nVertexs = mem_size / 2;
            CHAR_VBO_DATA mem(nVertexs);
            for (int k = 0; k < nVertexs; k++) {
                GLfloat xp = arr[i_start + k * 2];
                GLfloat yp = arr[i_start + k * 2 + 1];
                GLfloat zp = 0;
                //先放置在原点，待会再实施旋转和平移
                mem[k] = glm::vec3(xp, yp, zp);
            }
            CHAR_VBO_INFO info;
            info.m_data = mem;
            info.pos = glm::vec3(basepos[0], basepos[1], basepos[2]);
            arrs.emplace_back(info);
        }
        return arrs;
    }

    void init_ptnum_data(const std::vector<zeno::vec3f>& pos) {
        ptnum_vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        for (int idxPoint = 0; idxPoint < pos.size(); idxPoint++)
        {
            zeno::vec3f basepos = pos[idxPoint];
//DEBUG:
#if 0
            if (idxPoint > 20) {
                break;
            }
#endif

            std::string drawNum = std::to_string(idxPoint);
            auto vec = gen_vbo_info(drawNum, basepos);
            m_ptnum_data.insert(m_ptnum_data.end(), vec.begin(), vec.end());
        }
        ptnums_prog = get_ptnum_program();
    }

    void init_facenum_data(zeno::PrimitiveObject* prim) {
        facenum_vbo = std::make_unique<Buffer>(GL_ARRAY_BUFFER);
        auto& pos = prim->verts;
        if (prim->tris.size() > 0) {
            for (int idxFace = 0; idxFace < prim->tris.size(); idxFace++) {
                zeno::vec3i indice = prim->tris[idxFace];
                zeno::vec3f p1(pos[indice[0]]), p2(pos[indice[1]]), p3(pos[indice[2]]);
                zeno::vec3f basepos = (p1 + p2 + p3) / 3;

                std::string drawNum = std::to_string(idxFace);
                auto vec = gen_vbo_info(drawNum, basepos);
                m_facenum_data.insert(m_facenum_data.end(), vec.begin(), vec.end());
            }
        }
        else if (prim->loops.size() > 0) {
            for (int idxFace = 0; idxFace < prim->polys.size(); idxFace++)
            {
                //DEBUG:
#if 0
                if (idxFace > 20) {
                    break;
                }
#endif

                auto& [startIdx, sz] = prim->polys[idxFace];
                zeno::vec3f total(0, 0, 0);
                for (int i = 0; i < sz; i++) {
                    int idxPt = prim->loops[startIdx + i];
                    zeno::vec3f pt = pos[idxPt];
                    total += pt;
                }
                zeno::vec3f basepos = total / sz;

                std::string drawNum = std::to_string(idxFace);
                auto vec = gen_vbo_info(drawNum, basepos);
                m_facenum_data.insert(m_facenum_data.end(), vec.begin(), vec.end());
            }
        }
        facenums_prog = get_facenum_program();
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

        if (scene->is_show_ptnum()) {
            draw_ptnums();
            draw_facenums();
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
            {
                auto camera_center = scene->camera->m_pos;
                triObj.prog->set_uniform("mCameraCenter", camera_center);
            }

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

    void print_help(void)
    {
        //TODO: 以贴图方式取代画点线
        int i;
        const char* s, ** text;

        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        int win_width = scene->camera->m_nx;
        int win_height = scene->camera->m_ny;
        glOrtho(0, win_width, 0, win_height, -1, 1);

        static const char* helpprompt[] = { "Press F1 for help", 0 };
        text = helpprompt;

        for (i = 0; text[i]; i++) {
            glColor3f(0, 0.1, 0);
            glRasterPos2f(7, win_height - (i + 1) * 20 - 2);
            s = text[i];
            while (*s) {
                glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *s++);
            }
            glColor3f(0, 0.9, 0);
            glRasterPos2f(5, win_height - (i + 1) * 20);
            s = text[i];
            while (*s) {
                glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *s++);
            }
        }

        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        glPopAttrib();
    }

    glm::mat4 get_proper_view_matrix() {
        glm::vec3 lodfront = scene->camera->get_lodfront();
        glm::vec3 lodup = scene->camera->get_lodup();
        glm::vec3 cam_pos = scene->camera->getPos();
        glm::vec3 pivot = scene->camera->getPivot();

        //zeno::log_info("camera pos: x={}, y={}, z={}", cam_pos[0], cam_pos[1], cam_pos[2]);

        glm::vec3 _uz = glm::normalize(cam_pos);
        glm::vec3 _uy = glm::normalize(lodup);
        glm::vec3 _ux = glm::normalize(glm::cross(_uy, _uz));

        glm::vec4 uz(_uz[0], _uz[1], _uz[2], 0);
        glm::vec4 uy(_uy[0], _uy[1], _uy[2], 0);
        glm::vec4 ux(_ux[0], _ux[1], _ux[2], 0);

        float scale_factor = glm::length(cam_pos) * 1.0;
        glm::vec3 scale_cam(scale_factor);

        glm::mat4 scaleM(glm::vec4(scale_factor, 0, 0, 0), glm::vec4(0, scale_factor, 0, 0),
            glm::vec4(0, 0, scale_factor, 0), glm::vec4(0, 0, 0, 1));
        glm::mat4 rotateM(ux, uy, uz, glm::vec4(0, 0, 0, 1));
        glm::mat4 rsM = rotateM * scaleM;
        return rsM;
    }

    void draw_elemnums(Program* prog, std::vector<CHAR_VBO_INFO>& vbo_datas, Buffer* pVBO) {
        prog->use();
        scene->camera->set_program_uniforms(prog);
        glm::vec3 cam_pos = scene->camera->getPos();
        glm::mat4 rsM = get_proper_view_matrix();

        //zeno::log_info("camera abs pos: x={}, y={}, z={}", scale_cam[0], scale_cam[1], scale_cam[2]);

        //TODO: 挪到shader
        for (CHAR_VBO_INFO& vbo_data : vbo_datas) {
            int n = vbo_data.m_data.size();
            glm::vec4 trans(vbo_data.pos[0], vbo_data.pos[1], vbo_data.pos[2], 1);
            rsM[3] = trans;

            CHAR_VBO_DATA new_vbodata(n);
            for (int i = 0; i < n; i++) {
                auto& _pos = vbo_data.m_data[i];
                glm::vec4 pos(_pos[0], _pos[1], _pos[2], 1);
                glm::vec4 newpos = rsM * pos;
                newpos += (glm::vec4(cam_pos[0], cam_pos[1], cam_pos[2], 1) - newpos) * 0.01f;   //防止遮挡
                new_vbodata[i] = glm::vec3(newpos[0], newpos[1], newpos[2]);
            }
            pVBO->bind();
            pVBO->bind_data(new_vbodata.data(), new_vbodata.size() * sizeof(new_vbodata[0]));
            pVBO->attribute(0, 0, sizeof(GLfloat) * 3, GL_FLOAT, 3);
            CHECK_GL(glDrawArrays(GL_LINE_STRIP, 0, new_vbodata.size()));
            pVBO->unbind();
        }
    }

    void draw_ptnums() {
        if (ptnums_prog && !m_ptnum_data.empty() && ptnum_vbo) {
            draw_elemnums(ptnums_prog, m_ptnum_data, ptnum_vbo.get());
        }
    }

    void draw_facenums() {
        if (facenums_prog && !m_facenum_data.empty() && facenums_prog) {
            draw_elemnums(facenums_prog, m_facenum_data, facenum_vbo.get());
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

    Program* get_ptnum_program() {
        auto vert =
#include "shader/ptnum.vert"
            ;
        auto frag =
#include "shader/ptnum.frag"
            ;
        return scene->shaderMan->compile_program(vert, frag);
    }

    Program* get_facenum_program() {
        auto vert =
#include "shader/facenum.vert"
            ;
        auto frag =
#include "shader/facenum.frag"
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

void MakeGraphicVisitor::visit(zeno::GeometryObject_Adapter *obj) {
    //考虑到primitive又要创建各种nrm uv，可以在这里让PrimitiveObject中转一下
    this->out_result = std::make_unique<ZhxxGraphicPrimitive>(this->in_scene, obj->toPrimitiveObject().get());
}

} // namespace zenovis
