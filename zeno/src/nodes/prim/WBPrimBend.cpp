//
// Created by WangBo on 2022/7/5.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/CurveObject.h>
#include <zeno/types/ListObject.h>

#include <zeno/utils/orthonormal.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/log.h>

#include <glm/gtx/quaternion.hpp>
#include <random>
#include <sstream>
#include <ctime>
#include <iostream>
namespace zeno
{
namespace
{

struct WBPrimBend : INode {
    void apply() override
    {
        auto limitDeformation = get_input<NumericObject>("Limit Deformation")->get<int>();
        auto symmetricDeformation = get_input<NumericObject>("Symmetric Deformation")->get<int>();
        auto angle = get_input<NumericObject>("Bend Angle (degree)")->get<float>();
        auto upVector = has_input("Up Vector") ? get_input<NumericObject>("Up Vector")->get<zeno::vec3f>() : vec3f(0, 1, 0);
        auto capOrigin = has_input("Capture Origin") ? get_input<NumericObject>("Capture Origin")->get<zeno::vec3f>() : vec3f(0, 0, 0);
        auto dirVector = has_input("Capture Direction") ? get_input<NumericObject>("Capture Direction")->get<zeno::vec3f>() : vec3f(0, 0, 1);
        double capLen = has_input("Capture Length") ? get_input<NumericObject>("Capture Length")->get<float>() : 1.0;

        glm::vec3 up = normalize(glm::vec3(upVector[0], upVector[1], upVector[2]));
        glm::vec3 dir = normalize(glm::vec3(dirVector[0], dirVector[1], dirVector[2]));
        glm::vec3 axis1 = normalize(cross(dir, up));
        glm::vec3 axis2 = cross(axis1, dir);
        glm::vec3 origin = glm::vec3(capOrigin[0], capOrigin[1], capOrigin[2]);
        double rotMatEle[9] = { dir.x, dir.y, dir.z,
                                axis2.x, axis2.y, axis2.z,
                                axis1.x, axis1.y, axis1.z };
        glm::mat3 rotMat = glm::make_mat3x3(rotMatEle);
        glm::mat3 inverse = glm::transpose(rotMat);
        auto prim = get_input<PrimitiveObject>("prim");
        auto& pos = prim->verts;

#pragma omp parallel for
        for (intptr_t i = 0; i < prim->verts.size(); i++)
        {
            glm::vec3 original = glm::vec3(pos[i][0], pos[i][1], pos[i][2]);
            glm::vec3 deformedPos = original;

            original -= origin;
            deformedPos -= origin;

            deformedPos = inverse * deformedPos;
            original = inverse * original;

            double bend_threshold = 0.005;
            if (std::abs(angle) >= bend_threshold)
            {
                double angleRad = angle * M_PI / 180;
                double bend = angleRad * (deformedPos.x / capLen);
                glm::vec3 N = { 0, 1, 0 };
                glm::vec3 center = (float)(capLen / angleRad) * N;
                double d = deformedPos.x / capLen;
                if (symmetricDeformation)
                {
                    if (limitDeformation && std::abs(deformedPos.x) > capLen)
                    {
                        bend = angleRad;
                        d = 1;
                        if (-deformedPos.x > capLen)
                        {
                            bend *= -1;
                            d *= -1;
                        }
                    }
                }
                else
                {
                    if (deformedPos.x * capLen < 0)
                    {
                        bend = 0;
                        d = 0;
                    }
                    if (limitDeformation && deformedPos.x > capLen)
                    {
                        bend = angleRad;
                        d = 1;
                    }
                }
                double cb = std::cos(bend);
                double sb = std::sin(bend);
                double bendMatEle[9] = { cb, sb, 0,
                                         -sb, cb, 0,
                                         0, 0, 1 };
                glm::mat3 bendRotMat = glm::make_mat3x3(bendMatEle);
                original.x -= d * capLen;
                original -= center;
                original = bendRotMat * original;
                original += center;
            }
            deformedPos = rotMat * original + origin;
            pos[i] = vec3f(deformedPos.x, deformedPos.y, deformedPos.z);
        }
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(WBPrimBend,
           { /* inputs: */ {
                   "prim",
                   {"int", "Limit Deformation", "1"},
                   {"int", "Symmetric Deformation", "0"},
                   {"float", "Bend Angle (degree)", "90"},
                   {"vec3f", "Up Vector", "0,1,0"},
                   {"vec3f", "Capture Origin", "0,0,0"},
                   {"vec3f", "Capture Direction", "0,0,1"},
                   {"float", "Capture Length", "1.0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               }});

struct CreateCircle : INode {
    void apply() override
    {
        auto seg = get_input<NumericObject>("segments")->get<int>();
        auto r = get_input<NumericObject>("r")->get<float>();
        auto prim = std::make_shared<PrimitiveObject>();

        for (int i = 0; i < seg; i++)
        {
            float rad = 2.0 * M_PI * i / seg;
            prim->verts.push_back(vec3f(cos(rad) * r, 0, -sin(rad) * r));
        }
        prim->verts.push_back(vec3i(0, 0, 0));

        for (int i = 0; i < seg; i++)
        {
            prim->tris.push_back(vec3i(seg, i, (i + 1) % seg));
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(CreateCircle,
           {  /* inputs: */ {
                   {"int","segments","32"},
                   {"float","r","1.0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               } });

struct ParameterizeLine : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if(!prim->lines.has_attr("parameterization")) {
            prim->lines.add_attr<float>("parameterization");
            float total = 0;
            std::vector<float> linesLen(prim->lines.size());
            for (size_t i = 0; i < prim->lines.size(); i++) {
                auto const &ind = prim->lines[i];
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto area = length(b - a);
                total += area;
                linesLen[i] = total;
            }
            auto inv_total = 1 / total;
            for (size_t i = 0; i < prim->lines.size(); i++) {
                prim->lines.attr<float>("parameterization")[i] = linesLen[i] * inv_total;
            }
        }
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(ParameterizeLine,
           {  /* inputs: */ {
                "prim",
            }, /* outputs: */ {
                "prim",
            }, /* params: */ {
            }, /* category: */ {
                "primitive",
            } });
struct LineResample : INode {
    void apply() override
    {
        auto prim = get_input<PrimitiveObject>("prim");

        auto segments = get_input<NumericObject>("segments")->get<int>();
        if (segments < 1) { segments = 1; }
        std::vector<float> linesLen(prim->lines.size());
        if(!prim->lines.has_attr("parameterization")) {
            prim->lines.add_attr<float>("parameterization");
            float total = 0;
            for (size_t i = 0; i < prim->lines.size(); i++) {
                auto const &ind = prim->lines[i];
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto area = length(b - a);
                total += area;
                linesLen[i] = total;
            }
            auto inv_total = 1 / total;
            for (auto &_linesLen : linesLen) {
                _linesLen *= inv_total;
            }
            for (size_t i=0; i<prim->lines.size();i++) {
                prim->lines.attr<float>("parameterization")[i] = linesLen[i];
            }
        } else {
#pragma omp parallel for
            for (auto i=0; i<prim->lines.size();i++) {
                linesLen[i] = prim->lines.attr<float>("parameterization")[i];
            }
        }

        auto retprim = std::make_shared<PrimitiveObject>();
        if(has_input("PrimSampler")) {
            retprim = get_input<PrimitiveObject>("PrimSampler");
            auto sampleByAttr = get_input2<std::string>("SampleBy");
            auto& sampleBy = retprim->attr<float>(sampleByAttr);
            retprim->add_attr<float>("t");
            auto &t_arr = retprim->attr<float>("t");
#pragma omp parallel for
            for(auto i=0; i<retprim->size();i++) {
                t_arr[i] = sampleBy[i];
            }
        } else {
            retprim->resize(segments + size_t(1));
            retprim->lines.resize(segments);
            retprim->add_attr<float>("t");
            auto &t_arr = retprim->attr<float>("t");
#pragma omp parallel for
            for (auto i=0; i<retprim->size(); i++) {
                t_arr[i] = (float)i / float(segments);
            }
#pragma omp parallel for
            for (auto i=0; i<segments; i++) {
                retprim->lines[i] = zeno::vec2i(i, i+1);
            }
        }
        for(auto key:prim->attr_keys())
        {
            if(key!="pos")
                std::visit([&retprim, key](auto &&ref) {
                    using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                    retprim->add_attr<T>(key);
                }, prim->attr(key));
        }
#pragma omp parallel for
        for(auto i=0; i<retprim->size();i++) {
            float insertU = retprim->attr<float>("t")[i];
            auto it = std::upper_bound(linesLen.begin(), linesLen.end(), insertU);
            size_t index = it - linesLen.begin();
            index = std::min(index, prim->lines.size() - 1);

            // if (index <= 0) continue;

            auto const& ind = prim->lines[index];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto r1 = (insertU - linesLen[index - 1]) / (linesLen[index] - linesLen[index - 1]);
            retprim->verts[i] = a + (b - a) * r1;
            for(auto key:prim->attr_keys())
            {
                if(key!="pos")
                    std::visit([&retprim, &prim, &ind, &r1, &i, key](auto &&ref) {
                        using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                        auto a = prim->attr<T>(key)[ind[0]];
                        auto b = prim->attr<T>(key)[ind[1]];
                        retprim->attr<T>(key)[i] = a + (b-a)*r1;
                    }, prim->attr(key));
            }
        }
//
//        auto& cu = retprim->add_attr<float>("curveU");
//        for (int idx = 0; idx <= segments; idx++)
//        {
//            float delta = 1.0f / float(segments);
//            float insertU = float(idx) * delta;
//            if (idx == segments)
//            {
//                insertU = 1;
//            }
//
//            auto it = std::lower_bound(linesLen.begin(), linesLen.end(), insertU);
//            size_t index = it - linesLen.begin();
//            index = std::min(index, prim->lines.size() - 1);
//
//            auto const& ind = prim->lines[index];
//            auto a = prim->verts[ind[0]];
//            auto b = prim->verts[ind[1]];
//
//            auto r1 = (insertU - linesLen[index - 1]) / (linesLen[index] - linesLen[index - 1]);
//            auto p = a + (b - a) * r1;
//
//            retprim->verts[idx] = p;
//            cu[idx] = insertU;
//        }
//
//        retprim->lines.reserve(retprim->size());
//        for (int i = 1; i < retprim->size(); i++) {
//            retprim->lines.emplace_back(i - 1, i);
//        }

        set_output("prim", std::move(retprim));
    }
};
ZENDEFNODE(LineResample,
    {  /* inputs: */ {
            "prim",
            {"int", "segments", "3"},
            "PrimSampler",
            {"string", "SampleBy", "t"},
        }, /* outputs: */ {
            "prim",
        }, /* params: */ {
        }, /* category: */ {
            "primitive",
        } });

struct CurveOrientation  : INode {
    void apply() override
    {
        auto prim = get_input<PrimitiveObject>("prim");
        auto dirName = get_input2<std::string>("dirName");
        auto tanName = get_input2<std::string>("tanName");
        auto bitanName = get_input2<std::string>("bitanName");
        auto &directions = prim->add_attr<zeno::vec3f>(dirName);
        auto &bidirections = prim->add_attr<zeno::vec3f>(tanName);
        auto &tridirections = prim->add_attr<zeno::vec3f>(bitanName);
        size_t n = prim->size();
#pragma omp parallel for
        for (intptr_t i = 1; i < n - 1; i++) {
            auto lastpos = prim->verts[i - 1];
            auto currpos = prim->verts[i];
            auto nextpos = prim->verts[i + 1];
            auto direction = normalize(nextpos - lastpos);
            directions[i] = direction;
        }
        directions[0] = normalize(prim->verts[1] - prim->verts[0]);
        directions[n - 1] = normalize(prim->verts[n - 1] - prim->verts[n - 2]);
        orthonormal first_orb(directions[0]);
        directions[0] = first_orb.tangent;
        bidirections[0] = first_orb.bitangent;
        tridirections[0] = normalize(cross(directions[0], bidirections[0]));
        vec3f last_tangent = directions[0];
        for (intptr_t i = 1; i < n; i++) {
            orthonormal orb(directions[i], last_tangent);
            last_tangent = directions[i] = orb.tangent;
            bidirections[i] = orb.bitangent;
            tridirections[i] = normalize(cross(directions[i], bidirections[i]));
        }
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(CurveOrientation,
           {  /* inputs: */ {
                "prim",
                {"string", "dirName", "dir"},
                {"string", "tanName", "tan"},
                {"string", "bitanName", "bitan"},
            }, /* outputs: */ {
                "prim",
            }, /* params: */ {

            }, /* category: */ {
                "primitive",
            } });
struct LineCarve : INode {
    void apply() override
    {
        auto prim = get_input<PrimitiveObject>("prim");
        auto insertU = get_input<NumericObject>("insertU")->get<float>();

        float total = 0;
        std::vector<float> linesLen(prim->lines.size());
        for (size_t i = 0; i < prim->lines.size(); i++) {
            auto const& ind = prim->lines[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto area = length(b - a);
            total += area;
            linesLen[i] = total;
        }
        auto inv_total = 1 / total;
        for (auto& _linesLen : linesLen) { _linesLen *= inv_total; }

        insertU = std::min(1.f, std::max(0.f, insertU));
        auto it = std::lower_bound(linesLen.begin(), linesLen.end(), insertU);
        size_t index = it - linesLen.begin();
        index = std::min(index, prim->lines.size() - 1);
        if (index < 0) index = 0;

        auto const& ind = prim->lines[index];
        auto a = prim->verts[ind[0]];
        auto b = prim->verts[ind[1]];

        auto r1 = (insertU - linesLen[index - 1]) / (linesLen[index] - linesLen[index - 1]);
        auto p = a + (b - a) * r1;

        prim->verts.reserve(prim->size() + 1);
        prim->attr<zeno::vec3f>("pos").insert(prim->verts.begin() + int(index) + 1, { p[0], p[1], p[2] });

        linesLen.reserve(linesLen.size() + 1);
        linesLen.insert(linesLen.begin() + int(index), insertU);

        if (!prim->has_attr("curveU")) {
            prim->add_attr<float>("curveU");
        }
        auto& cu = prim->verts.attr<float>("curveU");
        cu[0] = 0.0f;
        for (size_t i = 1; i < prim->size(); i++)
        {
            cu[i] = linesLen[i - 1];
        }

        if (get_input<NumericObject>("cut")->get<int>())
        {
            if (get_input<NumericObject>("cut insert to end")->get<int>())
            {
                if (prim->verts.begin() + int(index) <= prim->verts.end())
                {
                    prim->verts->erase(prim->verts.begin() + int(index) + 2, prim->verts.end());
                    cu.erase(cu.begin() + int(index) + 2, cu.end());
                }
            }
            else
            {
                if (prim->verts.begin() + int(index) <= prim->verts.end())
                {
                    prim->verts->erase(prim->verts.begin(), prim->verts.begin() + int(index) + 1);
                    cu.erase(cu.begin(), cu.begin() + int(index) + 1);
                }
            }
        }

        prim->lines.clear();
        size_t line_n = prim->verts.size();
        prim->lines.reserve(line_n);
        for (size_t i = 1; i < line_n; i++) {
            prim->lines.emplace_back(i - 1, i);
        }
        prim->lines.update();

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(LineCarve,
    {  /* inputs: */ {
            "prim",
            {"float", "insertU", "0.15"},
            {"bool", "cut", "0"},
            {"bool", "cut insert to end", "1"},
        }, /* outputs: */ {
            "prim",
        }, /* params: */ {

        }, /* category: */ {
            "primitive",
        } });


///////////////////////////////////////////////////////////////////////////////
// 2022.07 VisVector
///////////////////////////////////////////////////////////////////////////////
struct VisVec3Attribute : INode {
    void apply() override
    {
        auto color = get_input<NumericObject>("color")->get<zeno::vec3f>();
        auto useNormalize = get_input<NumericObject>("normalize")->get<int>();
        auto lengthScale = get_input<NumericObject>("lengthScale")->get<float>();
        auto name = get_input2<std::string>("name");

        auto prim = get_input<PrimitiveObject>("prim");
        auto& attr = prim->verts.attr<zeno::vec3f>(name);
        auto& pos = prim->verts;

        auto primVis = std::make_shared<PrimitiveObject>();
        primVis->verts.resize(prim->size() * 2);
        primVis->lines.resize(prim->size());
        for (auto key : prim->attr_keys()) {
            if (key != "pos")
                std::visit(
                    [&primVis, key](auto &&ref) {
                      using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                      primVis->add_attr<T>(key);
                    },
                    prim->attr(key));
        }
        auto& visColor = primVis->verts.add_attr<zeno::vec3f>("clr");
        auto& visPos = primVis->verts;

#pragma omp parallel for
        for (int iPrim = 0; iPrim < prim->size(); iPrim++)
        {
            int i = iPrim * 2;
            visPos[i] = pos[iPrim];
            visColor[i] = color;
            for (auto key : prim->attr_keys()) {
                if (key != "pos")
                    std::visit(
                        [i, iPrim](auto &&dst, auto &&src) {
                          using DstT = std::remove_cv_t<std::remove_reference_t<decltype(dst)>>;
                          using SrcT = std::remove_cv_t<std::remove_reference_t<decltype(src)>>;
                          if constexpr (std::is_same_v<DstT, SrcT>) {
                              dst[i] = src[iPrim];
                              dst[i + 1] = src[iPrim];
                          } else {
                              throw std::runtime_error("the same attr of both primitives are of different types.");
                          }
                        },
                        primVis->attr(key), prim->attr(key));
            }
            ++i;
            auto a = attr[iPrim];
            if (useNormalize) a = normalize(a);
            visPos[i] = pos[iPrim] + a * lengthScale;
            visColor[i - 1] = color;
            visColor[i] = color * 0.25;
            primVis->lines[i / 2][0] = i - 1;
            primVis->lines[i / 2][1] = i;
        }

        set_output("primVis", std::move(primVis));
    }
};
ZENDEFNODE(VisVec3Attribute,
           {  /* inputs: */ {
                   "prim",
                   {"string", "name", "vel"},
                   {"bool", "normalize", "1"},
                   {"float", "lengthScale", "1.0"},
                   {"vec3f", "color", "1,1,0"},
               }, /* outputs: */ {
                   "primVis",
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});

struct TracePositionOneStep : INode {
    void apply() override
    {
        auto primData = get_input<PrimitiveObject>("primData");

        auto primVis = get_input<PrimitiveObject>("primStart");

        auto idName = get_input2<std::string>("lineTag");
        if (!primVis->verts.has_attr(idName))
        {
            primVis->verts.add_attr<int>(idName);
        }

        auto primVisVertsCount = primVis->verts.size();
        auto primVislinesCount = primVis->lines.size();
        auto primDataVertsCount = primData->verts.size();

        auto& attr_idName = primVis->verts.attr<int>(idName);
        if (primVisVertsCount != 0)
        {
            if (primVislinesCount == 0)
            {
                for (int i = 0; i < primVisVertsCount; i++)
                {
                    attr_idName[i] = i;
                }
            }
            primVis->lines.resize(size_t(primVislinesCount) + size_t(primDataVertsCount));
        }

        primVis->verts.resize(size_t(primVisVertsCount) + size_t(primDataVertsCount));

        auto& visPos = primVis->verts;
        auto& dataPos = primData->verts;
        auto& visLines = primVis->lines;
#pragma omp parallel for
        for (int i = 0; i < primDataVertsCount; i++)
        {
            visPos[primVisVertsCount + i] = dataPos[i];
            attr_idName[primVisVertsCount + i] = i;
            if (primVisVertsCount != 0)
            {
                visLines[primVislinesCount + i][0] = primVisVertsCount + i - primDataVertsCount;
                visLines[primVislinesCount + i][1] = primVisVertsCount + i;
            }
        }

        set_output("primVis", std::move(primVis));
    }
};
ZENDEFNODE(TracePositionOneStep,
           {  /* inputs: */ {
                   "primData",
                   "primStart",
                   {"string", "lineTag", "lineID"},
               }, /* outputs: */ {
                   "primVis",
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});

struct PrimCopyAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto sourceName = get_input2<std::string>("sourceName");
        auto targetName = get_input2<std::string>("targetName");
        auto scope = get_input2<std::string>("scope");

        if (scope == "vert") {
            if (!prim->verts.has_attr(sourceName)) {
                zeno::log_error("verts no such attr named '{}'.", sourceName);
            }
            prim->verts.attr_visit<AttrAcceptAll>(sourceName, [&] (auto const &attarr) {
                using T = std::decay_t<decltype(attarr[0])>;
                auto &targetAttr = prim->verts.template add_attr<T>(targetName);
                std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
            });
        }
        else if (scope == "tri") {
            if (!prim->tris.has_attr(sourceName)) {
                zeno::log_error("tris no such attr named '{}'.", sourceName);
            }
            prim->tris.attr_visit<AttrAcceptAll>(sourceName, [&] (auto const &attarr) {
                using T = std::decay_t<decltype(attarr[0])>;
                auto &targetAttr = prim->tris.template add_attr<T>(targetName);
                std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
            });
        }
        else if (scope == "loop") {
            if (!prim->loops.has_attr(sourceName)) {
                zeno::log_error("loops no such attr named '{}'.", sourceName);
            }
            prim->loops.attr_visit<AttrAcceptAll>(sourceName, [&] (auto const &attarr) {
                using T = std::decay_t<decltype(attarr[0])>;
                auto &targetAttr = prim->loops.template add_attr<T>(targetName);
                std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
            });
        }
        else if (scope == "poly") {
            if (!prim->polys.has_attr(sourceName)) {
                zeno::log_error("polys no such attr named '{}'.", sourceName);
            }
            prim->polys.attr_visit<AttrAcceptAll>(sourceName, [&] (auto const &attarr) {
                using T = std::decay_t<decltype(attarr[0])>;
                auto &targetAttr = prim->polys.template add_attr<T>(targetName);
                std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
            });
        }
        else if (scope == "line") {
            if (!prim->lines.has_attr(sourceName)) {
                zeno::log_error("lines no such attr named '{}'.", sourceName);
            }
            prim->lines.attr_visit<AttrAcceptAll>(sourceName, [&] (auto const &attarr) {
                using T = std::decay_t<decltype(attarr[0])>;
                auto &targetAttr = prim->lines.template add_attr<T>(targetName);
                std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
            });
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimCopyAttr,
           { /* inputs: */ {
                   "prim",
                   {"string", "sourceName", "s"},
                   {"string", "targetName", "t"},
                   {"enum vert tri loop poly line", "scope", "vert"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               }});

char* getPrimRawData(PrimitiveObject* prim, std::string type, std::string name, std::string scope)
{
  std::cout<<type<<" "<<scope<<" "<<name<<std::endl;
  if(type == "int")
  {
    if(scope == "verts")
      return (char*)(prim->attr<int>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<int>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<int>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<int>(name).data());
  }
  if(type == "float")
  {
    if(scope == "verts")
      return (char*)(prim->attr<float>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<float>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<float>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<float>(name).data());
  }
  if(type == "vec2f")
  {
    if(scope == "verts")
      return (char*)(prim->attr<vec2f>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<vec2f>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<vec2f>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<vec2f>(name).data());
  }
  if(type == "vec2i")
  {
    if(scope == "verts")
      return (char*)(prim->attr<vec2i>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<vec2i>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<vec2i>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<vec2i>(name).data());
  }
  if(type == "vec3f")
  {
    if(scope == "verts")
      return (char*)(prim->attr<vec3f>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<vec3f>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<vec3f>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<vec3f>(name).data());
  }
  if(type == "vec3i")
  {
    if(scope == "verts")
      return (char*)(prim->attr<vec3i>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<vec3i>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<vec3i>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<vec3i>(name).data());
  }
  if(type == "vec4f")
  {
    if(scope == "verts")
      return (char*)(prim->attr<vec4f>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<vec4f>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<vec4f>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<vec4f>(name).data());
  }
  if(type == "vec4i")
  {
    if(scope == "verts")
      return (char*)(prim->attr<vec4i>(name).data());
    if(scope == "tris")
      return (char*)(prim->tris.attr<vec4i>(name).data());
    if(scope == "polys")
      return (char*)(prim->polys.attr<vec4i>(name).data());
    if(scope == "lines")
      return (char*)(prim->lines.attr<vec4i>(name).data());
  }
}

struct PrimTwoCopyAttr : INode {
  void apply() override {
    auto prim   = get_input<PrimitiveObject>("primTo");
    auto primFrom = get_input<PrimitiveObject>("primFrom");
    auto sourceName = get_input2<std::string>("NameFrom");
    auto targetName = get_input2<std::string>("NameTo");
    auto srcscope = get_input2<std::string>("ScopeFrom");
    auto tarscope = get_input2<std::string>("ScopeTo");
    auto type = get_input2<std::string>("type");

    char* dataToPtr;
    char* dataFromPtr;
    size_t size;

    dataToPtr = getPrimRawData(prim.get(), type, targetName, tarscope);
    dataFromPtr = getPrimRawData(primFrom.get(), type, sourceName, srcscope);
    if(srcscope=="verts")
      size = primFrom->verts.size();
    if(srcscope=="tris")
      size = primFrom->tris.size();
    if(srcscope=="lines")
      size = primFrom->lines.size();
    if(srcscope=="polys")
      size = primFrom->polys.size();

    if(type=="int")
      size *= 4;
    if(type=="float")
      size *= 4;
    if(type=="vec2f")
      size *= 8;
    if(type=="vec2i")
      size *= 8;
    if(type=="vec3f")
      size *= 12;
    if(type=="vec3i")
      size *= 12;
    if(type=="vec4f")
      size *= 16;
    if(type=="vec4i")
      size *= 16;

    memcpy(dataToPtr, dataFromPtr,size);

    set_output("prim", std::move(prim));
  }
};
ZENDEFNODE(PrimTwoCopyAttr,
           { /* inputs: */ {
                "primTo",
                "primFrom",
                {"string", "NameTo", "s"},
                {"string", "NameFrom", "t"},
                {"enum verts tris loops polys lines", "ScopeTo", "verts"},
                {"enum verts tris loops polys lines", "ScopeFrom", "verts"},
                {"enum int float vec2f vec2i vec3f vec3i vec4f vec4i", "type", "vec3f"},
            }, /* outputs: */ {
                "prim",
            }, /* params: */ {
            }, /* category: */ {
                "erode",
            }});
///////////////////////////////////////////////////////////////////////////////
// 2022.07.22 BVH
///////////////////////////////////////////////////////////////////////////////
struct BVHNearestPos : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto primNei = get_input<PrimitiveObject>("primNei");

        auto bvh_id = prim->attr<float>(get_input2<std::string>("bvhIdTag"));
        auto bvh_ws = prim->attr<zeno::vec3f>(get_input2<std::string>("bvhWeightTag"));
        auto &bvh_pos = prim->add_attr<zeno::vec3f>(get_input2<std::string>("bvhPosTag"));

#pragma omp parallel for
        for (int i = 0; i < prim->size(); i++) {
            vec3i vertsIdx = primNei->tris[(int)bvh_id[i]];
            int v0 = vertsIdx[0], v1 = vertsIdx[1], v2 = vertsIdx[2];
            auto p0 = primNei->verts[v0];
            auto p1 = primNei->verts[v1];
            auto p2 = primNei->verts[v2];
            bvh_pos[i] = bvh_ws[i][0] * p0 + bvh_ws[i][1] * p1 + bvh_ws[i][2] * p2;
        }

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(BVHNearestPos,
           { /* inputs: */ {
                   "prim",
                   "primNei",
                   {"string", "bvhIdTag", "bvh_id"},
                   {"string", "bvhWeightTag", "bvh_ws"},
                   {"string", "bvhPosTag", "bvh_pos"},
               }, /* outputs: */ {
                   "prim"
               }, /* params: */ {
               }, /* category: */ {
                   "deprecated"
               }});

struct BVHNearestAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto primNei = get_input<PrimitiveObject>("primNei");
        auto bvhIdTag = get_input2<std::string>("bvhIdTag");
        auto bvhAttributesType = get_input2<std::string>("bvhAttributesType");
        auto targetType = get_input2<std::string>("targetPrimType");
        auto attr_tag = get_input2<std::string>("bvhAttrTag");

        if (!primNei->verts.has_attr(attr_tag))
        {
            zeno::log_error("primNei has no such Data named '{}'.", attr_tag);
        }

        std::visit([&](auto attrty) {
            using T = decltype(attrty);
        auto& inAttr = primNei->verts.attr<T>(attr_tag);
        if (!prim->verts.has_attr(attr_tag))
        {
            prim->add_attr<T>(attr_tag);
        }
        auto& outAttr = prim->verts.attr<T>(attr_tag);


        if(targetType == "tris"){
        auto bvhWeightTag = get_input2<std::string>("bvhWeightTag");
        auto& bvh_ws = prim->verts.attr<zeno::vec3f>(bvhWeightTag);
        auto& bvh_id = prim->verts.attr<float>(bvhIdTag);
        #pragma omp parallel for
        for (int i = 0; i < prim->size(); i++){
            vec3i vertsIdx = primNei->tris[(int)bvh_id[i]];
            int id0 = vertsIdx[0], id1 = vertsIdx[1], id2 = vertsIdx[2];
            auto attr0 = inAttr[id0];
            auto attr1 = inAttr[id1];
            auto attr2 = inAttr[id2];
            outAttr[i] = bvh_ws[i][0] * attr0 + bvh_ws[i][1] * attr1 + bvh_ws[i][2] * attr2;
            }
        }
        else if(targetType == "points"){
             auto& bvh_id = prim->verts.attr<int>(bvhIdTag);//int type for querynearestpoints node
            #pragma omp parallel for
            for (int i = 0; i < prim->size(); i++){
                int id = bvh_id[i];
                outAttr[i] = inAttr[id];
            }
        }

        }, enum_variant<std::variant<float, vec3f>>(array_index({"float", "vec3f"}, bvhAttributesType)));


        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(BVHNearestAttr,
           { /* inputs: */ {
                   "prim",
                   "primNei",
                   {"string", "bvhIdTag", "bvh_id"},
                   {"string", "bvhWeightTag", "bvh_ws"},
                   {"string", "bvhAttrTag", "bvh_attr"},
                   {"enum float vec3f", "bvhAttributesType", "float"},
                   {"enum tris points", "targetPrimType", "tris"},
               }, /* outputs: */ {
                   "prim"
               }, /* params: */ {
               }, /* category: */ {
                   "primitive"
               }});


///////////////////////////////////////////////////////////////////////////////
// 2023 英启恒 Terrain Shape
///////////////////////////////////////////////////////////////////////////////
struct HeightStarPattern : zeno::INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto rotate = get_input2<float>("rotate");
        auto anglerandom = get_input2<float>("anglerandom");
        auto shapesize = get_input2<float>("shapesize");
        auto posjitter = get_input2<float>("posjitter");
        auto sharpness = get_input2<float>("sharpness");
        auto starness = get_input2<float>("starness");
        auto sides = get_input2<int>("sides");
        prim->verts.add_attr<float>("result");

        std::uniform_real_distribution<float> dist(0, 1);

#pragma omp parallel for
        for (int i = 0; i < prim->verts.size(); i++) {
            auto coord = prim->verts.attr<zeno::vec3f>("res")[i];
            vec2f coord2d = vec2f(coord[0], coord[1]);
            vec2f cellcenter = vec2f(floor(coord2d[0]), floor(coord2d[1]));
            float result = 0;

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    vec2f lcl_cellcenter = cellcenter;
                    lcl_cellcenter[0] += dx;
                    lcl_cellcenter[1] += dy;
                    std::minstd_rand e((lcl_cellcenter[0]*100+dx)*(lcl_cellcenter[1]*100+dy));
                    vec2f center = coord2d - lcl_cellcenter;

                    float lcl_shapeangle = rotate + ((dist(e) - 0.5) * anglerandom) * M_PI * 2;
                    center[0] += (dist(e) - 0.5) * posjitter;
                    center[1] += (dist(e) - 0.5) * posjitter;

                    center /= shapesize;

                    //Evaluate star function
                    float distance = lengthSquared(center);
                    float curangle = atan2(center[0], center[1]) + lcl_shapeangle;
                    float star = pow(abs(cos(curangle * sides / 2)), sharpness);
                    star *= starness;

                    int res = distance + star < 1 ? 1 : 0;
                    result = max(result, res);
                    if (result == 1) {
                        break;
                    }
                }
            }
            prim->verts.attr<float>("result")[i] = result;
        }
        prim->verts.erase_attr("res");
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(HeightStarPattern,
           {/* inputs: */ {
                   {"PrimitiveObject", "prim"},
                   {"float", "rotate", "0"},
                   {"float", "anglerandom", "0"},
                   {"float", "shapesize", "0.5"},
                   {"float", "posjitter", "0"},
                   {"float", "sharpness", "0.5"},
                   {"float", "starness", "0.5"},
                   {"int", "sides", "5"},
               }, /* outputs: */ {
                   {"PrimitiveObject", "prim"},
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               }});


///////////////////////////////////////////////////////////////////////////////
// 2023.01.05 节点图中自撸循环，遍历 prim 设置 /获取属性
///////////////////////////////////////////////////////////////////////////////
// Set Attr
struct PrimSetAttr : INode {
    void apply() override {

        auto prim = get_input<PrimitiveObject>("prim");
        auto value = get_input<NumericObject>("value");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");
        auto index = get_input<NumericObject>("index")->get<int>();
        auto method = get_input<StringObject>("method")->get();

        std::visit(
            [&](auto ty) {
              using T = decltype(ty);

              auto val = value->get<T>();
              if (method == "vert") {
                  auto &attr_arr = prim->add_attr<T>(name);
                  if (index < attr_arr.size()) {
                      attr_arr[index] = val;
                  }
              } else if (method == "tri") {
                  auto &attr_arr = prim->tris.add_attr<T>(name);
                  if (index < attr_arr.size()) {
                      attr_arr[index] = val;
                  }
              } else if (method == "line") {
                  auto &attr_arr = prim->lines.add_attr<T>(name);
                  if (index < attr_arr.size()) {
                      attr_arr[index] = val;
                  }
              } else if (method == "loop") {
                  auto &attr_arr = prim->loops.add_attr<T>(name);
                  if (index < attr_arr.size()) {
                      attr_arr[index] = val;
                  }
              } else if (method == "poly") {
                  auto &attr_arr = prim->polys.add_attr<T>(name);
                  if (index < attr_arr.size()) {
                      attr_arr[index] = val;
                  }
              } else {
                  throw Exception("bad type: " + method);
              }
            },
            enum_variant<std::variant<float, vec2f, vec3f, vec4f, int, vec2i, vec3i, vec4i>>(
                array_index({"float", "vec2f", "vec3f", "vec4f", "int", "vec2i", "vec3i", "vec4i"}, type)));

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimSetAttr,
           { /* inputs: */ {
                   "prim",
                   {"int", "value", "0"},
                   {"string", "name", "index"},
                   {"enum float vec2f vec3f vec4f int vec2i vec3i vec4i", "type", "int"},
                   {"enum vert tri line loop poly", "method", "tri"},
                   {"int", "index", "0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               }});

// Get Attr
struct PrimGetAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");
        auto index = get_input<NumericObject>("index")->get<int>();
        auto method = get_input<StringObject>("method")->get();

        auto value = std::make_shared<NumericObject>();

        std::visit(
            [&](auto ty) {
              using T = decltype(ty);

              if (method == "vert") {
                  auto &attr_arr = prim->attr<T>(name);
                  if (index < attr_arr.size()) {
                      value->set<T>(attr_arr[index]);
                  }
              } else if (method == "tri") {
                  auto &attr_arr = prim->tris.attr<T>(name);
                  if (index < attr_arr.size()) {
                      value->set<T>(attr_arr[index]);
                  }
              } else if (method == "line") {
                  auto &attr_arr = prim->lines.attr<T>(name);
                  if (index < attr_arr.size()) {
                      value->set<T>(attr_arr[index]);
                  }
              } else if (method == "loop") {
                  auto &attr_arr = prim->loops.attr<T>(name);
                  if (index < attr_arr.size()) {
                      value->set<T>(attr_arr[index]);
                  }
              } else if (method == "poly") {
                  auto &attr_arr = prim->polys.attr<T>(name);
                  if (index < attr_arr.size()) {
                      value->set<T>(attr_arr[index]);
                  }
              } else {
                  throw Exception("bad type: " + method);
              }
            },
            enum_variant<std::variant<float, vec2f, vec3f, vec4f, int, vec2i, vec3i, vec4i>>(
                array_index({"float", "vec2f", "vec3f", "vec4f", "int", "vec2i", "vec3i", "vec4i"}, type)));

        set_output("value", std::move(value));
    }
};
ZENDEFNODE(PrimGetAttr,
           { /* inputs: */ {
                   "prim",
                   {"string", "name", "index"},
                   {"enum float vec2f vec3f vec4f int vec2i vec3i vec4i", "type", "int"},
                   {"enum vert tri line loop poly", "method", "tri"},
                   {"int", "index", "0"},
               }, /* outputs: */ {
                   "value",
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               }});

// 删除多个属性
struct PrimitiveDelAttrs : zeno::INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto invert = get_input2<bool>("invert");
        auto nameString = get_input2<std::string>("names");
        auto scope = get_input2<std::string>("scope");
        bool flag_verts = false;
        bool flag_tris = false;
        bool flag_lines = false;
        bool flag_loops = false;
        bool flag_polys = false;
        if (scope == "vert") {
            flag_verts = true;
        }
        else if (scope == "tri") {
            flag_tris = true;
        }
        else if (scope == "loop") {
            flag_loops = true;
        }
        else if (scope == "poly") {
            flag_polys = true;
        }
        else if (scope == "line") {
            flag_lines = true;
        }
        else {
            flag_verts = true;
            flag_tris = true;
            flag_lines = true;
            flag_loops = true;
            flag_polys = true;
        }

        std::vector<std::string> names;
        std::istringstream ss(nameString);
        std::string name;
        while(ss >> name) {
            names.push_back(name);
        }

        if (!invert) {
            for(std::string attr : names) {
                if (flag_verts) prim->verts.attrs.erase(attr);
                if (flag_tris)  prim->tris.attrs.erase(attr);
                if (flag_lines) prim->lines.attrs.erase(attr);
                if (flag_loops) prim->loops.attrs.erase(attr);
                if (flag_polys) prim->polys.attrs.erase(attr);
            }
        } else {
            std::vector<std::string> myKeys = prim->verts.attr_keys();

            auto reserve_attr = std::remove_if(myKeys.begin(), myKeys.end(), [&](const auto &item) {
              return std::find(names.begin(), names.end(), item) != names.end();
            });
            myKeys.erase(reserve_attr, myKeys.end());

            for(std::string attr : myKeys){
                if (flag_verts) prim->verts.attrs.erase(attr);
                if (flag_tris)  prim->tris.attrs.erase(attr);
                if (flag_lines) prim->lines.attrs.erase(attr);
                if (flag_loops) prim->loops.attrs.erase(attr);
                if (flag_polys) prim->polys.attrs.erase(attr);
            }
        }

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(PrimitiveDelAttrs,
           { /* inputs: */ {
                   "prim",
                   {"bool", "invert", "0"},
                   {"string", "names", "name_1 name_2"},
                   {"enum vert tri loop poly line all", "scope", "all"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               } });


///////////////////////////////////////////////////////////////////////////////
// 2023.03.28 Quat and Rotation
///////////////////////////////////////////////////////////////////////////////
// 起始朝向 and 结束朝向 => 四元数
struct QuatRotBetweenVectors : INode {
    void apply() override
    {
        auto start = normalize(get_input<NumericObject>("start")->get<zeno::vec3f>());
        auto dest = normalize(get_input<NumericObject>("dest")->get<zeno::vec3f>());

        glm::vec3 gl_start(start[0], start[1], start[2]);
        glm::vec3 gl_dest(dest[0], dest[1], dest[2]);
        glm::quat gl_quat = glm::rotation(gl_start, gl_dest);

        vec4f rot(gl_quat.x, gl_quat.y, gl_quat.z, gl_quat.w);
        auto rotation = std::make_shared<NumericObject>();
        rotation->set<vec4f>(rot);
        set_output("quat", rotation);
    }
};
ZENDEFNODE(QuatRotBetweenVectors,
           {  /* inputs: */ {
                   {"vec3f", "start", "1,0,0"},
                   {"vec3f", "dest", "1,0,0"},
               }, /* outputs: */ {
                   {"vec4f", "quat", "0,0,0,1"},
               }, /* params: */ {
               }, /* category: */ {
                   "quat",
               }});

// 矢量 * 四元数 => 矢量
struct QuatRotate : INode {
    void apply() override {
        auto quat = get_input<NumericObject>("quat")->get<zeno::vec4f>();
        auto vec3 = get_input<NumericObject>("vec3")->get<zeno::vec3f>();

        glm::vec3 gl_vec3(vec3[0], vec3[1], vec3[2]);
        glm::quat gl_quat(quat[3], quat[0], quat[1], quat[2]);
        glm::vec3 gl_vec3_out = glm::rotate(gl_quat, gl_vec3);

        vec3f vec3_o(gl_vec3_out.x, gl_vec3_out.y, gl_vec3_out.z);
        auto vec3_out = std::make_shared<NumericObject>();
        vec3_out->set<vec3f>(vec3_o);
        set_output("vec3", vec3_out);
    }
};
ZENDEFNODE(QuatRotate,
           {/* inputs: */ {
                   {"vec4f", "quat", "0,0,0,1"},
                   {"vec3f", "vec3", "1,0,0"},
               }, /* outputs: */ {
                   {"vec3f", "vec3", "1,0,0"},
               }, /* params: */ {
               }, /* category: */ {
                   "quat",
               }});

// 旋转轴 + 旋转角度 => 四元数
struct QuatAngleAxis : INode {
    void apply() override
    {
        auto angle = get_input<NumericObject>("angle(D)")->get<float>();
        auto axis = normalize(get_input<NumericObject>("axis")->get<zeno::vec3f>());

        float gl_angle = glm::radians(angle);
        glm::vec3 gl_axis(axis[0], axis[1], axis[2]);
        glm::quat gl_quat = glm::angleAxis(gl_angle, gl_axis);

        vec4f rot(gl_quat.x, gl_quat.y, gl_quat.z, gl_quat.w);
        auto rotation = std::make_shared<NumericObject>();
        rotation->set<vec4f>(rot);

        set_output("quat", rotation);
    }
};
ZENDEFNODE(QuatAngleAxis,
           {  /* inputs: */ {
                   {"float", "angle(D)", "0"},
                   {"vec3f", "axis", "1,0,0"},
               }, /* outputs: */ {
                   {"vec4f", "quat", "0,0,0,1"},
               }, /* params: */ {
               }, /* category: */ {
                   "quat",
               }});

// 四元数 -> 旋转角度
struct QuatGetAngle : INode {
    void apply() override {
        auto quat = get_input<NumericObject>("quat")->get<zeno::vec4f>();

        glm::quat gl_quat(quat[3], quat[0], quat[1], quat[2]);
        float gl_angle = glm::degrees(glm::angle(gl_quat));

        auto angle = std::make_shared<NumericObject>();
        angle->set<float>(gl_angle);

        set_output("angle(D)", angle);
    }
};
ZENDEFNODE(QuatGetAngle,
           {/* inputs: */ {
                   {"vec4f", "quat", "0,0,0,1"},
               }, /* outputs: */ {
                   {"float", "angle(D)", "0"},
               }, /* params: */ {
               }, /* category: */ {
                   "quat",
               }});

// 四元数 -> 旋转轴
struct QuatGetAxis : INode {
    void apply() override {
        auto quat = get_input<NumericObject>("quat")->get<zeno::vec4f>();

        glm::quat gl_quat(quat[3], quat[0], quat[1], quat[2]);
        glm::vec3 gl_axis = glm::axis(gl_quat);

        vec3f axis_o(gl_axis.x, gl_axis.y, gl_axis.z);
        auto axis = std::make_shared<NumericObject>();
        axis->set<vec3f>(axis_o);
        set_output("axis", axis);
    }
};
ZENDEFNODE(QuatGetAxis,
           { /* inputs: */ {
                   {"vec4f", "quat", "0,0,0,1"},
               }, /* outputs: */ {
                   {"vec3f", "axis", "1,0,0"},
               }, /* params: */ {
               }, /* category: */ {
                   "quat",
               }});

// 矩阵转置
struct MatTranspose : INode {
    void apply() override
    {
        glm::mat mat = std::get<glm::mat4>(get_input<MatrixObject>("mat")->m);
        glm::mat transposeMat = glm::transpose(mat);
        auto oMat = std::make_shared<MatrixObject>();
        oMat->m = transposeMat;
        set_output("transposeMat", oMat);
    }
};
ZENDEFNODE(MatTranspose,
           { /* inputs: */ {
                   "mat",
               }, /* outputs: */ {
                   "transposeMat",
               }, /* params: */ {
               }, /* category: */ {
                   "math",
               }});


///////////////////////////////////////////////////////////////////////////////
// 2023.03.31 primCurve
///////////////////////////////////////////////////////////////////////////////
// dir为指向末端的切线方向
struct PrimCurveDir : INode {
    void apply() override
    {
        auto prim = get_input<PrimitiveObject>("prim_curve");
        auto dirName = get_input2<std::string>("dirName");
        auto &directions = prim->add_attr<zeno::vec3f>(dirName);
        size_t n = prim->size();
#pragma omp parallel for
        for (intptr_t i = 1; i < n - 1; i++) {
            auto lastpos = prim->verts[i - 1];
            auto currpos = prim->verts[i];
            auto nextpos = prim->verts[i + 1];
            auto direction = normalize(nextpos - lastpos);
            directions[i] = direction;
        }
        directions[0] = normalize(prim->verts[1] - prim->verts[0]);
        directions[n - 1] = normalize(prim->verts[n - 1] - prim->verts[n - 2]);
        set_output("prim_curve", std::move(prim));
    }
};
ZENDEFNODE(PrimCurveDir,
           {  /* inputs: */ {
                   "prim_curve",
                   {"string", "dirName", "nrm"},
               }, /* outputs: */ {
                   "prim_curve",
               }, /* params: */ {
               }, /* category: */ {
                   "primCurve",
               }});

template<typename T>
static void smooth(const std::vector<int> &neighborIdxs,
                   const std::vector<T> &neighborVals,
                   const std::vector<float> &neighborEdgeWeights,
                   const int useEdgeWeight,
                   const T inData,
                   const float weight,
                   const float w,
                   T &outData)
{
    T ndata { 0 };
    int count = 0;

    if (useEdgeWeight) {
#pragma omp parallel for
        for(int i = 0; i < neighborIdxs.size(); i++) {
            if (neighborIdxs[i] != -1) {
                ndata += neighborVals[i] * neighborEdgeWeights[i];
            }
        }
        outData = inData + weight * w * (ndata - inData);
    } else {
#pragma omp parallel for
        for(int i = 0; i < neighborIdxs.size(); i++) {
            if (neighborIdxs[i] != -1) {
                ndata += neighborVals[i];
                count++;
            }
        }
        float denom = 1.0f / (float)count;
        outData = inData + weight * w * (ndata * denom - inData);
    }
}

// 平滑属性，非常有用的功能
struct PrimAttribBlur : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto prim_type = get_input2<std::string>("primType");
        auto attr_name = get_input2<std::string>("attributes");
        auto attr_type = get_input2<std::string>("attributesType");

        auto useEdgeLength = get_input<NumericObject>("useEdgeLengthWeight")->get<bool>();

        auto iterations = get_input<NumericObject>("blurringIterations")->get<int>();

        auto mode = get_input2<std::string>("mode");
        auto mu = get_input<NumericObject>("stepSize")->get<float>();
        auto lambda = mu;
        if (mode == "VolumePreserving") {
            auto passband = get_input<NumericObject>("cutoffFrequency")->get<float>();
            if (passband < 1e-5) {
                float sqrt2_5 = 0.6324555320336758664;
                lambda = sqrt2_5;
                mu = -sqrt2_5;
            } else {
                // See:
                //   Gabriel Taubin. "A signal processing approach to fair surface design".
                //   SIGGRAPH 1995
                //
                // Let l be lambda, u be mu, and b be the passband frequency.
                // f(k) = (1-k*l)(1-k*u).  This function has maximum at 1/l + 1/u.
                // We want the maximum to occur at b, so we have the constraint
                //          1/l + 1/u = passband
                // so       u = l/(bl - 1)
                // We also want f(1) = -f(2).  This gives the constraint:
                //          u = (2-3l)/(3-5l)
                // Equating this equations gives:
                //          (3b-5)l^2 - 2bl + 2 = 0
                // Solve for l using the quadratic formula.  We want l>0.  Since b>0 and
                // 6b-10<0, subtract the square root of the discriminant to get a negative
                // numerator so the quotient is positive.
                auto discriminant = 4.0 * passband * passband - 24.0 * passband + 40.0;
                lambda = 2.0 * passband - sqrt(discriminant);
                lambda /= (6.0 * passband - 10.0);

                // Now solve for u.  Note that u is also the other root of the quadratic.
                mu = lambda / (passband * lambda - 1.0);
            }
        } else if (mode == "custom") {
            lambda = get_input<NumericObject>("oddStepSize")->get<float>();
            mu = get_input<NumericObject>("evenStepSize")->get<float>();
        }

        auto weightName = get_input2<std::string>("weightAttributes");
        if (!prim->verts.has_attr(weightName)) {
            auto &_weight = prim->verts.add_attr<float>(weightName);
            std::fill(_weight.begin(), _weight.end(), 1.0);
        }
        auto &weight = prim->verts.attr<float>(weightName);

        // 找临近点，假设最多 8 个临近点
        auto &neighbor_0 = prim->verts.add_attr<int>("_neighbor_0");
        auto &neighbor_1 = prim->verts.add_attr<int>("_neighbor_1");
        auto &neighbor_2 = prim->verts.add_attr<int>("_neighbor_2");
        auto &neighbor_3 = prim->verts.add_attr<int>("_neighbor_3");
        auto &neighbor_4 = prim->verts.add_attr<int>("_neighbor_4");
        auto &neighbor_5 = prim->verts.add_attr<int>("_neighbor_5");
        auto &neighbor_6 = prim->verts.add_attr<int>("_neighbor_6");
        auto &neighbor_7 = prim->verts.add_attr<int>("_neighbor_7");
        std::fill(neighbor_0.begin(), neighbor_0.end(), -1);
        std::fill(neighbor_1.begin(), neighbor_1.end(), -1);
        std::fill(neighbor_2.begin(), neighbor_2.end(), -1);
        std::fill(neighbor_3.begin(), neighbor_3.end(), -1);
        std::fill(neighbor_4.begin(), neighbor_4.end(), -1);
        std::fill(neighbor_5.begin(), neighbor_5.end(), -1);
        std::fill(neighbor_6.begin(), neighbor_6.end(), -1);
        std::fill(neighbor_7.begin(), neighbor_7.end(), -1);
        auto &edgeweight_0 = prim->verts.add_attr<float>("_edgeweight_0");
        auto &edgeweight_1 = prim->verts.add_attr<float>("_edgeweight_1");
        auto &edgeweight_2 = prim->verts.add_attr<float>("_edgeweight_2");
        auto &edgeweight_3 = prim->verts.add_attr<float>("_edgeweight_3");
        auto &edgeweight_4 = prim->verts.add_attr<float>("_edgeweight_4");
        auto &edgeweight_5 = prim->verts.add_attr<float>("_edgeweight_5");
        auto &edgeweight_6 = prim->verts.add_attr<float>("_edgeweight_6");
        auto &edgeweight_7 = prim->verts.add_attr<float>("_edgeweight_7");
        std::fill(edgeweight_0.begin(), edgeweight_0.end(), 0);
        std::fill(edgeweight_1.begin(), edgeweight_1.end(), 0);
        std::fill(edgeweight_2.begin(), edgeweight_2.end(), 0);
        std::fill(edgeweight_3.begin(), edgeweight_3.end(), 0);
        std::fill(edgeweight_4.begin(), edgeweight_4.end(), 0);
        std::fill(edgeweight_5.begin(), edgeweight_5.end(), 0);
        std::fill(edgeweight_6.begin(), edgeweight_6.end(), 0);
        std::fill(edgeweight_7.begin(), edgeweight_7.end(), 0);

        //========================================
//        LARGE_INTEGER t1_0,t2_0,tc_0;
//        LARGE_INTEGER t1_1,t2_1,tc_1;
//        LARGE_INTEGER t1_2,t2_2,tc_2;
        //========================================

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//        QueryPerformanceFrequency(&tc_0);
//        QueryPerformanceCounter(&t1_0);

#pragma omp parallel for
        for (auto point_idx = 0; point_idx < prim->verts.size(); point_idx++) {   // 遍历所有点，找它的邻居
            std::map<std::string, int> neighborVertID;
            std::map<std::string, float> neighborEdgeLength;
            for(int i = 0; i < 8; i++) {
                neighborVertID["neighbor_" + std::to_string(i)] = -1;
                neighborEdgeLength["edgeweight_" + std::to_string(i)] = 0;
            }

            int find_neighbor_count = 0;
            float edgeLengthSum = 0;
            volatile bool flag = false;

            if (prim_type == "line") {
#pragma omp parallel for shared(flag)
                for (auto line_idx = 0; line_idx < prim->lines.size(); line_idx++) {
                    if(flag) continue;
                    if (prim->lines[line_idx][0] == point_idx) {
                        neighborVertID["neighbor_" + std::to_string(find_neighbor_count)] = prim->lines[line_idx][1];
                        if (useEdgeLength) {
                            float edgeLength = length(prim->verts[prim->lines[line_idx][1]] - prim->verts[point_idx]);
                            neighborEdgeLength["edgeweight_" + std::to_string(find_neighbor_count)] = edgeLength;
                            edgeLengthSum += edgeLength;
                        }
                        find_neighbor_count ++;
                    } else if (prim->lines[line_idx][1] == point_idx) {
                        neighborVertID["neighbor_" + std::to_string(find_neighbor_count)] = prim->lines[line_idx][0];
                        if (useEdgeLength) {
                            float edgeLength = length(prim->verts[prim->lines[line_idx][0]] - prim->verts[point_idx]);
                            neighborEdgeLength["edgeweight_" + std::to_string(find_neighbor_count)] = edgeLength;
                            edgeLengthSum += edgeLength;
                        }
                        find_neighbor_count++;
                    }
                    if (find_neighbor_count >= 7)
                        flag = true;
                }
            } else if (prim_type == "tri") {
                std::vector<int> pointNeighborSign(prim->verts.size());
                std::fill(pointNeighborSign.begin(), pointNeighborSign.end(), 0);

                //========================================
//                if (point_idx == 50000) {
//                    QueryPerformanceFrequency(&tc_1);
//                    QueryPerformanceCounter(&t1_1);
//                }
                //========================================

#pragma omp parallel for
                for (auto tri_idx = 0; tri_idx < prim->tris.size(); tri_idx++) {
                    auto const &ind = prim->tris[tri_idx];
                    if (ind[0] == point_idx) {
                        pointNeighborSign[ind[1]] = 1;
                        pointNeighborSign[ind[2]] = 1;
                    } else if (ind[1] == point_idx) {
                        pointNeighborSign[ind[0]] = 1;
                        pointNeighborSign[ind[2]] = 1;
                    } else if (ind[2] == point_idx) {
                        pointNeighborSign[ind[0]] = 1;
                        pointNeighborSign[ind[1]] = 1;
                    }
                }

#pragma omp parallel for shared(flag)
                for (int i = 0; i < prim->verts.size(); i++) {
                    if(flag) continue;
                    if (pointNeighborSign[i]) {
                        neighborVertID["neighbor_" + std::to_string(find_neighbor_count)] = i;
                        if (useEdgeLength) {
                            float edgeLength = length(prim->verts[i] - prim->verts[point_idx]);
                            neighborEdgeLength["edgeweight_" + std::to_string(find_neighbor_count)] = edgeLength;
                            edgeLengthSum += edgeLength;
                        }
                        find_neighbor_count ++;
                    }
                    if (find_neighbor_count >= 7)
                        flag = true;
                }

                //========================================
//                if (point_idx == 50000) {
//                    QueryPerformanceCounter(&t2_1);
//                    double time_1 = (double)(t2_1.QuadPart - t1_1.QuadPart)/(double)tc_1.QuadPart;
//                    printf("time_1 = %f s\n", time_1);
//                }
                //========================================
            }

            neighbor_0[point_idx] = neighborVertID["neighbor_0"];
            neighbor_1[point_idx] = neighborVertID["neighbor_1"];
            neighbor_2[point_idx] = neighborVertID["neighbor_2"];
            neighbor_3[point_idx] = neighborVertID["neighbor_3"];
            neighbor_4[point_idx] = neighborVertID["neighbor_4"];
            neighbor_5[point_idx] = neighborVertID["neighbor_5"];
            neighbor_6[point_idx] = neighborVertID["neighbor_6"];
            neighbor_7[point_idx] = neighborVertID["neighbor_7"];

            if (useEdgeLength) {
                float min_length = (edgeLengthSum / (float)(find_neighbor_count)) * 0.001f;
                float sum = 0;

#pragma omp parallel for
                for (int i = 0; i < find_neighbor_count; i++)
                {
                    float length = neighborEdgeLength["edgeweight_" + std::to_string(i)];

                    if ( length > min_length )
                        neighborEdgeLength["edgeweight_" + std::to_string(i)] = 1.0 / length;
                    else    // 基本重合的点，不考虑其影响，权重打到 0
                        neighborEdgeLength["edgeweight_" + std::to_string(i)] = 0;

                    sum += neighborEdgeLength["edgeweight_" + std::to_string(i)];   // 累计总权重
                }
                if ( sum > 0 )
                {
#pragma omp parallel for
                    for (int i = 0; i < find_neighbor_count; ++i)
                    {
                        neighborEdgeLength["edgeweight_" + std::to_string(i)] /= sum;   // 权重归一化
                    }
                }

                edgeweight_0[point_idx] = neighborEdgeLength["edgeweight_0"];
                edgeweight_1[point_idx] = neighborEdgeLength["edgeweight_1"];
                edgeweight_2[point_idx] = neighborEdgeLength["edgeweight_2"];
                edgeweight_3[point_idx] = neighborEdgeLength["edgeweight_3"];
                edgeweight_4[point_idx] = neighborEdgeLength["edgeweight_4"];
                edgeweight_5[point_idx] = neighborEdgeLength["edgeweight_5"];
                edgeweight_6[point_idx] = neighborEdgeLength["edgeweight_6"];
                edgeweight_7[point_idx] = neighborEdgeLength["edgeweight_7"];
            }

        }

//        QueryPerformanceCounter(&t2_0);
//        double time_0 = (double)(t2_0.QuadPart - t1_0.QuadPart)/(double)tc_0.QuadPart;
//        printf("time_0 = %f s\n", time_0);
//        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//        QueryPerformanceFrequency(&tc_2);
//        QueryPerformanceCounter(&t1_2);

        // 平滑属性计算
        std::visit(
            [&](auto ty) {
              using T = decltype(ty);

              auto &data = prim->verts.attr<T>(attr_name);
              auto &data_temp = prim->verts.add_attr<T>("_data_temp");
              std::fill(data_temp.begin(), data_temp.end(), T(0));

              for (int loop = 0; loop < iterations; loop++) {
#pragma omp parallel for
                  // data => data_temp
                  for (auto i = 0; i < prim->verts.size(); i++) {
                      std::vector<int> neighborIDs(8);
                      neighborIDs[0] = neighbor_0[i];
                      neighborIDs[1] = neighbor_1[i];
                      neighborIDs[2] = neighbor_2[i];
                      neighborIDs[3] = neighbor_3[i];
                      neighborIDs[4] = neighbor_4[i];
                      neighborIDs[5] = neighbor_5[i];
                      neighborIDs[6] = neighbor_6[i];
                      neighborIDs[7] = neighbor_7[i];
                      std::vector<T> neighborValues(8);
                      for (int i = 0; i < neighborIDs.size(); i++) {
                          if (neighborIDs[i] != -1)
                              neighborValues[i] = data[neighborIDs[i]];
                      }
                      std::vector<float> neighborEdgeWeights(8);
                      neighborEdgeWeights[0] = edgeweight_0[i];
                      neighborEdgeWeights[1] = edgeweight_1[i];
                      neighborEdgeWeights[2] = edgeweight_2[i];
                      neighborEdgeWeights[3] = edgeweight_3[i];
                      neighborEdgeWeights[4] = edgeweight_4[i];
                      neighborEdgeWeights[5] = edgeweight_5[i];
                      neighborEdgeWeights[6] = edgeweight_6[i];
                      neighborEdgeWeights[7] = edgeweight_7[i];
                      smooth(neighborIDs, neighborValues, neighborEdgeWeights, useEdgeLength, data[i], weight[i], lambda, data_temp[i]);
                  }
#pragma omp parallel for
                  // data_temp => data
                  for (auto i = 0; i < prim->verts.size(); i++) {
                      std::vector<int> neighborIDs(8);
                      neighborIDs[0] = neighbor_0[i];
                      neighborIDs[1] = neighbor_1[i];
                      neighborIDs[2] = neighbor_2[i];
                      neighborIDs[3] = neighbor_3[i];
                      neighborIDs[4] = neighbor_4[i];
                      neighborIDs[5] = neighbor_5[i];
                      neighborIDs[6] = neighbor_6[i];
                      neighborIDs[7] = neighbor_7[i];
                      std::vector<T> neighborValues(8);
                      for(int i = 0; i < neighborIDs.size(); i++)
                      {
                          if (neighborIDs[i] != -1)
                              neighborValues[i] = data_temp[neighborIDs[i]];
                      }
                      std::vector<float> neighborEdgeWeights(8);
                      neighborEdgeWeights[0] = edgeweight_0[i];
                      neighborEdgeWeights[1] = edgeweight_1[i];
                      neighborEdgeWeights[2] = edgeweight_2[i];
                      neighborEdgeWeights[3] = edgeweight_3[i];
                      neighborEdgeWeights[4] = edgeweight_4[i];
                      neighborEdgeWeights[5] = edgeweight_5[i];
                      neighborEdgeWeights[6] = edgeweight_6[i];
                      neighborEdgeWeights[7] = edgeweight_7[i];
                      smooth(neighborIDs, neighborValues, neighborEdgeWeights, useEdgeLength, data_temp[i], weight[i], mu, data[i]);
                  }
              }
              prim->verts.erase_attr("_data_temp");

            },
            enum_variant<std::variant<float, vec3f>>(
                array_index({"float", "vec3f"}, attr_type)));

//        QueryPerformanceCounter(&t2_2);
//        double time_2 = (double)(t2_2.QuadPart - t1_2.QuadPart)/(double)tc_2.QuadPart;
//        printf("time_2 = %f s\n", time_2);
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        prim->verts.erase_attr("_neighbor_0");
        prim->verts.erase_attr("_neighbor_1");
        prim->verts.erase_attr("_neighbor_2");
        prim->verts.erase_attr("_neighbor_3");
        prim->verts.erase_attr("_neighbor_4");
        prim->verts.erase_attr("_neighbor_5");
        prim->verts.erase_attr("_neighbor_6");
        prim->verts.erase_attr("_neighbor_7");
        prim->verts.erase_attr("_edgeweight_0");
        prim->verts.erase_attr("_edgeweight_1");
        prim->verts.erase_attr("_edgeweight_2");
        prim->verts.erase_attr("_edgeweight_3");
        prim->verts.erase_attr("_edgeweight_4");
        prim->verts.erase_attr("_edgeweight_5");
        prim->verts.erase_attr("_edgeweight_6");
        prim->verts.erase_attr("_edgeweight_7");

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimAttribBlur,
           {/* inputs: */ {
                   "prim",
                   {"enum line tri", "primType", "tri"},
//                   {"string", "group", "mask"},
                   {"string", "attributes", "ratio"},
                   {"enum float vec3f ", "attributesType", "float"},
                   {"bool", "useEdgeLengthWeight", "false"},
                   {"int", "blurringIterations", "0"},
                   {"enum laplacian VolumePreserving custom", "mode", "laplacian"},
                   {"float", "stepSize", "0.683"},
                   {"float", "cutoffFrequency", "0.1"},
                   {"float", "evenStepSize", "0.5"},
                   {"float", "oddStepSize", "0.5"},
                   {"string", "weightAttributes", "weight"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primCurve",
               }});

// 点连成线
struct PrimCurveFromVerts : INode {
    virtual void apply() override
    {
        auto prim = get_input<PrimitiveObject>("primVerts");
        size_t lines_count = prim->size() - 1;
        prim->lines.resize(lines_count);
        for (int i = 0; i < lines_count; i++) {
            prim->lines[i] = zeno::vec2i(i, i + 1);
        }

        set_output("primCurve", get_input("primVerts"));
    }
};
ZENDEFNODE(PrimCurveFromVerts,
           { /* inputs: */ {
                   "primVerts",
               }, /* outputs: */ {
                   "primCurve",
               }, /* params: */ {
               }, /* category: */ {
                   "primCurve",
               }});

/**
 * @brief _CreateBezierCurve 生成N阶贝塞尔曲线点
 * @param src 源贝塞尔控制点
 * @param dest 目的贝塞尔曲线点
 * @param precision 生成精度
 */
static void _CreateBezierCurve(const std::vector<zeno::vec3f> src, std::vector<zeno::vec3f> &dest, double precision) {
    int size = src.size();
    std::vector<double> coff(size, 0);

    std::vector<std::vector<int>> a(size, std::vector<int>(size));
    {
        for(int i=0;i<size;++i)
        {
            a[i][0]=1;
            a[i][i]=1;
        }
        for(int i=1;i<size;++i)
            for(int j=1;j<i;++j)
                a[i][j] = a[i-1][j-1] + a[i-1][j];
    }

    for (double t1 = 0; t1 < 1; t1 += precision) {
        double t2  = 1 - t1;
        int n = size - 1;

        coff[0] = pow(t2, n);
        coff[n] = pow(t1, n);
        for (int i = 1; i < size - 1; ++i) {
            coff[i] = pow(t2, n - i) * pow(t1, i) * a[n][i];
        }

        zeno::vec3f ret(0, 0, 0);
        for (int i = 0; i < size; ++i) {
            zeno::vec3f tmp(src[i][0] * coff[i], src[i][1] * coff[i], src[i][2] * coff[i]);
            ret[0] = ret[0] + tmp[0];
            ret[1] = ret[1] + tmp[1];
            ret[2] = ret[2] + tmp[2];
        }
        dest.push_back(ret);
    }
}

// 用指定的 verts 生成二阶贝塞尔曲线点
struct CreatePrimCurve : INode {
    virtual void apply() override {
        auto inPrim = get_input<zeno::PrimitiveObject>("inputPoints").get();
        auto outprim = std::make_shared<zeno::PrimitiveObject>();
        auto precision = get_input<zeno::NumericObject>("precision")->get<float>();

        auto tmpPos = inPrim->attr<zeno::vec3f>("pos");
        int subCurveCount = inPrim->verts.size() - 2;

        for(int i = 0; i < subCurveCount; i++)
        {
            std::vector<vec3f> subCurveInput(std::vector<vec3f>(3));
            if (i == 0) {
                subCurveInput[0] = tmpPos[i];
            } else {
                subCurveInput[0] = (tmpPos[i] + tmpPos[i + 1])/2;
            }

            subCurveInput[1] = tmpPos[i + 1];

            if (i == subCurveCount - 1) {
                subCurveInput[2] = tmpPos[i + 2];
            } else {
                subCurveInput[2] = (tmpPos[i + 1] + tmpPos[i + 2])/2;
            }

            std::vector<zeno::vec3f> outputPoints;
            _CreateBezierCurve(subCurveInput, outputPoints, precision);

            int oldVertCount = outprim->verts.size();
            outprim->verts.resize(oldVertCount + outputPoints.size());
            for (int i = 0; i < outputPoints.size(); i++) {
                outprim->verts[oldVertCount + i] = outputPoints[i];
            }

        }

        vec3f lastInPoint = inPrim->verts[inPrim->verts.size() - 1];
        vec3f lastOutPoint = outprim->verts[outprim->verts.size() - 1];
        if (length(lastInPoint - lastOutPoint) > 0.0001)
        {
            outprim->verts.resize( outprim->verts.size() + 1);
            outprim->verts[outprim->verts.size() - 1] = inPrim->verts[inPrim->verts.size() - 1];
        }

        size_t lines_count = outprim->size() - 1;
        outprim->lines.resize(lines_count);
        for (int i = 0; i < lines_count; i++) {
            outprim->lines[i] = zeno::vec2i(i, i + 1);
        }

        set_output("prim", std::move(outprim));
    }
};
ZENDEFNODE(CreatePrimCurve,
           {{
                   {"prim", "inputPoints"},
                   {"float", "precision", "0.01"},
               },
               {
                   "prim",
               },
               {
                   {"enum Bezier", "Type", "Bezier"},
               },
               {
                   "primCurve",
               }});

struct PrimHasAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto attrName = get_input2<std::string>("attrName");
        auto scope = get_input2<std::string>("scope");

        bool x = false;
        if (scope == "vert") {
            x = prim->verts.has_attr(attrName);
        }
        else if (scope == "tri") {
            x = prim->tris.has_attr(attrName);
        }
        else if (scope == "loop") {
            x = prim->loops.has_attr(attrName);
        }
        else if (scope == "poly") {
            x = prim->polys.has_attr(attrName);
        }
        else if (scope == "line") {
            x = prim->lines.has_attr(attrName);
        }

        auto hasAttr = std::make_shared<NumericObject>();
        hasAttr->set<bool>(x);
        set_output("hasAttr", hasAttr);
    }
};
ZENDEFNODE(PrimHasAttr,
           { /* inputs: */ {
                   "prim",
                   {"enum vert tri loop poly line", "scope", "vert"},
                   {"string", "attrName", "attr_x"},
               }, /* outputs: */ {
                   "hasAttr",
               }, /* params: */ {
               }, /* category: */ {
                   "erode",
               }});

} // namespace
} // namespace zeno