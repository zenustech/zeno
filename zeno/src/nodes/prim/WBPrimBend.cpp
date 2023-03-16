//
// Created by WangBo on 2022/7/5.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/log.h>
#include <glm/gtx/quaternion.hpp>
#include <random>
#include <numeric>
#include <zeno/utils/orthonormal.h>
#include <atomic>
#include <zeno/para/parallel_for.h> // enable by -DZENO_PARALLEL_STL:BOOL=ON
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/variantswitch.h>


namespace zeno
{
namespace
{
struct WBPrimBend : INode {
    void apply() override {
        auto limitDeformation = get_input<NumericObject>("Limit Deformation")->get<int>();
        auto symmetricDeformation = get_input<NumericObject>("Symmetric Deformation")->get<int>();
        auto angle = get_input<NumericObject>("Bend Angle (degree)")->get<float>();
        auto upVector = has_input("Up Vector") ? get_input<NumericObject>("Up Vector")->get<vec3f>() : vec3f(0, 1, 0);
        auto capOrigin = has_input("Capture Origin") ? get_input<NumericObject>("Capture Origin")->get<vec3f>() : vec3f(0, 0, 0);
        auto dirVector = has_input("Capture Direction") ? get_input<NumericObject>("Capture Direction")->get<vec3f>() : vec3f(0, 0, 1);
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
        } });

struct PrintPrimInfo : INode {
    void apply() override
    {
        auto prim = get_input<PrimitiveObject>("prim");

        if (get_param<bool>("printInfo"))
        {
            const std::vector<std::string>& myKeys = prim->attr_keys();
            printf("--------------------------------\n");
            printf("wb-Debug ==> vert has attr :\n");
            for (const std::string& i : myKeys)
            {
                printf("wb-Debug ==> vertAttr.%s\n", i.c_str());
            }
            printf("--------------------------------\n");
            printf("wb-Debug ==> vertsCount = %i\n", int(prim->verts.size()));
            printf("wb-Debug ==> pointsCount = %i\n", int(prim->points.size()));
            printf("wb-Debug ==> linesCount = %i\n", int(prim->lines.size()));
            printf("wb-Debug ==> trisCount = %i\n", int(prim->tris.size()));
            printf("wb-Debug ==> quadsCount = %i\n", int(prim->quads.size()));
            printf("--------------------------------\n");
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrintPrimInfo,
    { /* inputs: */ {
            "prim",
        }, /* outputs: */ {
            "prim",
        }, /* params: */ {
            {"int", "printInfo", "1"},
        }, /* category: */ {
            "primitive",
        } });

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

struct QuatRotBetweenVectors : INode {
    void apply() override
    {
        auto start = get_input<NumericObject>("start")->get<vec3f>();
        auto dest = get_input<NumericObject>("dest")->get<vec3f>();

        glm::vec3 gl_start(start[0], start[1], start[2]);
        glm::vec3 gl_dest(dest[0], dest[1], dest[2]);

        vec4f rot;
        glm::quat quat = glm::rotation(gl_start, gl_dest);
        rot[0] = quat.x;
        rot[1] = quat.y;
        rot[2] = quat.z;
        rot[3] = quat.w;

        auto rotation = std::make_shared<NumericObject>();
        rotation->set<vec4f>(rot);
        set_output("quatRotation", rotation);
    }
};
ZENDEFNODE(QuatRotBetweenVectors,
    {  /* inputs: */ {
            {"vec3f", "start", "1,0,0"},
            {"vec3f", "dest", "1,0,0"},
        }, /* outputs: */ {
            {"vec4f", "quatRotation", "0,0,0,1"},
        }, /* params: */ {
        }, /* category: */ {
            "math",
        } });

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
        } });

struct ParameterizeLine : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if(! prim->lines.has_attr("parameterization")) {
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
        if(! prim->lines.has_attr("parameterization")) {
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
            for (size_t i=0; i<prim->lines.size();i++) {
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
            for(size_t i=0; i<retprim->size();i++) {
                t_arr[i] = sampleBy[i];
            }
        } else {
            retprim->resize(segments + size_t(1));
            retprim->lines.resize(segments);
            retprim->add_attr<float>("t");
            auto &t_arr = retprim->attr<float>("t");
#pragma omp parallel for
            for (size_t i=0; i<retprim->size(); i++) {
                t_arr[i] = (float)i / float(segments);
            }
#pragma omp parallel for
            for (size_t i=0; i<segments; i++) {
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
        for(size_t i=0; i<retprim->size();i++) {
            float insertU = retprim->attr<float>("t")[i];
            auto it = std::lower_bound(linesLen.begin(), linesLen.end(), insertU);
            size_t index = it - linesLen.begin();
            index = std::min(index, prim->lines.size() - 1);
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
        prim->attr<vec3f>("pos").insert(prim->verts.begin() + int(index) + 1, { p[0], p[1], p[2] });

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

struct VisVec3Attribute : INode {
    void apply() override
    {
        auto color = get_input<NumericObject>("color")->get<vec3f>();
        auto useNormalize = get_input<NumericObject>("normalize")->get<int>();
        auto lengthScale = get_input<NumericObject>("lengthScale")->get<float>();
        auto name = get_input2<std::string>("name");

        auto prim = get_input<PrimitiveObject>("prim");
        auto& attr = prim->verts.attr<vec3f>(name);
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
        auto& visColor = primVis->verts.add_attr<vec3f>("clr");
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
        } });

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
        auto primDataVertsCount = primData->verts.size();;

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
        } });

struct PrimCopyFloatAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto sourceName = get_input<StringObject>("sourceName")->get();
        if (!prim->verts.has_attr(sourceName))
        {
            zeno::log_error("no such attr named '{}'.", sourceName);
        }
        auto& sourceAttr = prim->verts.attr<float>(sourceName); // 源属性

        auto targetName = get_input<StringObject>("targetName")->get();
        if (!prim->verts.has_attr(targetName))
        {
            prim->verts.add_attr<float>(targetName);
        }
        auto& targetAttr = prim->verts.attr<float>(targetName); // 目标属性

#pragma omp parallel for
        for (intptr_t i = 0; i < prim->verts.size(); i++)
        {
            targetAttr[i] = sourceAttr[i];
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimCopyFloatAttr,
    { /* inputs: */ {
            "prim",
            {"string", "sourceName", "s"},
            {"string", "targetName", "t"},
        }, /* outputs: */ {
            "prim",
        }, /* params: */ {
        }, /* category: */ {
            "primitive",
        } });


///////////////////////////////////////////////////////////////////////////////
// 2022.07.22 BVH
///////////////////////////////////////////////////////////////////////////////
struct BVHNearestPos : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto primNei = get_input<PrimitiveObject>("primNei");

        auto bvh_id = prim->attr<float>(get_input2<std::string>("bvhIdTag"));
        auto bvh_ws = prim->attr<vec3f>(get_input2<std::string>("bvhWeightTag"));
        auto &bvh_pos = prim->add_attr<vec3f>(get_input2<std::string>("bvhPosTag"));

#pragma omp parallel for
        for (int i = 0; i < prim->size(); i++) {
            vec3i vertsIdx = primNei->tris[(int)bvh_id[i]];
            int v0 = vertsIdx[0], v1 = vertsIdx[1], v2 = vertsIdx[2];
            auto p0 = primNei->verts[v0];
            auto p1 = primNei->verts[v1];
            auto p2 = primNei->verts[v2];
            bvh_pos[i] = bvh_ws[i][0] * p0 + bvh_ws[i][1] * p1 + bvh_ws[i][2] * p2;
        };

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
                   "primitive"
               } });

struct BVHNearestAttr : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto primNei = get_input<PrimitiveObject>("primNei");

        auto bvhIdTag = get_input<StringObject>("bvhIdTag")->get();
        auto& bvh_id = prim->verts.attr<float>(bvhIdTag);
        auto bvhWeightTag = get_input<StringObject>("bvhWeightTag")->get();
        auto& bvh_ws = prim->verts.attr<vec3f>(bvhWeightTag);

        auto attr_tag = get_input<StringObject>("bvhAttrTag")->get();
        if (!primNei->verts.has_attr(attr_tag))
        {
            zeno::log_error("primNei has no such Data named '{}'.", attr_tag);
        }
        auto& inAttr = primNei->verts.attr<float>(attr_tag);
        if (!prim->verts.has_attr(attr_tag))
        {
            prim->add_attr<float>(attr_tag);
        }
        auto& outAttr = prim->verts.attr<float>(attr_tag);

#pragma omp parallel for
        for (int i = 0; i < prim->size(); i++)
        {
            vec3i vertsIdx = primNei->tris[(int)bvh_id[i]];
            int id0 = vertsIdx[0], id1 = vertsIdx[1], id2 = vertsIdx[2];
            auto attr0 = inAttr[id0];
            auto attr1 = inAttr[id1];
            auto attr2 = inAttr[id2];

            outAttr[i] = bvh_ws[i][0] * attr0 + bvh_ws[i][1] * attr1 + bvh_ws[i][2] * attr2;
        };

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
    }, /* outputs: */ {
        "prim"
    }, /* params: */ {
    }, /* category: */ {
        "erode"
    } });



// WXL  =============================================================

#include <zeno/para/parallel_for.h> // enable by -DZENO_PARALLEL_STL:BOOL=ON
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/variantswitch.h>

#if defined(_OPENMP)
#define WXL 1
#else
#define WXL 0
#endif
#if WXL
#if defined(_OPENMP)
#include <omp.h>
#endif
#endif

template <class Cond>
static float tri_intersect(Cond cond, vec3f const& ro, vec3f const& rd, vec3f const& v0, vec3f const& v1, vec3f const& v2)
{
    const float eps = 1e-6f;
    vec3f u = v1 - v0;
    vec3f v = v2 - v0;
    vec3f n = cross(u, v);
    float b = dot(n, rd);
    if (std::abs(b) > eps) {
        float a = dot(n, v0 - ro);
        float r = a / b;
        if (cond(r)) {
            vec3f ip = ro + r * rd;
            float uu = dot(u, u);
            float uv = dot(u, v);
            float vv = dot(v, v);
            vec3f w = ip - v0;
            float wu = dot(w, u);
            float wv = dot(w, v);
            float d = uv * uv - uu * vv;
            float s = uv * wv - vv * wu;
            float t = uv * wu - uu * wv;
            d = 1.0f / d;
            s *= d;
            t *= d;
            if (-eps <= s && s <= 1 + eps && -eps <= t && s + t <= 1 + eps * 2)
                return r;
        }
    }
    return std::numeric_limits<float>::infinity();
}

/// ref: An Efficient and Robust Ray-Box Intersection Algorithm, 2005
static bool ray_box_intersect(vec3f const& ro, vec3f const& rd, std::pair<vec3f, vec3f> const& box) {
    vec3f invd{ 1 / rd[0], 1 / rd[1], 1 / rd[2] };
    int sign[3] = { invd[0] < 0, invd[1] < 0, invd[2] < 0 };
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = ((sign[0] ? box.second : box.first)[0] - ro[0]) * invd[0];
    tmax = ((sign[0] ? box.first : box.second)[0] - ro[0]) * invd[0];
    tymin = ((sign[1] ? box.second : box.first)[1] - ro[1]) * invd[1];
    tymax = ((sign[1] ? box.first : box.second)[1] - ro[1]) * invd[1];
    if (tmin > tymax || tymin > tmax)
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    tzmin = ((sign[2] ? box.second : box.first)[2] - ro[2]) * invd[2];
    tzmax = ((sign[2] ? box.first : box.second)[2] - ro[2]) * invd[2];
    if (tmin > tzmax || tzmin > tmax)
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    return tmax >= 0.f;
}

struct BVH { // TODO: WXL please complete this to accel up
    PrimitiveObject const* prim{};
#if WXL
    using TV = vec3f;
    using Box = std::pair<TV, TV>;
    using Ti = int;
    static constexpr Ti threshold = 128;
    using Tu = std::make_unsigned_t<Ti>;
    std::vector<Box> sortedBvs;
    std::vector<Ti> auxIndices, levels, parents, leafIndices;
#endif

    void build(PrimitiveObject const* prim) {
        this->prim = prim;
#if WXL
        const auto& verts = prim->verts;
        const auto& tris = prim->tris;
        if (tris.size() >= threshold) {
            const Ti numLeaves = tris.size();
            const Ti numTrunk = numLeaves - 1;
            const Ti numNodes = numLeaves + numTrunk;
            /// utilities
            auto getbv = [&verts, &tris](int tid) -> Box {
                auto ind = tris[tid];
                Box bv = std::make_pair(verts[ind[0]], verts[ind[0]]);
                for (int i = 1; i != 3; ++i) {
                    const auto& v = verts[ind[i]];
                    for (int d = 0; d != 3; ++d) {
                        if (v[d] < bv.first[d])
                            bv.first[d] = v[d];
                        if (v[d] > bv.second[d])
                            bv.second[d] = v[d];
                    }
                }
                return bv;
            };
            auto getMortonCode = [](const TV& p) -> Tu {
                auto expand_bits = [](Tu v) -> Tu { // expands lower 10-bits to 30 bits
                    v = (v * 0x00010001u) & 0xFF0000FFu;
                    v = (v * 0x00000101u) & 0x0F00F00Fu;
                    v = (v * 0x00000011u) & 0xC30C30C3u;
                    v = (v * 0x00000005u) & 0x49249249u;
                    return v;
                };
                return (expand_bits((Tu)(p[0] * 1024.f)) << (Tu)2) | (expand_bits((Tu)(p[1] * 1024.f)) << (Tu)1) |
                    expand_bits((Tu)(p[2] * 1024.f));
            };
            auto clz = [](Tu x) -> Tu {
                static_assert(std::is_same_v<Tu, unsigned int>, "Tu should be unsigned int");
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
                return __lzcnt((unsigned int)x);
#elif defined(__clang__) || defined(__GNUC__)
                return __builtin_clz((unsigned int)x);
#endif
            };

            /// total box
            constexpr auto ma = std::numeric_limits<float>::max();
            constexpr auto mi = std::numeric_limits<float>::lowest();
            Box wholeBox{ TV{ma, ma, ma}, TV{mi, mi, mi} };
            TV minVec = { ma, ma, ma };
            TV maxVec = { mi, mi, mi };
            for (int d = 0; d != 3; ++d) {
                float& v = minVec[d];
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp parallel for reduction(min : v)
#endif
#endif
                for (Ti i = 0; i < verts.size(); ++i) {
                    const auto& p = verts[i];
                    if (p[d] < v)
                        v = p[d];
                }
            }
            for (int d = 0; d != 3; ++d) {
                float& v = maxVec[d];
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp parallel for reduction(max : v)
#endif
#endif
                for (Ti i = 0; i < verts.size(); ++i) {
                    const auto& p = verts[i];
                    if (p[d] > v)
                        v = p[d];
                }
            }
            wholeBox.first = minVec;
            wholeBox.second = maxVec;

            /// morton codes
            std::vector<std::pair<Tu, Ti>> records(numLeaves); // <mc, id>
            {
                const auto lengths = wholeBox.second - wholeBox.first;
                auto getUniformCoord = [&wholeBox, &lengths](const TV& p) {
                    auto offsets = p - wholeBox.first;
                    for (int d = 0; d != 3; ++d)
                        offsets[d] = std::clamp(offsets[d], (float)0, lengths[d]) / lengths[d];
                    return offsets;
                };
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti i = 0; i < numLeaves; ++i) {
                    auto tri = tris[i];
                    auto uc = getUniformCoord((verts[tri[0]] + verts[tri[1]] + verts[tri[2]]) / 3);
                    records[i] = std::make_pair(getMortonCode(uc), i);
                }
            }
            std::sort(std::begin(records), std::end(records));

            /// precomputations
            std::vector<Tu> splits(numLeaves);
            constexpr auto numTotalBits = sizeof(Tu) * 8;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti i = 0; i < numLeaves; ++i) {
                if (i != numLeaves - 1)
                    splits[i] = numTotalBits - clz(records[i].first ^ records[i + 1].first);
                else
                    splits[i] = numTotalBits + 1;
            }
            ///
            std::vector<Box> leafBvs(numLeaves);
            std::vector<Box> trunkBvs(numLeaves - 1);
            std::vector<Ti> leafLca(numLeaves);
            std::vector<Ti> leafDepths(numLeaves);
            std::vector<Ti> trunkR(numLeaves - 1);
            std::vector<Ti> trunkLc(numLeaves - 1);

            std::vector<std::atomic<Ti>> trunkBuildFlags(numLeaves - 1); // already zero-initialized
            {
                std::vector<Ti> trunkL(numLeaves - 1);
                std::vector<Ti> trunkRc(numLeaves - 1);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (Ti idx = 0; idx < numLeaves; ++idx) {
                    leafBvs[idx] = getbv(records[idx].second);

                    leafLca[idx] = -1, leafDepths[idx] = 1;
                    Ti l = idx - 1, r = idx; ///< (l, r]
                    bool mark{ false };

                    if (l >= 0)
                        mark = splits[l] < splits[r]; ///< true when right child, false otherwise

                    int cur = mark ? l : r;
                    if (mark)
                        trunkRc[cur] = numTrunk + idx, trunkR[cur] = idx;
                    else
                        trunkLc[cur] = numTrunk + idx, trunkL[cur] = idx;

                    while (trunkBuildFlags[cur].fetch_add(1) == 1) {
                        { // refit
                            int lc = trunkLc[cur], rc = trunkRc[cur];
                            const auto& leftBox = lc >= numTrunk ? leafBvs[lc - numTrunk] : trunkBvs[lc];
                            const auto& rightBox = rc >= numTrunk ? leafBvs[rc - numTrunk] : trunkBvs[rc];
                            Box bv{};
                            for (int d = 0; d != 3; ++d) {
                                bv.first[d] =
                                    leftBox.first[d] < rightBox.first[d] ? leftBox.first[d] : rightBox.first[d];
                                bv.second[d] =
                                    leftBox.second[d] > rightBox.second[d] ? leftBox.second[d] : rightBox.second[d];
                            }
                            trunkBvs[cur] = bv;
                        }

                        l = trunkL[cur] - 1, r = trunkR[cur];
                        leafLca[l + 1] = cur, leafDepths[l + 1]++;
                        atomic_thread_fence(std::memory_order_acquire);

                        if (l >= 0)
                            mark = splits[l] < splits[r]; ///< true when right child, false otherwise
                        else
                            mark = false;

                        if (l + 1 == 0 && r == numLeaves - 1) {
                            // trunkPar(cur) = -1;
                            break;
                        }

                        int par = mark ? l : r;
                        // trunkPar(cur) = par;
                        if (mark) {
                            trunkRc[par] = cur, trunkR[par] = r;
                        }
                        else {
                            trunkLc[par] = cur, trunkL[par] = l + 1;
                        }
                        cur = par;
                    }
                }
            }

            std::vector<Ti> leafOffsets(numLeaves + 1);
            leafOffsets[0] = 0;
            for (Ti i = 0; i != numLeaves; ++i)
                leafOffsets[i + 1] = leafOffsets[i] + leafDepths[i];
            std::vector<Ti> trunkDst(numLeaves - 1);
            /// compute trunk order
            // [levels], [parents], [trunkDst]
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti i = 0; i < numLeaves; ++i) {
                auto offset = leafOffsets[i];
                parents[offset] = -1;
                for (Ti node = leafLca[i], level = leafDepths[i]; --level; node = trunkLc[node]) {
                    levels[offset] = level;
                    parents[offset + 1] = offset;
                    trunkDst[node] = offset++;
                }
            }
            // only left-branch-node's parents are set so far
            // levels store the number of node within the left-child-branch from bottom
            // up starting from 0

            /// reorder trunk
            // [sortedBvs], [auxIndices], [parents]
            // auxIndices here is escapeIndex (for trunk nodes)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti i = 0; i < numLeaves - 1; ++i) {
                const auto dst = trunkDst[i];
                const auto& bv = trunkBvs[i];
                // auto l = trunkL[i];
                auto r = trunkR[i];
                sortedBvs[dst] = bv;
                const auto rb = r + 1;
                if (rb < numLeaves) {
                    auto lca = leafLca[rb]; // rb must be in left-branch
                    auto brother = (lca != -1 ? trunkDst[lca] : leafOffsets[rb]);
                    auxIndices[dst] = brother;
                    if (parents[dst] == dst - 1)
                        parents[brother] = dst - 1; // setup right-branch brother's parent
                }
                else
                    auxIndices[dst] = -1;
            }

            /// reorder leaf
            // [sortedBvs], [auxIndices], [levels], [parents], [leafIndices]
            // auxIndices here is primitiveIndex (for leaf nodes)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (Ti i = 0; i < numLeaves; ++i) {
                const auto& bv = leafBvs[i];
                // const auto leafDepth = leafDepths[i];

                auto dst = leafOffsets[i + 1] - 1;
                leafIndices[i] = dst;
                sortedBvs[dst] = bv;
                auxIndices[dst] = records[i].second;
                levels[dst] = 0;
                if (parents[dst] == dst - 1)
                    parents[dst + 1] = dst - 1; // setup right-branch brother's parent
                                                // if (leafDepth > 1) parents[dst + 1] = dst - 1;  // setup right-branch
                                                // brother's parent
            }
        }

#endif // WXL
    }

    template <class Cond> float intersect(Cond cond, vec3f const& ro, vec3f const& rd) const
    {
        float ret = std::numeric_limits<float>::infinity();
#if WXL
        const auto& tris = prim->tris;
        if (tris.size() >= threshold) {
            const auto& verts = prim->verts;
            const Ti numLeaves = tris.size();
            const Ti numTrunk = numLeaves - 1;
            const Ti numNodes = numLeaves + numTrunk;
            Ti node = 0;
            while (node != -1 && node != numNodes) {
                Ti level = levels[node];
                // level and node are always in sync
                for (; level; --level, ++node)
                    if (!ray_box_intersect(ro, rd, sortedBvs[node]))
                        break;
                // leaf node check
                if (level == 0) {
                    const auto eid = auxIndices[node];
                    auto ind = tris[eid];
                    auto a = verts[ind[0]];
                    auto b = verts[ind[1]];
                    auto c = verts[ind[2]];
                    float d = tri_intersect(cond, ro, rd, a, b, c);
                    if (std::abs(d) < std::abs(ret))
                        ret = d;
                    if (d < ret) {
                        // id = eid;
                        ret = d;
                    }
                    node++;
                }
                else // separate at internal nodes
                    node = auxIndices[node];
            }
        }
        else {
            for (size_t i = 0; i < prim->tris.size(); i++) {
                auto ind = prim->tris[i];
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto c = prim->verts[ind[2]];
                float d = tri_intersect(cond, ro, rd, a, b, c);
                if (std::abs(d) < std::abs(ret))
                    ret = d;
            }
        }
#else
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            float d = tri_intersect(cond, ro, rd, a, b, c);
            if (std::abs(d) < std::abs(ret))
                ret = d;
        }
#endif
        return ret;
    }
};

struct WXL_PrimProject : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto targetPrim = get_input<PrimitiveObject>("targetPrim");
        auto offset = get_input2<float>("offset");
        auto limit = get_input2<float>("limit");
        auto nrmAttr = get_input2<std::string>("nrmAttr");
        auto allowDir = get_input2<std::string>("allowDir");

        BVH bvh;
        bvh.build(targetPrim.get());

        if (limit <= 0)
            limit = std::numeric_limits<float>::infinity();

        struct allow_front {
            bool operator()(float x) const {
                return x >= 0;
            }
        };

        struct allow_back {
            bool operator()(float x) const {
                return x <= 0;
            }
        };

        struct allow_both {
            bool operator()(float x) const {
                return true;
            }
        };

        auto const& nrm = prim->verts.attr<vec3f>(nrmAttr);
        auto cond = enum_variant<std::variant<allow_front, allow_back, allow_both>>(
            array_index({ "front", "back", "both" }, allowDir));

        std::visit(
            [&](auto cond) {
                parallel_for((size_t)0, prim->verts.size(), [&](size_t i) {
                    auto ro = prim->verts[i];
                    auto rd = normalizeSafe(nrm[i]);
                    float t = bvh.intersect(cond, ro, rd);
                    if (std::abs(t) >= limit)
                        t = 0;
                    t -= offset;
                    prim->verts[i] = ro + t * rd;
                    });
            },
            cond);

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(WXL_PrimProject, {
                            {
                                {"PrimitiveObject", "prim"},
                                {"PrimitiveObject", "targetPrim"},
                                {"string", "nrmAttr", "nrm"},
                                {"float", "offset", "0"},
                                {"float", "limit", "0"},
                                {"enum front back both", "allowDir", "both"},
                            },
                            {
                                {"PrimitiveObject", "prim"},
                            },
                            {},
                            {"primitive"},
    });
// WXL 

static float erode_random_float(const float min, const float max, int seed)
{
    if (seed == -1) seed = std::random_device{}();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> uni(min, max);
    float value = uni(gen);
    return value;
}

int getintersectdist(float dist, const vec3f start, const vec3f up, const float maxdist, const int hitfarthest, const int twosided)
{
    float hitu = 0;
    float hitv = 0;
    int hitprim = 0;
    vec3f hitpos = vec3f(0, 0, 0);

    vec3f raydir = up * maxdist;
    if (!hitfarthest)
        raydir *= -1;

//    hitprim = intersect('opinput:1', start, raydir, hitpos, hitu, hitv, "farthest", 1);
    // If we failed to find in the main direction, try again in opposite direction
    // but grab the closest.
    if (hitprim < 0 && twosided)
    {
    //    hitprim = intersect('opinput:1', start, raydir * -1, hitpos, hitu, hitv, "farthest", 0);
    }

    if (hitprim < 0)
    {
        return 0;
    }
    else
    {
        dist = dot(hitpos - start, up);
        return 1;
    }
}

int Pos2Idx(const int x, const int z, const int nx)
{
    return z * nx + x;
}

struct erode_project : INode {
    void apply() override {

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 地面网格标准处理过程
        //////////////////////////////////////////////////////////////////////////////////////// 

        // 获取地形
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        // 获取用户数据，里面存有网格精度
        int nx, nz;
        auto& ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
        {
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        }
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");

        // 获取网格大小，目前只支持方格
        auto& pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化数据层
        ////////////////////////////////////////////////////////////////////////////////////////

        auto heightLayerName = get_input<StringObject>("heightLayerName")->get();
        if (!terrain->verts.has_attr(heightLayerName))
        {
            zeno::log_error("no such data layer named '{}'.", heightLayerName);
        }
        auto& height = terrain->verts.attr<float>(heightLayerName);


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 创建临时属性，将外部数据拷贝到临时属性，我们将使用临时属性进行计算
        ////////////////////////////////////////////////////////////////////////////////////////

        auto& _height = terrain->verts.add_attr<float>("_height");

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                _height[idx] = height[idx];     // 外部数据拷贝到临时属性
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 获取计算所需参数
        ////////////////////////////////////////////////////////////////////////////////////////

        auto targetPrim = get_input<PrimitiveObject>("targetPrim");
        BVH bvh;
        bvh.build(targetPrim.get());

        auto hitFarthest = get_input<NumericObject>("hitFarthest")->get<int>();
        auto maxDist = get_input<NumericObject>("maxDist")->get<float>();

        auto doJitter = get_input<NumericObject>("doJitter")->get<int>();
        auto numSamples = get_input<NumericObject>("numSamples")->get<int>();
        auto jitterScale = get_input<NumericObject>("jitterScale")->get<float>();
        auto jitterCombine = get_input<NumericObject>("jitterCombine")->get<int>();
        auto seed = get_input<NumericObject>("seed")->get<int>();

        auto combineMethod = get_input<NumericObject>("combineMethod")->get<int>();
        auto allowDir = get_input<StringObject>("allowDir")->get();

        vec3f forward = vec3f(1, 0, 0);
        vec3f up = vec3f(0, 1, 0);
        vec3f right = vec3f(0, 0, 1);
        float centerDist = 0;
        float combinedDist = 0;
        float dist = 0;
        int centerIntersected = 0;
        int combinedIntersected = 0;
        int intersected = 0;

        struct allow_front {
            bool operator()(float x) const {
                return x >= 0;
            }
        };

        struct allow_back {
            bool operator()(float x) const {
                return x <= 0;
            }
        };

        struct allow_both {
            bool operator()(float x) const {
                return true;
            }
        };

        auto cond = enum_variant<std::variant<allow_front, allow_back, allow_both>>(
            array_index({ "front", "back", "both" }, allowDir));

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

        if (!doJitter)
        {
            numSamples = 1;
        }

        std::visit(
            [&](auto cond) {
                parallel_for((size_t)0, terrain->verts.size(), [&](size_t idx) {

                    vec3f basepos = pos[idx];

                    //centerIntersected = getintersectdist(centerDist, basepos, up, maxDist, hitFarthest, twosided);
                    centerDist = bvh.intersect(cond, basepos, up);
                    if (std::abs(centerDist) >= maxDist)
                        centerDist = 0;
                    centerIntersected = centerDist != 0;

                    if (numSamples == 1)
                    {
                        combinedDist = centerDist;
                        combinedIntersected = centerIntersected;
                        //zeno::log_info("----- combinedDist = {}, combinedIntersected = {}", combinedDist, combinedIntersected);
                    }
                    else
                    {
                        std::vector<float> dists;
                        if (centerIntersected)
                            dists.push_back(centerDist);

                        // Generate random jittered rays and calculate their distances to intersection
                        for (int i = 1; i < numSamples; ++i)
                        {
                            wangsrng rng(basepos[0], basepos[1], basepos[2], (i + seed) * M_PI);
                            auto mySeed = rng.next_int32();
                            float u = erode_random_float(0.0f, 1.0f, mySeed);
                            float v = erode_random_float(0.0f, 1.0f, mySeed + 1);
                            vec2f dir = vec2f(u, v) * jitterScale;
                            vec3f offset = u * right + v * forward;

                            //intersected = getintersectdist(dist, basepos + offset, up, maxDist, hitFarthest, twosided);
                            dist = bvh.intersect(cond, basepos + offset, up);
                            if (std::abs(dist) >= maxDist)
                                dist = 0;
                            intersected = dist != 0;

                            if (intersected)
                                dists.push_back(dist);
                        }

                        // Combine sample rays using specified jittercombine operation
                        combinedIntersected = 1;
                        if (dists.size() == 0)          // 没有命中点
                            combinedIntersected = 0;
                        else if (dists.size() == 1)     // 唯一命中点
                            combinedDist = dists[0];
                        // 多个随机命中点
                        else if (jitterCombine == 0)    // 多个命中点的均值
                        {
                            float sum = std::accumulate(std::begin(dists), std::end(dists), 0.0);
                            combinedDist = sum / dists.size();
                        }
                        else if (jitterCombine == 1)    // 排序后的多命中点的中间值
                        {
                            std::sort(dists.begin(), dists.end());
                            combinedDist = dists[(numSamples + 1) / 2];
                        }
                        else if (jitterCombine == 2)    // 多命中点的最小值
                        {
                            std::vector<float>::iterator minVal = std::min_element(dists.begin(), dists.end());
                            combinedDist = *minVal;
                        }
                        else if (jitterCombine == 3)    // 多命中点的最大值
                        {
                            std::vector<float>::iterator maxVal = std::max_element(dists.begin(), dists.end());
                            combinedDist = *maxVal;
                        }
                    }

                    if (combinedIntersected)
                    {
                        int method = combineMethod;
                        if (method == 0)
                            _height[idx] = combinedDist;
                        else if (method == 1)
                            _height[idx] += combinedDist;
                        else if (method == 2)
                            _height[idx] *= combinedDist;
                        else if (method == 3)
                            _height[idx] = max(combinedDist, _height[idx]);
                        else if (method == 4)
                            _height[idx] = min(combinedDist, _height[idx]);
                    }

                    });
            },
            cond);

        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 将计算结果返回给外部数据，并删除临时属性
        ////////////////////////////////////////////////////////////////////////////////////////

#pragma omp parallel for
        for (int id_z = 0; id_z < nz; id_z++)
        {
#pragma omp parallel for
            for (int id_x = 0; id_x < nx; id_x++)
            {
                int idx = Pos2Idx(id_x, id_z, nx);
                height[idx] = _height[idx]; // 计算结果返回给外部数据
            }
        }

        terrain->verts.erase_attr("_height");

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(erode_project,
    { /* inputs: */ {
            //"prim_2DGrid",
            //"targetPrim",
            {"PrimitiveObject", "prim_2DGrid"},
            {"PrimitiveObject", "targetPrim"},
            {"string", "heightLayerName", "height"},

            {"int", "hitFarthest", "1"},
            {"float", "maxDist", "1000.0"},

            {"int", "doJitter", "0"},
            {"int", "numSamples", "3"},
            {"float", "jitterScale", "0.25"},
            {"int", "jitterCombine", "1"},
            {"int", "seed", "1"},
            {"int", "combineMethod", "3"},
            {"enum front back both", "allowDir", "both"},
        }, /* outputs: */ {
            "prim_2DGrid"
        }, /* params: */ {
        }, /* category: */ {
            "erode"
        } });

struct HeightStarPattern : zeno::INode {
    virtual void apply() override {
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
            auto coord = prim->verts.attr<vec3f>("res")[i];
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

ZENDEFNODE(HeightStarPattern, {/* inputs: */ {
                                   {"PrimitiveObject", "prim"},
                                   {"float", "rotate", "0"},
                                   {"float", "anglerandom", "0"},
                                   {"float", "shapesize", "0.5"},
                                   {"float", "posjitter", "0"},
                                   {"float", "sharpness", "0.5"},
                                   {"float", "starness", "0.5"},
                                   {"int", "sides", "5"},
                               },
                               /* outputs: */
                               {
                                   {"PrimitiveObject", "prim"},
                               },
                               /* params: */ {}, /* category: */
                               {
                                   "erode",
                               }});
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





} // namespace
} // namespace zeno