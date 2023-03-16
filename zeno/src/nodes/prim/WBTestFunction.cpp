//
// WangBo 2023/02/03.
//

#include <zeno/zeno.h>

#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/UserData.h>

#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/log.h>

#include <zeno/core/Graph.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/extra/GlobalState.h>

#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <glm/gtx/quaternion.hpp>
#include <cmath>
#include <sstream>
#include <random>
#include <numeric>
#include <cassert>


namespace zeno
{
namespace
{

struct testPoly1 : INode {
    void apply() override {
        std::vector<zeno::vec3f> verts = {vec3f(0,0,0), vec3f(1,0,0), vec3f(0,0,1), vec3f(1,0,1)};
        std::vector<int> poly = {0, 1, 3, 2};
        std::vector<vec3i> triangles;

        polygonDecompose(verts, poly, triangles);
        //printf("x0 = %i, y0 = %i, z0 = %i\n", triangles[0][0], triangles[0][1], triangles[0][2]);

        auto prim = std::make_shared<PrimitiveObject>();

        for (int i = 0; i < verts.size(); i++) {
            prim->verts.push_back(verts[i]);
        }

        for (int i = 0; i < triangles.size(); i++) {
            prim->tris.push_back(triangles[i]);
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(testPoly1, {/* inputs: */ {
                       },
                       /* outputs: */
                       {
                           "prim",
                       },
                       /* params: */ {}, /* category: */
                       {
                           "WBTest",
                       }});


struct testPoly2 : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto list = get_input<ListObject>("list")->getLiterial<int>();

        std::vector<vec3i> triangles;
        polygonDecompose(prim->verts, list, triangles);

        for (int i = 0; i < triangles.size(); i++) {
            prim->tris.push_back(triangles[i]);
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(testPoly2, {
                    /* inputs: */ 
                    {
                        "prim", 
                        "list",
                    },
                    /* outputs: */
                    {
                        "prim",
                    },
                    /* params: */ 
                    {
                    },
                    /* category: */
                    {
                        "WBTest",
                    }});


/// <summary>
/// 以下是退役节点，为了跑老图又加回来的 //////////////////////////////
/// </summary>
struct AddVertID : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto &id = prim->verts.add_attr<float>("ID");
        for (int i = 0; i < int(prim->verts.size()); i++) {
            id[i] = i;
        }

        set_output("primOut", std::move(prim));
    }
};
ZENDEFNODE(AddVertID, {/* inputs: */ {
                           "prim",
                       },
                       /* outputs: */
                       {
                           "primOut",
                       },
                       /* params: */ {}, /* category: */
                       {
                           "WBTest",
                       }});

struct PrimAddVec3fAttr : INode {
    void apply() override {
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 1;

        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto value = get_input<NumericObject>("value")->get<vec3f>();
        if (!prim->has_attr(name)) {
            prim->add_attr<vec3f>(name, value);
        } else {
            auto &attr_arr = prim->attr<vec3f>(name);
#pragma omp parallel for
            for (intptr_t i = 0; i < prim->size(); i++) {
                attr_arr[i] = value;
            }
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimAddVec3fAttr, {/* inputs: */ {
                                  "prim",
                                  {"string", "name", "pos"},
                                  {"vec3f", "value", "0,0,0"},
                              },
                              /* outputs: */
                              {
                                  "prim",
                              },
                              /* params: */ {}, /* category: */
                              {
                                  "deprecated", // PrimFillAttr
                              }});

struct PrimAddFloatAttr : INode {
    void apply() override {
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 1;

        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto value = get_input<NumericObject>("value")->get<float>();

        if (!prim->has_attr(name)) {
            prim->add_attr<float>(name, value);
        } else {
            auto &attr_arr = prim->attr<float>(name);
#pragma omp parallel for
            for (intptr_t i = 0; i < prim->size(); i++) {
                attr_arr[i] = value;
            }
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimAddFloatAttr, {/* inputs: */ {
                                  "prim",
                                  {"string", "name", "pos"},
                                  {"float", "value", "0"},
                              },
                              /* outputs: */
                              {
                                  "prim",
                              },
                              /* params: */ {}, /* category: */
                              {
                                  "deprecated", // PrimFillAttr
                              }});

// 根据属性名称设置每点属性，name 是 input
struct PrimSetAttrPerVertByName : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");
        auto index = get_input<NumericObject>("index")->get<int>();

        if (type == "float") {
            auto value = get_input<NumericObject>("value")->get<float>();
            auto &attr_arr = prim->attr<float>(name);
            if (index < attr_arr.size()) {
                attr_arr[index] = value;
            }
        } else if (type == "float3") {
            auto value = get_input<NumericObject>("value")->get<vec3f>();
            auto &attr_arr = prim->attr<vec3f>(name);
            if (index < attr_arr.size()) {
                attr_arr[index] = value;
            }
        } else {
            throw Exception("Bad attribute type: " + type);
        }
    }
};
ZENDEFNODE(PrimSetAttrPerVertByName, {/* inputs: */ {
                                          "prim",
                                          {"string", "name", "pos"},
                                          {"enum float float3", "type", "float3"},
                                          {"int", "index", "0"},
                                          "value",
                                      },
                                      /* outputs: */ {},
                                      /* params: */ {}, /* category: */
                                      {
                                          "deprecated", // PrimitiveSetAttrValue
                                      }});

// 根据属性名称获取每点属性，name 是 input
struct PrimGetAttrPerVertByName : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");
        auto index = get_input<NumericObject>("index")->get<int>();
        auto value = std::make_shared<NumericObject>();

        if (type == "float") {
            value->set<float>(0);
            auto &attr_arr = prim->attr<float>(name);
            if (index < attr_arr.size()) {
                value->set<float>(attr_arr[index]);
            }
        } else if (type == "float3") {
            value->set<vec3f>(vec3f(0, 0, 0));
            auto &attr_arr = prim->attr<vec3f>(name);
            if (index < attr_arr.size()) {
                value->set<vec3f>(attr_arr[index]);
            }
        } else {
            throw Exception("Bad attribute type: " + type);
        }

        set_output("value", std::move(value));
    }
};
ZENDEFNODE(PrimGetAttrPerVertByName, {/* inputs: */ {
                                          "prim",
                                          {"string", "name", "pos"},
                                          {"enum float float3", "type", "float3"},
                                          {"int", "index", "0"},
                                      },
                                      /* outputs: */
                                      {
                                          "value",
                                      },
                                      /* params: */ {}, /* category: */
                                      {
                                          "deprecated", // PrimitiveGetAttrValue
                                      }});

///////////////////////////////////////////////////////////////////////////////
// 2023.01.05 节点图中自撸循环，遍历 prim 获取属性
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

                if (method == "vert") {
                    if (!prim->has_attr(name)) {
                        prim->add_attr<T>(name);
                    }
                    auto &attr_arr = prim->attr<T>(name);

                    auto val = value->get<T>();
                    if (index < attr_arr.size()) {
                        attr_arr[index] = val;
                    }
                } else if (method == "tri") {
                    if (!prim->tris.has_attr(name)) {
                        prim->tris.add_attr<T>(name);
                    }
                    auto &attr_arr = prim->tris.attr<T>(name);
                    auto val = value->get<T>();
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
ZENDEFNODE(PrimSetAttr, {/* inputs: */ {
                                         "prim",
                                         {"int", "value", "0"},
                                         {"string", "name", "index"},
                                         {"enum float vec2f vec3f vec4f int vec2i vec3i vec4i", "type", "int"},
                                         {"enum vert tri", "method", "tri"},
                                         {"int", "index", "0"},
                                     },
                                     /* outputs: */
                                     {
                                         "prim",
                                     },
                                     /* params: */ {}, /* category: */
                                     {
                                         "WBTest",
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
                } else {
                    throw Exception("bad type: " + method);
                }
            },
            enum_variant<std::variant<float, vec2f, vec3f, vec4f, int, vec2i, vec3i, vec4i>>(
                array_index({"float", "vec2f", "vec3f", "vec4f", "int", "vec2i", "vec3i", "vec4i"}, type)));

        set_output("value", std::move(value));
    }
};
ZENDEFNODE(PrimGetAttr, {/* inputs: */ {
                                         "prim",
                                         {"string", "name", "index"},
                                         {"enum float vec2f vec3f vec4f int vec2i vec3i vec4i", "type", "int"},
                                         {"enum vert tri", "method", "tri"},
                                         {"int", "index", "0"},
                                     },
                                     /* outputs: */
                                     {
                                         "value",
                                     },
                                     /* params: */ {}, /* category: */
                                     {
                                         "WBTest",
                                     }});


struct PrimMarkTrisIdx : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto idxName = get_input<StringObject>("idxName")->get();
        auto &tris_idx = prim->tris.add_attr<int>(idxName);

        for (int i = 0; i < int(prim->tris.size()); i++) {
            tris_idx[i] = i;
        }
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimMarkTrisIdx, {/* inputs: */ {
                           "prim",
                           {"string", "idxName", "index"},
                       },
                       /* outputs: */
                       {
                           "prim",
                       },
                       /* params: */ {}, /* category: */
                       {
                           "WBTest",
                       }});


struct PrimGetTrisSize : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto n = std::make_shared<NumericObject>();
        n->set<int>(int(prim->tris.size()));
        set_output("TrisSize", n);
    }
};
ZENDEFNODE(PrimGetTrisSize, {/* inputs: */ {
                                 "prim",
                             },
                             /* outputs: */
                             {
                                 {"int", "TrisSize", "0"},
                             },
                             /* params: */ {}, /* category: */
                             {
                                 "WBTest",
                             }});


struct PrimPointTris : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto index = get_input<NumericObject>("pointID")->get<int>();
        auto list = std::make_shared<ListObject>();

        for (int i = 0; i < int(prim->tris.size()); i++)
        {
            auto const &ind = prim->tris[i];
            if (ind[0] == index || ind[1] == index || ind[2] == index)
            {
                auto num = std::make_shared<NumericObject>();
                vec4i x;
                x[0] = ind[0];
                x[1] = ind[1];
                x[2] = ind[2];
                x[3] = i;
                num->set<vec4i>(x);
                list->arr.push_back(num);
            }
        }
        set_output("list", std::move(list));
    }
};
ZENDEFNODE(PrimPointTris, {/* inputs: */ {
                              "prim",
                              {"int", "pointID", "0"},
                          },
                          /* outputs: */
                          {
                              "list",
                          },
                          /* params: */ {}, /* category: */
                          {
                              "WBTest",
                          }});


struct PrimTriPoints : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto index = get_input<NumericObject>("trisID")->get<int>();
        auto points = std::make_shared<NumericObject>();
        points->set<vec3i>(prim->tris[index]);
        set_output("points", std::move(points));
    }
};
ZENDEFNODE(PrimTriPoints, {/* inputs: */ {
                               "prim",
                               {"int", "trisID", "0"},
                           },
                           /* outputs: */
                           {
                               "points",
                           },
                           /* params: */ {}, /* category: */
                           {
                               "WBTest",
                           }});


struct DictEraseItem : zeno::INode {
    virtual void apply() override {
        auto dict = get_input<zeno::DictObject>("dict");
        auto key = get_input<zeno::StringObject>("key")->get();
        dict->lut.erase(key);
        set_output("dict", std::move(dict));
    }
};
ZENDEFNODE(DictEraseItem, {
                              {{"DictObject", "dict"}, {"string", "key"}},
                              {{"DictObject", "dict"}},
                              {},
                              {"dict"},
                          });


struct str2num : INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto type = get_input<zeno::StringObject>("type")->value;
        auto obj = std::make_unique<zeno::NumericObject>();
        std::stringstream strStream(str);

        float num_float = 0;
        int num_int = 0;
        if (type == "float") {
            strStream >> num_float;
        }
        if (type == "int") {
            strStream >> num_int;
        }

        if (strStream.bad()) {
            throw zeno::makeError("[string format error]");
        } else if (strStream.fail()) {
            strStream.clear();
            strStream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            throw zeno::makeError("[string format error]");
        } else {
            if (type == "float") {
                obj->set(num_float);
            }
            if (type == "int") {
                obj->set(num_int);
            }
        }

        set_output("num", std::move(obj));
    }
};
ZENDEFNODE(str2num, {{ /* inputs: */
                        {"enum float int", "type", "int"},
                        {"string", "str", "0"},
                    }, { /* outputs: */
                        "num",
                    }, { /* params: */
                    }, { /* category: */
                        "WBTest",
                    }});





template <class T>
struct number_printer {
    void operator()(std::ostringstream &ss, T const &value) {
        ss << value;
    }
};

template <size_t N, class T>
struct number_printer<vec<N, T>> {
    void operator()(std::ostringstream &ss, vec<N, T> const &value) {
        ss << value[0];
        for (size_t i = 1; i < N; i++)
            ss << ',' << value[i];
    }
};

struct VisPrimAttrValue_Modify : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto attrName = get_input2<std::string>("attrName");
        auto scale = get_input2<float>("scale");
        auto precision = get_input2<int>("precision");
        auto includeSelf = get_input2<bool>("includeSelf");
        auto dotDecoration = get_input2<bool>("dotDecoration");
        bool textDecoration = !attrName.empty();

        std::vector<PrimitiveObject *> outprim2;
        std::vector<std::shared_ptr<PrimitiveObject>> outprim;
        if (textDecoration) {
            prim->verts.attr_visit<AttrAcceptAll>(attrName, [&](auto const &attarr) {
                outprim.resize(attarr.size());
                outprim2.resize(attarr.size());
#pragma omp parallel for
                for (int i = 0; i < attarr.size(); i++) {
                    auto value = attarr[i];
                    auto pos = prim->verts[i];
                    std::ostringstream ss;
                    ss << std::setprecision(precision);
                    number_printer<std::decay_t<decltype(value)>>{}(ss, value);
                    auto str = ss.str();

                    auto numprim = std::static_pointer_cast<PrimitiveObject>(
                        getThisGraph()
                            ->callTempNode("LoadStringPrim",
                                           {
                                               {"triangulate", std::make_shared<NumericObject>((bool)0)},
                                               {"decodeUVs", std::make_shared<NumericObject>((bool)0)},
                                               {"str", objectFromLiterial(str)},
                                           })
                            .at("prim"));
                    //auto numprim = std::make_shared<PrimitiveObject>();
                    for (int j = 0; j < numprim->verts.size(); j++) {
                        auto &v = numprim->verts[j];
                        //                        v = (v + vec3f(dotDecoration ? 0.5f : 0.3f, 0.15f, 0.0f)) * scale + pos;
                        v = (v + vec3f(dotDecoration ? 0.5f : 0.3f, 0.15f, 0.0f));
                        v[2] *= 0.1;
                        v = v * scale + pos;
                    }
                    outprim2[i] = numprim.get();
                    outprim[i] = std::move(numprim);
                }
            });
        }
        if (dotDecoration) {
            int attarrsize = textDecoration ? outprim.size() : prim->verts.size();
            outprim.resize(attarrsize * (1 + (int)textDecoration));
            outprim2.resize(attarrsize * (1 + (int)textDecoration));
            auto numprim = std::static_pointer_cast<PrimitiveObject>(
                getThisGraph()
                    ->callTempNode("LoadSampleModel",
                                   {
                                       {"triangulate", std::make_shared<NumericObject>((bool)0)},
                                       {"decodeUVs", std::make_shared<NumericObject>((bool)0)},
                                       {"name", objectFromLiterial("star")},
                                   })
                    .at("prim"));
#pragma omp parallel for
            for (int i = 0; i < attarrsize; i++) {
                auto pos = prim->verts[i];
                auto offprim = std::make_shared<PrimitiveObject>(*numprim);
                for (int j = 0; j < offprim->verts.size(); j++) {
                    auto &v = offprim->verts[j];
                    v = v * (scale * 0.25f) + pos;
                }
                outprim2[i + attarrsize * (int)textDecoration] = offprim.get();
                outprim[i + attarrsize * (int)textDecoration] = std::move(offprim);
            }
        }
        if (includeSelf) {
            outprim2.push_back(prim.get());
        }

        auto retprim = primMerge(outprim2);
        set_output("outPrim", std::move(retprim));
    }
};
ZENO_DEFNODE(VisPrimAttrValue_Modify)
({
    {
        {"prim"},
        {"string", "attrName", "pos"},
        {"float", "scale", "0.05"},
        {"int", "precision", "3"},
        {"bool", "includeSelf", "0"},
        {"bool", "dotDecoration", "0"},
    },
    {
        {"outPrim"},
    },
    {},
    {"WBTest"},
});


// FDGather.cpp
template <class T>
T lerp(T a, T b, float c) {
    return (1.0 - c) * a + c * b;
}

template <class T>
void sample2D_M(std::vector<zeno::vec3f> &coord, std::vector<T> &field, std::vector<T> &field2, int nx, int ny, float h,
                zeno::vec3f bmin) {
    std::vector<T> temp(field.size());
#pragma omp parallel for
    for (size_t tidx = 0; tidx < coord.size(); tidx++) {
        auto uv = coord[tidx];
        auto uv2 = (uv - bmin) / h;
        uv2 = zeno::min(zeno::max(uv2, zeno::vec3f(0.01, 0.0, 0.01)), zeno::vec3f(nx - 1.01, 0.0, ny - 1.01));
        int i = uv2[0];
        int j = uv2[2];
        float cx = uv2[0] - i, cy = uv2[2] - j;
        size_t idx00 = j * nx + i, idx01 = j * nx + i + 1, idx10 = (j + 1) * nx + i, idx11 = (j + 1) * nx + i + 1;
        temp[tidx] = lerp<T>(lerp<T>(field2[idx00], field2[idx01], cx), lerp<T>(field2[idx10], field2[idx11], cx), cy);
    }
#pragma omp parallel for
    for (size_t tidx = 0; tidx < coord.size(); tidx++) {
        field[tidx] = temp[tidx];
    }
}
struct Grid2DSample_M : zeno::INode {
    virtual void apply() override {
        auto nx = get_input<zeno::NumericObject>("nx")->get<int>();
        auto ny = get_input<zeno::NumericObject>("ny")->get<int>();
        auto bmin = get_input2<zeno::vec3f>("bmin");
        auto grid = get_input<zeno::PrimitiveObject>("grid");
        auto grid2 = get_input<zeno::PrimitiveObject>("grid2");
        auto attrT = get_param<std::string>("attrT");
        auto channel = get_input<zeno::StringObject>("channel")->get();
        auto sampleby = get_input<zeno::StringObject>("sampleBy")->get();
        auto h = get_input<zeno::NumericObject>("h")->get<float>();
        if (grid->has_attr(channel) && grid->has_attr(sampleby)) {
            if (attrT == "float") {
                sample2D_M<float>(grid->attr<zeno::vec3f>(sampleby), grid->attr<float>(channel),
                                  grid2->attr<float>(channel), nx, ny, h, bmin);
            } else if (attrT == "vec3f") {
                sample2D_M<zeno::vec3f>(grid->attr<zeno::vec3f>(sampleby), grid->attr<zeno::vec3f>(channel),
                                        grid2->attr<zeno::vec3f>(channel), nx, ny, h, bmin);
            }
        }

        set_output("prim", std::move(grid));
    }
};
ZENDEFNODE(Grid2DSample_M, {
                               {{"PrimitiveObject", "grid"},
                                {"PrimitiveObject", "grid2"},
                                {"int", "nx", "1"},
                                {"int", "ny", "1"},
                                {"float", "h", "1"},
                                {"vec3f", "bmin", "0,0,0"},
                                {"string", "channel", "pos"},
                                {"string", "sampleBy", "pos"}},
                               {{"PrimitiveObject", "prim"}},
                               {
                                   {"enum vec3 float", "attrT", "float"},
                               },
                               {"zenofx"},
                           });



} // namespace
} // namespace zeno