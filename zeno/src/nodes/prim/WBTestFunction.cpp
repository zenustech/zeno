
//
// WangBo 2023/02/03.
//

#include <zeno/zeno.h>

#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/UserData.h>

#include <zeno/core/Graph.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/extra/GlobalState.h>

/* #include <zfx/zfx.h> */
/* #include <zfx/x64.h> */
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

        auto prim = std::make_unique<PrimitiveObject>();

        for (int i = 0; i < verts.size(); i++) {
            prim->verts.push_back(verts[i]);
        }

        for (int i = 0; i < triangles.size(); i++) {
            prim->tris.push_back(triangles[i]);
        }

        ZImpl(set_output("prim", std::move(prim)));
    }
};
ZENDEFNODE(testPoly1, {
    /* inputs: */
    {
    },
    /* outputs: */
    {
{gParamType_Primitive, "prim"},
},
    /* params: */ {}, /* category: */
    {
        "WBTest",
    }});


struct testPoly2 : INode {
    void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto list = ZImpl(get_input<ListObject>("list"))->m_impl->getLiterial<int>();

        std::vector<vec3i> triangles;
        polygonDecompose(prim->verts, list, triangles);

        for (int i = 0; i < triangles.size(); i++) {
            prim->tris.push_back(triangles[i]);
        }

        ZImpl(set_output("prim", std::move(prim)));
    }
};
ZENDEFNODE(testPoly2, {
    /* inputs: */
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_List, "list", "", zeno::Socket_ReadOnly},
    },
    /* outputs: */
    {
{gParamType_Primitive, "prim"},
},
    /* params: */
    {
    },
    /* category: */
    {
        "WBTest",
    }});


struct PrimMarkTrisIdx : INode {
    void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto idxName = ZImpl(get_input<StringObject>("idxName"))->get();
        auto &tris_idx = prim->tris.add_attr<int>(idxName);

        for (int i = 0; i < int(prim->tris.size()); i++) {
            tris_idx[i] = i;
        }
        ZImpl(set_output("prim", std::move(prim)));
    }
};
ZENDEFNODE(PrimMarkTrisIdx, {
    /* inputs: */
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_String, "idxName", "index"},
    },
    /* outputs: */
    {
{gParamType_Primitive, "prim"},
},
    /* params: */ {}, /* category: */
    {
        "WBTest",
    }});


struct PrimGetTrisSize : INode {
    void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto n = std::make_unique<NumericObject>();
        n->set<int>(int(prim->tris.size()));
        ZImpl(set_output("TrisSize", std::move(n)));
    }
};
ZENDEFNODE(PrimGetTrisSize, {
    /* inputs: */
    {
        {gParamType_Primitive, "prim", "", Socket_ReadOnly},
    },
    /* outputs: */
    {
        {gParamType_Int, "TrisSize", "0"},
    },
    /* params: */ {}, /* category: */
    {
        "WBTest",
    }});


struct PrimPointTris : INode {
    void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto index = ZImpl(get_input<NumericObject>("pointID"))->get<int>();
        auto list = create_ListObject();

        for (int i = 0; i < int(prim->tris.size()); i++)
        {
            auto const &ind = prim->tris[i];
            if (ind[0] == index || ind[1] == index || ind[2] == index)
            {
                auto num = std::make_unique<NumericObject>();
                vec4i x;
                x[0] = ind[0];
                x[1] = ind[1];
                x[2] = ind[2];
                x[3] = i;
                num->set<vec4i>(x);
                list->m_impl->push_back(num->clone());
            }
        }
        ZImpl(set_output("list", std::move(list)));
    }
};
ZENDEFNODE(PrimPointTris, {
    /* inputs: */
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_Int, "pointID", "0"},
    },
    /* outputs: */
    {
        {gParamType_List, "list"},
    },
    /* params: */ {}, /* category: */
    {
        "WBTest",
    }});


struct PrimTriPoints : INode {
    void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto index = ZImpl(get_input<NumericObject>("trisID"))->get<int>();
        auto points = std::make_unique<NumericObject>();
        points->set<vec3i>(prim->tris[index]);
        ZImpl(set_output("points", std::move(points)));
    }
};
ZENDEFNODE(PrimTriPoints, {
    /* inputs: */
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_Int, "trisID", "0"},
    },
    /* outputs: */
    {
        {"vec3i","points"},
    },
    /* params: */
    {
    },
    /* category: */
    {
        "WBTest",
    }});


struct DictEraseItem : zeno::INode {
    virtual void apply() override {
        auto dict = ZImpl(get_input<zeno::DictObject>("dict"));
        auto key = ZImpl(get_input<zeno::StringObject>("key"))->get();
        dict->lut.erase(key);
        ZImpl(set_output("dict", std::move(dict)));
    }
};
ZENDEFNODE(DictEraseItem, {
    /* inputs: */
    {
        {gParamType_Dict,"dict", "", zeno::Socket_ReadOnly},
        {gParamType_String, "key"},
    },
    /* outputs: */
    {
        {gParamType_Dict,"dict"},
    },
    /* params: */
    {
    },
    /* category: */
    {
        "WBTest",
    }});


struct str2num : INode {
    virtual void apply() override {
        auto str = ZImpl(get_input2<std::string>("str"));
        auto type = ZImpl(get_input<zeno::StringObject>("type"))->value;
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

        ZImpl(set_output("num", std::move(obj)));
    }
};
ZENDEFNODE(str2num, {
    /* inputs: */
    {
        {"enum float int", "type", "int"},
        {gParamType_String, "str", "0"},
    },
    /* outputs: */
    {
        {"object", "num"},
    },
    /* params: */
    {
    },
    /* category: */
    {
        "deprecated",
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

#if 0
struct VisPrimAttrValue_Modify : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto attrName = ZImpl(get_input2<std::string>("attrName"));
        auto scale = ZImpl(get_input2<float>("scale"));
        auto precision = ZImpl(get_input2<int>("precision"));
        auto includeSelf = ZImpl(get_input2<bool>("includeSelf"));
        auto dotDecoration = ZImpl(get_input2<bool>("dotDecoration"));
        bool textDecoration = !attrName.empty();

        std::vector<PrimitiveObject *> outprim2;
        std::vector<std::unique_ptr<PrimitiveObject>> outprim;

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
                      ZImpl(getThisGraph())
                          ->callTempNode("LoadStringPrim",
                                         {
                                             {"triangulate", std::make_unique<NumericObject>((bool)0)},
                                             {"decodeUVs", std::make_unique<NumericObject>((bool)0)},
                                             {"str", objectFromLiterial(str)},
                                         })
                          .at("prim"));
                  //auto numprim = std::make_unique<PrimitiveObject>();
                  for (int j = 0; j < numprim->verts.size(); j++) {
                      auto &v = numprim->verts[j];
                      // v = (v + vec3f(dotDecoration ? 0.5f : 0.3f, 0.15f, 0.0f)) * scale + pos;
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
                ZImpl(getThisGraph())
                    ->callTempNode("LoadSampleModel",
                                   {
                                       {"triangulate", std::make_unique<NumericObject>((bool)0)},
                                       {"decodeUVs", std::make_unique<NumericObject>((bool)0)},
                                       {"name", objectFromLiterial("star")},
                                   })
                    .at("prim"));
#pragma omp parallel for
            for (int i = 0; i < attarrsize; i++) {
                auto pos = prim->verts[i];
                auto offprim = std::make_unique<PrimitiveObject>(*numprim);
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

        ZImpl(set_output("outPrim", std::move(retprim)));
    }
};
ZENO_DEFNODE(VisPrimAttrValue_Modify)( {
     {
         {gParamType_Primitive, "prim"},
         {gParamType_String, "attrName", "pos"},
         {gParamType_Float, "scale", "0.05"},
         {gParamType_Int, "precision", "3"},
         {gParamType_Bool, "includeSelf", "0"},
         {gParamType_Bool, "dotDecoration", "0"},
     },
     {
         {gParamType_Primitive, "outPrim"},
     },
     {},
     {"WBTest"},
    });
#endif


// FDGather.cpp
template <class T>
T lerp(T a, T b, float c) {
    return (1.0 - c) * a + c * b;
}

template <class T>
void sample2D_M(const std::vector<zeno::vec3f> &coord, std::vector<T> &field, const std::vector<T> &field2, int nx, int ny, float h,
                zeno::vec3f bmin) {
    std::vector<T> temp(field.size());
#pragma omp parallel for
    for (auto tidx = 0; tidx < coord.size(); tidx++) {
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
    for (auto tidx = 0; tidx < coord.size(); tidx++) {
        field[tidx] = temp[tidx];
    }
}
struct Grid2DSample_M : zeno::INode {
    virtual void apply() override {
        auto nx = ZImpl(get_input<zeno::NumericObject>("nx"))->get<int>();
        auto ny = ZImpl(get_input<zeno::NumericObject>("ny"))->get<int>();
        auto bmin = ZImpl(get_input2<zeno::vec3f>("bmin"));
        auto grid = clone_input_Geometry("grid");
        auto grid2 = get_input_Geometry("grid2");
        auto attrT = get_input2_string("attrT");
        auto channel = get_input2_string("channel");
        auto sampleby = get_input2_string("sampleBy");
        auto h = get_input2_float("h");
        if (grid->has_point_attr(channel) && grid->has_point_attr(sampleby)) {
            if (attrT == "float") {
                const auto& dat1 =   grid->get_vec3f_attr(ATTR_POINT, sampleby);
                auto dat2 = grid->get_float_attr(ATTR_POINT, channel);
                const auto& dat3 = grid2->get_float_attr(ATTR_POINT, channel);
                sample2D_M<float>(dat1, dat2, dat3, nx, ny, h, bmin);
                grid->set_point_attr(channel, dat2);
            }
            else if (attrT == "vec3f") {
                const auto& dat1 = grid->get_vec3f_attr(ATTR_POINT, sampleby);
                auto dat2 = grid->get_vec3f_attr(ATTR_POINT, channel);
                const auto& dat3 = grid2->get_vec3f_attr(ATTR_POINT, channel);
                sample2D_M<zeno::vec3f>(dat1, dat2, dat3, nx, ny, h, bmin);
                grid->set_point_attr(channel, dat2);
            }
        }
        set_output("prim", std::move(grid));
    }
};
ZENDEFNODE(Grid2DSample_M, {
    /* inputs: */
    {
        {gParamType_Geometry, "grid"},
        {gParamType_Geometry, "grid2"},
        {gParamType_Int, "nx", "1"},
        {gParamType_Int, "ny", "1"},
        {gParamType_Float, "h", "1"},
        {gParamType_Vec3f, "bmin", "0,0,0"},
        {gParamType_String, "channel", "pos"},
        {gParamType_String, "sampleBy", "pos"},
    },
    /* outputs: */
    {
        {gParamType_Geometry, "prim"},
    },
    /* params: */
    {
        {"enum vec3 float", "attrT", "float"},
    },
    /* category: */
    {
        "deprecated",
    }});



template <class T>
static void sample2D(const std::vector<zeno::vec3f>& coord, const std::vector<T>& field, std::vector<T>& primAttr, int nx, int ny, float h,
    zeno::vec3f bmin, bool isPeriodic) {
    std::vector<T> temp(coord.size());
//#pragma omp parallel for
    for (auto tidx = 0; tidx < coord.size(); tidx++) {
        auto uv = coord[tidx];
        zeno::vec3f uv2;
        if (isPeriodic) {
            auto Lx = (nx - 1) * h;
            auto Ly = (ny - 1) * h;
            int gid_x = std::floor((uv[0] - bmin[0]) / Lx);
            int gid_y = std::floor((uv[2] - bmin[2]) / Ly);
            uv2 = (uv - (bmin + zeno::vec3f{ gid_x * Lx, 0, gid_y * Ly })) / h;
            uv2 = zeno::min(zeno::max(uv2, zeno::vec3f(0.00, 0.0, 0.00)),
                zeno::vec3f((float)nx - 1.01, 0.0, (float)ny - 1.01));
        }
        else {
            uv2 = (uv - bmin) / h;
            uv2 = zeno::min(zeno::max(uv2, zeno::vec3f(0.01, 0.0, 0.01)),
                zeno::vec3f((float)nx - 1.01, 0.0, (float)ny - 1.01));
        }

        int i = uv2[0];
        int j = uv2[2];
        float cx = uv2[0] - i, cy = uv2[2] - j;
        size_t idx00 = j * nx + i, idx01 = j * nx + i + 1, idx10 = (j + 1) * nx + i, idx11 = (j + 1) * nx + i + 1;
        temp[tidx] = lerp<T>(lerp<T>(field[idx00], field[idx01], cx), lerp<T>(field[idx10], field[idx11], cx), cy);
    }
//#pragma omp parallel for
    for (auto tidx = 0; tidx < coord.size(); tidx++) {
        primAttr[tidx] = temp[tidx];
    }
}

struct Grid2DSample_M2 : zeno::INode {
    virtual void apply() override {
        auto nx = get_input2_int("nx");
        auto ny = get_input2_int("ny");
        auto bmin = toVec3f(get_input2_vec3f("bmin"));
        auto prim = clone_input_Geometry("prim");
        auto grid = get_input_Geometry("sampleGrid");
        auto channelList = get_input2_string("channel");
        auto sampleby = get_input2_string("sampleBy");
        auto isPeriodic = get_input2_string("sampleType") == "Periodic";
        auto h = get_input2_float("h");

        bool sampleAll = false;

        std::vector<std::string> channels;
        std::istringstream iss(zsString2Std(channelList));
        std::string word;
        while (iss >> word) {
            if (word == "*") {
                sampleAll = true;
                channels = grid->attributes(ATTR_POINT);
                break;
            }
            channels.push_back(word);
        }

        if (prim->has_point_attr(sampleby)) {
            if (!(sampleby == "pos" || prim->has_attr(ATTR_POINT, sampleby, ATTR_VEC3)))
                throw std::runtime_error("[sampleBy] has to be a vec3f attribute!");

            for (const auto& ch : channels) {
                if (ch != "pos") {
                    auto _ch = stdString2zs(ch);
                    GeoAttrType type = grid->get_attr_type(ATTR_POINT, _ch);
                    if (type == ATTR_VEC3) {
                        if (!prim->has_point_attr(_ch)) {
                            prim->create_point_attr(_ch, zeno::vec3f());
                        }
                        std::vector<vec3f> primAttr = prim->get_vec3f_attr(ATTR_POINT, _ch);
                        sample2D<vec3f>(
                            prim->get_vec3f_attr(ATTR_POINT, sampleby),
                            grid->get_vec3f_attr(ATTR_POINT, stdString2zs(ch)),
                            primAttr,
                            nx, ny, h, bmin, isPeriodic);
                        prim->set_point_attr(_ch, primAttr);
                    }
                    else if (type == ATTR_FLOAT) {
                        if (!prim->has_point_attr(_ch)) {
                            prim->create_point_attr(_ch, 0.f);
                        }
                        std::vector<float> primAttr = prim->get_float_attr(ATTR_POINT, _ch);
                        sample2D<float>(
                            prim->get_vec3f_attr(ATTR_POINT, sampleby),
                            grid->get_float_attr(ATTR_POINT, stdString2zs(ch)),
                            primAttr,
                            nx, ny, h, bmin, isPeriodic);
                        prim->set_point_attr(_ch, primAttr);
                    }
                    else {
                        throw makeError<UnimplError>("only support vec3f or float in Grid2DSample");
                    }
                    /*std::visit(
                        [&](auto&& ref) {
                            using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                            prim->add_attr<T>(ch);
                            sample2D<T>(prim->attr<zeno::vec3f>(sampleby), grid->attr<T>(ch), prim->attr<T>(ch), nx, ny,
                                h, bmin, isPeriodic);
                        },
                        grid->attr(ch));*/
                }
            }
        }
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(Grid2DSample_M2, {
                             {{gParamType_Geometry, "prim"},
                              {gParamType_Geometry, "sampleGrid"},
                              {gParamType_Int, "nx", "1"},
                              {gParamType_Int, "ny", "1"},
                              {gParamType_Float, "h", "1"},
                              {gParamType_Vec3f, "bmin", "0,0,0"},
                              {gParamType_String, "channel", "*"},
                              {gParamType_String, "sampleBy", "pos"},
                              {"enum Clamp Periodic", "sampleType", "Clamp"}},
                             {{gParamType_Geometry, "prim"}},
                             {},
                             {"zenofx"},
    });



} // namespace
} // namespace zeno