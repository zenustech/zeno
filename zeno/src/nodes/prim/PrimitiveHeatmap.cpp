#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/types/UserData.h>
#include <sstream>

namespace zeno {
struct MakeHeatmap : zeno::INode {
    virtual void apply() override {
        HeatmapData heatmap = zeno::reflect::any_cast<HeatmapData>(ZImpl(get_param_result("heatmap")));
        ZImpl(set_primitive_output("heatmap", heatmap));
    }
};

ZENDEFNODE(MakeHeatmap, {
    {
        {gParamType_Heatmap, "heatmap", "", zeno::NoSocket, zeno::Heatmap},
        {gParamType_Int, "nres", "1024"}
    },
    {
        {gParamType_Heatmap, "heatmap"},
    },
    {}, 
    {"visualize"}
    }
);

struct HeatmapFromImage : zeno::INode {
    virtual void apply() override {
        auto image = ZImpl(get_input<zeno::PrimitiveObject>("image"));
        int w = image->userData()->get_int("w");
        auto heatmap = std::make_unique<HeatmapObject>();

        auto spos = ZImpl(get_input<NumericObject>("startPos"))->get<int>();
        auto epos = ZImpl(get_input<NumericObject>("endPos"))->get<int>();
        int start = 0;
        int end = w;
        if ( spos >= 0 && spos < epos && epos <= w)
        {
            start = spos;
            end = epos;
        }

        for (auto i = start; i < end; i++) {
            heatmap->colors.push_back(image->verts[i]);
        }
        ZImpl(set_output("heatmap", std::move(heatmap)));
    }
};

ZENDEFNODE(HeatmapFromImage,
{ /* inputs: */ {
    {gParamType_Primitive, "image", "", zeno::Socket_ReadOnly},
    {gParamType_Int, "startPos", "0"},
    {gParamType_Int, "endPos", "-1"},
}, /* outputs: */ {
    {"color", "heatmap"},
}, /* params: */ {
}, /* category: */ {
    "visualize",
}});

struct HeatmapFromImage2 : zeno::INode {
    virtual void apply() override {
        auto image = ZImpl(get_input<zeno::PrimitiveObject>("image"));
        int w = image->userData()->get_int("w");
        auto heatmap = std::make_unique<HeatmapObject>();

        auto spos = ZImpl(get_input2<float>("startPos"));
        auto epos = ZImpl(get_input2<float>("endPos"));
        int start = zeno::clamp(spos, 0.0f, 1.0f) * w;
        int end = zeno::clamp(epos, 0.0f, 1.0f) * w;
        std::vector<vec3f> temp;
        for (auto i = start; i < end; i++) {
            temp.push_back(image->verts[i]);
        }

        auto resample = ZImpl(get_input2<int>("resample"));
        if (0 < resample && resample < w) {
            for (auto i = 0; i < resample; i++) {
                float x = i / float(resample);
                x = zeno::clamp(x, 0, 1) * temp.size();
                int j = (int) zeno::floor(x);
                j = zeno::clamp(j, 0, temp.size() - 2);
                float f = x - j;
                auto c = (1 - f) * temp.at(j) + f * temp.at(j + 1);
                heatmap->colors.push_back(c);
            }
        }
        else {
            heatmap->colors = temp;
        }

        ZImpl(set_output("heatmap", std::move(heatmap)));
    }
};

ZENDEFNODE(HeatmapFromImage2,
           { /* inputs: */ {
                   {gParamType_Primitive, "image", "", zeno::Socket_ReadOnly},
                   {gParamType_Float, "startPos", "0"},
                   {gParamType_Float, "endPos", "1"},
                   {gParamType_Int, "resample", "0"},
               }, /* outputs: */ {
                   {gParamType_Heatmap, "heatmap"},
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});

struct HeatmapFromPrimAttr : zeno::INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        int attrNum = ZImpl(get_input2<int>("attrNum"));
        auto heatmap = std::make_unique<HeatmapObject>();
        auto attrName = ZImpl(get_input2<std::string>("attrName"));
        bool reverse = ZImpl(get_input2<bool>("reverse Result"));
        std::vector<vec3f> temp;
        for (auto i = 0; i < attrNum; i++) {
            temp.push_back(prim->attr<zeno::vec3f>(attrName)[i]);
        }
        auto resample = ZImpl(get_input2<int>("resample"));
        if (0 < resample && resample < attrNum) {
            for (auto i = 0; i < resample; i++) {
                float x = i / float(resample);
                x = zeno::clamp(x, 0, 1) * temp.size();
                int j = (int) zeno::floor(x);
                j = zeno::clamp(j, 0, temp.size() - 2);
                float f = x - j;
                auto c = (1 - f) * temp.at(j) + f * temp.at(j + 1);
                heatmap->colors.push_back(c);
            }
        }
        else {
            heatmap->colors = temp;
        }
        if (reverse) {
            std::reverse(heatmap->colors.begin(), heatmap->colors.end());
        }
        ZImpl(set_output("heatmap", std::move(heatmap)));
    }
};

ZENDEFNODE(HeatmapFromPrimAttr,
           { /* inputs: */ {
                   {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
                   {gParamType_String, "attrName", "clr"},
                   {gParamType_Int, "attrNum", "10"},
                   {gParamType_Int, "resample", "0"},
                   {gParamType_Bool, "reverse Result", "false"},
               }, /* outputs: */ {
                   {gParamType_Heatmap,"heatmap"},
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});

static zeno::vec3f interp(const std::vector<zeno::vec3f>& colors, float x) {
    if (x <= 0) return colors[0];
    if (x >= 1) return colors[colors.size() - 1];
    x = zeno::clamp(x, 0, 1) * (colors.size() - 1);
    int i = (int)zeno::floor(x);
    float f = x - i;
    return zeno::mix(colors[i], colors[i + 1], f);
}

struct PrimitiveColorByHeatmap : zeno::INode {
    virtual void apply() override {
        auto prim = clone_input_Geometry("prim");
        auto heatmap = zeno::reflect::any_cast<HeatmapData>(ZImpl(get_param_result("heatmap")));
        zeno::String attrName;
        if (ZImpl(has_input("attrName2"))) {
            attrName = get_input2_string("attrName2");
        } else {
            attrName = get_input2_string("attrName");
        }

        std::vector<zeno::vec3f> heatmap_clrs = heatmap.toVecColors(1024);

        float maxv = get_input2_float("max");
        float minv = get_input2_float("min");
        std::vector<zeno::vec3f> clr(prim->npoints());
        auto &src = prim->get_float_attr(ATTR_POINT, attrName);
        #pragma omp parallel for //ideally this could be done in opengl
        for (int i = 0; i < src.size(); i++) {
            auto x = (src[i]-minv)/(maxv-minv);
            // src[i] = (src[i]-minv)/(maxv-minv);
            clr[i] = interp(heatmap_clrs, x);
        }
        prim->set_point_attr("clr", clr);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveColorByHeatmap,
        { /* inputs: */ {
            {gParamType_Geometry, "prim", "", zeno::Socket_ReadOnly},
            {gParamType_String,"attrName2"},
            {gParamType_Heatmap, "heatmap", "", zeno::Socket_Primitve, zeno::Heatmap},
            {gParamType_Float, "min", "0"},
            {gParamType_Float, "max", "1"},
        }, 
        /* outputs: */ {
            {gParamType_Geometry, "prim"},
        }, /* params: */ {
            {gParamType_String, "attrName", "rho"},
        }, /* category: */ {
            "visualize",
        }});
        
struct PrimSample1D : zeno::INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto srcChannel = ZImpl(get_input2<std::string>("srcChannel"));
        auto dstChannel = ZImpl(get_input2<std::string>("dstChannel"));
        auto heatmap = ZImpl(get_input<HeatmapObject>("heatmap"));
        auto remapMin = ZImpl(get_input2<float>("remapMin"));
        auto remapMax = ZImpl(get_input2<float>("remapMax"));
        primSampleHeatmap(prim.get(), srcChannel, dstChannel, heatmap.get(), remapMin, remapMax);

        ZImpl(set_output("outPrim", std::move(prim)));
    }
};
ZENDEFNODE(PrimSample1D, {
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_Heatmap, "heatmap", "", zeno::Socket_Primitve},
        {gParamType_String, "srcChannel", "rho"},
        {gParamType_String, "dstChannel", "clr"},
        {gParamType_Float, "remapMin", "0"},
        {gParamType_Float, "remapMax", "1"},
    },
    {
        {gParamType_Primitive, "outPrim"}
    },
    {},
    {"primitive"},
});
void primSampleHeatmap(
        PrimitiveObject* prim,
        const std::string &srcChannel,
        const std::string &dstChannel,
        HeatmapObject* heatmap,
        float remapMin,
        float remapMax
) {
    auto &clr = prim->add_attr<zeno::vec3f>(dstChannel);
    auto &src = prim->attr<float>(srcChannel);
#pragma omp parallel for //ideally this could be done in opengl
    for (int i = 0; i < src.size(); i++) {
        auto x = (src[i]-remapMin)/(remapMax-remapMin);
        clr[i] = heatmap->interp(x);
    }
}
}
