#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/types/UserData.h>
#include <sstream>

namespace zeno {
struct MakeHeatmap : zeno::INode {
    virtual void apply() override {
        //auto nres = get_param<int>("nres");
        //auto ramps = get_param<std::string>("_RAMPS");
        //std::stringstream ss(ramps);
        //std::vector<std::pair<float, zeno::vec3f>> colors;
        //int count;
        //ss >> count;
        //for (int i = 0; i < count; i++) {
        //    float f = 0.f, x = 0.f, y = 0.f, z = 0.f;
        //    ss >> f >> x >> y >> z;
        //    //printf("%f %f %f %f\n", f, x, y, z);
        //    colors.emplace_back(
        //            f, zeno::vec3f(x, y, z));
        //}

        //auto heatmap = std::make_shared<HeatmapObject>();
        //for (int i = 0; i < nres; i++) {
        //    float fac = i * (1.f / nres);
        //    zeno::vec3f clr;
        //    for (int j = 0; j < colors.size(); j++) {
        //        auto [f, rgb] = colors[j];
        //        if (f >= fac) {
        //            if (j != 0) {
        //                auto [last_f, last_rgb] = colors[j - 1];
        //                auto intfac = (fac - last_f) / (f - last_f);
        //                //printf("%f %f %f %f\n", fac, last_f, f, intfac);
        //                clr = zeno::mix(last_rgb, rgb, intfac);
        //            } else {
        //                clr = rgb;
        //            }
        //            break;
        //        }
        //    }
        //    heatmap->colors.push_back(clr);
        //}
        auto heatmap = get_input2<HeatmapObject>("heatmap");
        set_output("heatmap", std::move(heatmap));
    }
};

ZENDEFNODE(MakeHeatmap,
        { /* inputs: */ {{gParamType_Heatmap, "heatmap", "", zeno::NoSocket, zeno::Heatmap},
        }, /* outputs: */ {
            {gParamType_Heatmap, "heatmap"},
        }, /* params: */ {
        //{gParamType_Int, "nres", "1024"},
        //{gParamType_String, "_RAMPS", "0 0 0.8 0.8 0.8 1"},
        }, /* category: */ {
        "visualize",
        }});

struct HeatmapFromImage : zeno::INode {
    virtual void apply() override {
        auto image = get_input<zeno::PrimitiveObject>("image");
        int w = image->userData().get2<int>("w");
        auto heatmap = std::make_shared<HeatmapObject>();

        auto spos = get_input<NumericObject>("startPos")->get<int>();
        auto epos = get_input<NumericObject>("endPos")->get<int>();
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
        set_output("heatmap", std::move(heatmap));
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
        auto image = get_input<zeno::PrimitiveObject>("image");
        int w = image->userData().get2<int>("w");
        auto heatmap = std::make_shared<HeatmapObject>();

        auto spos = get_input2<float>("startPos");
        auto epos = get_input2<float>("endPos");
        int start = zeno::clamp(spos, 0.0f, 1.0f) * w;
        int end = zeno::clamp(epos, 0.0f, 1.0f) * w;
        std::vector<vec3f> temp;
        for (auto i = start; i < end; i++) {
            temp.push_back(image->verts[i]);
        }

        auto resample = get_input2<int>("resample");
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

        set_output("heatmap", std::move(heatmap));
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
        auto prim = get_input<PrimitiveObject>("prim");
        int attrNum = get_input2<int>("attrNum");
        auto heatmap = std::make_shared<HeatmapObject>();
        auto attrName = get_input2<std::string>("attrName");
        bool reverse = get_input2<bool>("reverse Result");
        std::vector<vec3f> temp;
        for (auto i = 0; i < attrNum; i++) {
            temp.push_back(prim->attr<vec3f>(attrName)[i]);
        }
        auto resample = get_input2<int>("resample");
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
        set_output("heatmap", std::move(heatmap));
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

struct PrimitiveColorByHeatmap : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto heatmap = get_input<HeatmapObject>("heatmap");
        std::string attrName;
        if (has_input("attrName2")) {
            attrName = get_input2<std::string>("attrName2");
        } else {
            attrName = get_param<std::string>("attrName");
        }

        float maxv = 1.0f;
        float minv = 0.0f;
        if(has_input("max"))
            maxv = get_input<NumericObject>("max")->get<float>();
        if(has_input("min"))
            minv = get_input<NumericObject>("min")->get<float>();
        auto &clr = prim->add_attr<zeno::vec3f>("clr");
        auto &src = prim->attr<float>(attrName);
        #pragma omp parallel for //ideally this could be done in opengl
        for (int i = 0; i < src.size(); i++) {
            auto x = (src[i]-minv)/(maxv-minv);
            // src[i] = (src[i]-minv)/(maxv-minv);
            clr[i] = heatmap->interp(x);
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveColorByHeatmap,
        { /* inputs: */ {
            {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
            {gParamType_String,"attrName2"},
            {gParamType_Heatmap, "heatmap", "", zeno::Socket_Primitve, zeno::Heatmap},
            {gParamType_Float, "min", "0"},
            {gParamType_Float, "max", "1"},
        }, /* outputs: */ {
{gParamType_Primitive, "prim"},
}, /* params: */ {
            {gParamType_String, "attrName", "rho"},
        }, /* category: */ {
            "visualize",
        }});
        
struct PrimSample1D : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto srcChannel = get_input2<std::string>("srcChannel");
        auto dstChannel = get_input2<std::string>("dstChannel");
        auto heatmap = get_input<HeatmapObject>("heatmap");
        auto remapMin = get_input2<float>("remapMin");
        auto remapMax = get_input2<float>("remapMax");
        primSampleHeatmap(prim, srcChannel, dstChannel, heatmap, remapMin, remapMax);

        set_output("outPrim", std::move(prim));
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
        std::shared_ptr<PrimitiveObject> prim,
        const std::string &srcChannel,
        const std::string &dstChannel,
        std::shared_ptr<HeatmapObject> heatmap,
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
