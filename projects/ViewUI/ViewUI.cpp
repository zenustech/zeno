#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/pybjson.h>

namespace {
struct MakeViewCompnent : zeno::INode {
    virtual void apply() override {
        auto component = get_param<std::string>("component");
        auto name = get_param<std::string>("name");
        auto tips = get_param<std::string>("tips");
        auto id = get_param<std::string>("id");
        auto length = get_param<int>("length");
        auto width = get_param<int>("width");
        auto layoutIndex = get_param<int>("layoutIndex");
        auto maxValue = get_param<float>("maxValue");
        auto minValue = get_param<float>("minValue");

        /*Value uni(kObjectType);
        uni.AddMember("name", name, allocator);
        uni.AddMember("tips", tips, allocator);
        uni.AddMember("id", id, allocator);
        uni.AddMember("length", length, allocator);
        uni.AddMember("width", width, allocator);
        uni.AddMember("layoutIndex", layoutIndex, allocator);
        uni.AddMember("maxValue", maxValue, allocator);
        uni.AddMember("minValue", minValue, allocator);*/

        rapidjson::Document document;
        document.SetObject();
        rapidjson::Document::AllocatorType &allocator = document.GetAllocator();
        rapidjson::Value component_r(rapidjson::kStringType);
        component_r.SetString(component.data(), component.size());
        document.AddMember("component", component_r, allocator);
        rapidjson::Value name_r(rapidjson::kStringType);
        name_r.SetString(name.data(), name.size());
        document.AddMember("name", name_r, allocator);
        rapidjson::Value tips_r(rapidjson::kStringType);
        tips_r.SetString(tips.data(), tips.size());
        document.AddMember("tips", tips_r, allocator);
        rapidjson::Value id_r(rapidjson::kStringType);
        id_r.SetString(id.data(), id.size());
        document.AddMember("id", id_r, allocator);
        document.AddMember("length", length, allocator);
        document.AddMember("width", width, allocator);
        document.AddMember("layoutIndex", layoutIndex, allocator);
        document.AddMember("maxValue", maxValue, allocator);
        document.AddMember("minValue", minValue, allocator);

        rapidjson::StringBuffer strBuffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer{strBuffer};
        document.Accept(writer);

        zeno::log_info("{} json:{}", component, strBuffer.GetString());

        std::string out = strBuffer.GetString();
        set_output2("ui", out);
    }
};

ZENDEFNODE(MakeViewCompnent, 
 {/*输入*/
    {},
    /*输出*/
    {"ui"},
    /*参数*/
    {
        {"enum Button Slide", "component", "Button"},
        {"string", "name", ""},
        {"string", "tips", ""},
        {"string", "id", ""},
        {"int", "length", "100"},
        {"int", "width", "20"},
        {"int", "layoutIndex", "0"},
                                  {"float", "maxValue", "100.0"},
                                  {"float", "minValue", "0"},
    },
    /*类别*/
    {"ViewUI"}
});

struct MakeViewUILayout : zeno::INode {    
    virtual void apply() override {
        /*{
            "component" : component 
            "margin" : margin,
            "layoutSpacing" : layoutSpacing,
            "spacing" : spacing,
            "spacingIndex" : spacingIndex,
            "stretch" : stretch,
            "stretchIndex" : stretchIndex,
            [ 
                { "component" : component,
                  "tips": tips,
                  "id": id,
                  "length": length,
                  "width": width,
                  "layoutIndex": layoutIndex,
                  "maxValue": maxValue,
                  "minValue": minValue,
                }
            ]
        }*/
        auto component = get_param<std::string>("component");
        auto margin = get_param<int>("margin");
        auto layoutSpacing = get_param<int>("layoutSpacing");
        auto spacing = get_param<int>("spacing");
        auto spacingIndex = get_param<int>("spacingIndex");
        auto stretch = get_param<int>("stretch");
        auto stretchIndex = get_param<int>("stretchIndex");

        rapidjson::Document document;
        document.SetObject();
        rapidjson::Value component_r(rapidjson::kStringType);
        component_r.SetString(component.data(), component.size());
        document.AddMember("component", component_r, document.GetAllocator());
        document.AddMember("margin", margin, document.GetAllocator());
        document.AddMember("layoutSpacing", layoutSpacing, document.GetAllocator());
        document.AddMember("spacing", spacing, document.GetAllocator());
        document.AddMember("spacingIndex", spacingIndex, document.GetAllocator());
        document.AddMember("stretch", stretch, document.GetAllocator());
        document.AddMember("stretchIndex", stretchIndex, document.GetAllocator());

        auto list = get_input<zeno::ListObject>("uiList").get();
        zeno::log_info("UI List size {}", int(list->size()));
        rapidjson::Value arr(rapidjson::kArrayType);
        for (int i = 0; i < list->size(); i++)
        {
            std::string str = ((zeno::StringObject *)(list->arr[i].get()))->get();
            //zeno::log_info("Subjson:{}", str);
            rapidjson::Value str_r(rapidjson::kStringType);
            str_r.SetString(str.data(), str.size(), document.GetAllocator());

            arr.PushBack(str_r, document.GetAllocator());
        }        
        document.AddMember("uiList", arr, document.GetAllocator());

        rapidjson::StringBuffer strBuffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(strBuffer);
        document.Accept(writer);
        zeno::log_info("Subjson json:{}", strBuffer.GetString());

        std::string out = strBuffer.GetString();
        set_output2("ui", out);
    }
};

ZENDEFNODE(MakeViewUILayout, 
{
    /*输入*/
    {
        "uiList"
    },
    /*输出*/
    {   
        "ui"
    },
    /*参数*/
    {
        {"enum HLayout VLayout","component", "HLayout"},
                                  {"int", "margin", "10"},
                                  {"int", "layoutSpacing", "10"},
                                  {"int", "spacing", "0"},
                                  {"int", "spacingIndex", "0"},
                                  {"int", "stretch", "0"},
                                  {"int", "stretchIndex", "0"},
    },
    /*类别*/
    {"ViewUI"}
});

}
