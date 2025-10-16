//
// Created by zh on 2023/11/14.
//

#include "zeno/types/DictObject.h"
#include "zeno/types/ListObject_impl.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/fileio.h"
#include "zeno/utils/log.h"
#include "zeno/utils/string.h"
#include "zeno/utils/scope_exit.h"
#include "zeno/utils/inputoutput_wrapper.h"
#include <sstream>
#include <string>
#include <tinygltf/json.hpp>
#include <zeno/zeno.h>
#include <filesystem>
#include "uuid_v4.h"
namespace fs = std::filesystem;

#ifdef ZENO_WITH_PYTHON
    #include <Python.h>
#endif
using Json = nlohmann::json;

namespace zeno {
struct JsonObject : IObjectClone<JsonObject> {
    Json json;
};

struct ReadJson : zeno::INode {
    virtual void apply() override {
        auto json = std::make_unique<JsonObject>();
        auto path = ZImpl(get_input2<std::string>("path"));
        std::string native_path = std::filesystem::u8path(path).string();
        auto content = zeno::file_get_content(native_path);
        json->json = Json::parse(content);
        set_output("json", std::move(json));
    }
};
ZENDEFNODE(ReadJson, {
    {
        {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit}
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});

struct WriteJson : zeno::INode {
    virtual void apply() override {
        auto _json = ZImpl(get_input2<JsonObject>("json"));
        auto path = ZImpl(get_input2<std::string>("path"));
        path = create_directories_when_write_file(path);
        file_put_content(path, _json->json.dump());
    }
};
ZENDEFNODE(WriteJson, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::WritePathEdit},
    },
    {
    },
    {},
    {
        "json"
    },
});
static Json iobject_to_json(const zany& iObject) {
    Json json;
    if (objectIsRawLiterial<int>(iObject.get())) {
        json = objectToLiterial<int>(iObject);
    }
    else if (objectIsRawLiterial<vec2i>(iObject.get())) {
        auto value = objectToLiterial<vec2i>(iObject);
        json = { value[0], value[1]};
    }
    else if (objectIsRawLiterial<vec3i>(iObject.get())) {
        auto value = objectToLiterial<vec3i>(iObject);
        json = { value[0], value[1], value[2]};
    }
    else if (objectIsRawLiterial<vec4i>(iObject.get())) {
        auto value = objectToLiterial<vec4i>(iObject);
        json = { value[0], value[1], value[2], value[3]};
    }
    else if (objectIsRawLiterial<float>(iObject.get())) {
        json = objectToLiterial<float>(iObject);
    }
    else if (objectIsRawLiterial<vec2f>(iObject.get())) {
        auto value = objectToLiterial<vec2f>(iObject);
        json = { value[0], value[1]};
    }
    else if (objectIsRawLiterial<vec3f>(iObject.get())) {
        auto value = objectToLiterial<vec3f>(iObject);
        json = { value[0], value[1], value[2]};
    }
    else if (objectIsRawLiterial<vec4f>(iObject.get())) {
        auto value = objectToLiterial<vec4f>(iObject);
        json = { value[0], value[1], value[2], value[3]};
    }
    else if (objectIsRawLiterial<std::string>(iObject.get())) {
        json = objectToLiterial<std::string>(iObject);
    }
    else if (auto list = dynamic_cast<ListObject*>(iObject.get())) {
        for (auto iObj: list->get()) {
            json.push_back(iobject_to_json(iObj->clone()));
        }
    }
    else if (auto dict = dynamic_cast<DictObject*>(iObject.get())) {
        for (const auto& [key, iObj]: dict->lut) {
            json[key] = iobject_to_json(iObj);
        }
    }
    else if (auto sub_json = dynamic_cast<JsonObject*>(iObject.get())) {
        json = sub_json->json;
    }
    return std::move(json);
}
struct FormJson : zeno::INode {
  virtual void apply() override {
      auto _json = std::make_unique<JsonObject>();
      auto iObject = clone_input("iObject");
      _json->json = iobject_to_json(iObject);
      set_output("json", std::move(_json));
  }
};
ZENDEFNODE(FormJson, {
     {
         {gParamType_IObject, "iObject"},
     },
     {
         {gParamType_JsonObject, "json"},
     },
     {},
     {
         "json"
     },
 });

struct PrimUserDataToJson : zeno::INode {
    void iobject_to_json(Json &json, std::string key, const zany& iObject) {
        if (objectIsRawLiterial<int>(iObject.get())) {
            json[key] = objectToLiterial<int>(iObject);
        }
        else if (objectIsRawLiterial<vec2i>(iObject.get())) {
            auto value = objectToLiterial<vec2i>(iObject);
            json[key] = { value[0], value[1]};
        }
        else if (objectIsRawLiterial<vec3i>(iObject.get())) {
            auto value = objectToLiterial<vec3i>(iObject);
            json[key] = { value[0], value[1], value[2]};
        }
        else if (objectIsRawLiterial<vec4i>(iObject.get())) {
            auto value = objectToLiterial<vec4i>(iObject);
            json[key] = { value[0], value[1], value[2], value[3]};
        }
        else if (objectIsRawLiterial<float>(iObject.get())) {
            json[key] = objectToLiterial<float>(iObject);
        }
        else if (objectIsRawLiterial<vec2f>(iObject.get())) {
            auto value = objectToLiterial<vec2f>(iObject);
            json[key] = { value[0], value[1]};
        }
        else if (objectIsRawLiterial<vec3f>(iObject.get())) {
            auto value = objectToLiterial<vec3f>(iObject);
            json[key] = { value[0], value[1], value[2]};
        }
        else if (objectIsRawLiterial<vec4f>(iObject.get())) {
            auto value = objectToLiterial<vec4f>(iObject);
            json[key] = { value[0], value[1], value[2], value[3]};
        }
        else if (objectIsRawLiterial<std::string>(iObject.get())) {
            json[key] = objectToLiterial<std::string>(iObject);
        }
    }
    void apply() override {
        auto keys_string = ZImpl(get_input2<std::string>("keys"));
        auto output_all = ZImpl(get_input2<bool>("output_all"));
        auto _json = std::make_unique<JsonObject>();
        auto iObject = get_input("iObject");
        auto ud = dynamic_cast<UserData*>(iObject->userData());

        std::vector<std::string> keys = zeno::split_str(keys_string, {' ', '\n'});
        std::set<std::string> keys_set(keys.begin(), keys.end());

        for (auto i = ud->begin(); i != ud->end(); i++) {
            if (output_all == false && keys_set.count(i->first) == 0) {
                continue;
            }
            iobject_to_json(_json->json, i->first, i->second);
        }

        set_output("json", std::move(_json));
    }
};
ZENDEFNODE(PrimUserDataToJson, {
     {
         {gParamType_IObject, "iObject"},
         {gParamType_Bool, "output_all", "0"},
         ParamPrimitive("keys", gParamType_String, "abc_path\n_pivot\n_rotate\n_scale\n_translate\n_transform_row0\n_transform_row1\n_transform_row2\n_transform_row3", Multiline),
     },
     {
         {gParamType_JsonObject, "json"},
     },
     {},
     {
         "json"
     },
 });
struct JsonToString : zeno::INode {
  virtual void apply() override {
    auto json = ZImpl(get_input<JsonObject>("json"));
    set_output_string("out", stdString2zs(json->json.dump()));
  }
};
ZENDEFNODE(JsonToString, {
     {
         {gParamType_JsonObject, "json"},
     },
     {
         ParamPrimitive("out", gParamType_String)
     },
     {},
     {
         "json"
     },
 });
struct JsonSetDataSimple : zeno::INode {
    virtual void apply() override {
        auto in_json = std::make_unique<JsonObject>();
        if (ZImpl(has_input<JsonObject>("json"))) {
            in_json = ZImpl(get_input<JsonObject>("json"));
        }

        auto path = ZImpl(get_input2<std::string>("path"));
        auto names = split_str(path, '/');
        if (!names.empty()) {
            if (names.begin()->empty()) {
                names.erase(names.begin());
            }
        }
        Json *tmp_json = &in_json->json;
        for (auto & name : names) {
            if (tmp_json->is_array()) {
                tmp_json = &tmp_json->operator[](std::stoi(name));
            }
            else {
                tmp_json = &tmp_json->operator[](name);
            }
        }
        auto value = ZImpl(clone_input("value"));
        *tmp_json = iobject_to_json(value);

        ZImpl(set_output2("json", std::move(in_json)));
    }
};

ZENDEFNODE(JsonSetDataSimple, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "path"},
        {gParamType_IObject, "value"},
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});


struct JsonSetData : zeno::INode {
    virtual void apply() override {
        auto in_json = std::make_unique<JsonObject>();
        if (ZImpl(has_input<JsonObject>("json"))) {
            in_json = ZImpl(get_input<JsonObject>("json"));
        }
        auto multi_path = ZImpl(get_input2<std::string>("paths"));
        std::istringstream iss(multi_path);
        std::vector<std::string> paths;
        std::string line;
        while (std::getline(iss, line)) {
            line = zeno::trim_string(line);
            if (line.size()) {
                paths.push_back(line);
            }
        }
        auto dict = ZImpl(get_input<DictObject>("dict"));
        for (auto &path: paths) {
            auto strings = zeno::split_str(path, ':');
            auto names = split_str(strings[1], '/');
            if (!names.empty()) {
                if (names.begin()->empty()) {
                    names.erase(names.begin());
                }
            }

            Json *tmp_json = &in_json->json;
            for (auto & name : names) {
                if (tmp_json->is_array()) {
                    tmp_json = &tmp_json->operator[](std::stoi(name));
                }
                else {
                    tmp_json = &tmp_json->operator[](name);
                }
            }
            std::string new_name = zeno::trim_string(strings[0]);
            *tmp_json = iobject_to_json(dict->lut[new_name]);
        }

        ZImpl(set_output("json", std::move(in_json)));
    }
};

ZENDEFNODE(JsonSetData, {
    {
        {gParamType_JsonObject, "json"},
        ParamPrimitive("paths", gParamType_String, "input_name:json_path", zeno::Multiline),
        {gParamType_Dict, "dict"}
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});

struct ReadJsonFromString : zeno::INode {
    virtual void apply() override {
        auto json = std::make_unique<JsonObject>();
        auto content = ZImpl(get_input2<std::string>("content"));
        json->json = Json::parse(content);
        set_output("json", std::move(json));
    }
};
ZENDEFNODE(ReadJsonFromString, {
    {
        {gParamType_String, "content"},
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});
struct JsonGetArraySize : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        ZImpl(set_output("size", std::make_unique<NumericObject>((int)json->json.size())));
    }
};
ZENDEFNODE(JsonGetArraySize, {
    {
        {gParamType_JsonObject, "json"},
    },
    {
        {gParamType_Int, "size"},
    },
    {},
    {
        "json"
    },
});
struct JsonGetArrayItem : zeno::INode {
    virtual void apply() override {
        auto out_json = std::make_unique<JsonObject>();
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto index = ZImpl(get_input2<int>("index"));
        out_json->json = json->json[index];
        ZImpl(set_output("json", std::move(out_json)));
    }
};
ZENDEFNODE(JsonGetArrayItem, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_Int, "index"}
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});

struct JsonGetChild : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto name = ZImpl(get_input2<std::string>("name"));
        auto type = ZImpl(get_input2<std::string>("type"));
        if (type == "json") {
            auto output = std::make_unique<JsonObject>();
            if (json->json.contains(name)) {
                output->json = json->json[name];
            }
            ZImpl(set_output("out", std::move(output)));
        }
        else if (type == "int") {
            auto output = std::make_unique<NumericObject>();
            if (json->json.contains(name)) {
                output->set<int>(int(json->json[name]));
            }
            ZImpl(set_output("out", std::move(output)));
        }
        else if (type == "float") {
            auto output = std::make_unique<NumericObject>();
            if (json->json.contains(name)) {
                output->set<float>(float(json->json[name]));
            }
            ZImpl(set_output("out", std::move(output)));
        }
        else if (type == "string") {
            auto output = std::make_unique<StringObject>();
            if (json->json.contains(name)) {
                output->set(std::string(json->json[name]));
            }
            ZImpl(set_output("out", std::move(output)));
        }
        else if (type == "vec2f") {
            auto output = std::make_unique<NumericObject>();
            if (json->json.contains(name)) {
                vec2f vec;
                vec[0] = float(json->json[name][0]);
                vec[1] = float(json->json[name][1]);
                output->set<vec2f>(vec);
            }
            ZImpl(set_output("out", std::move(output)));
        }
        else if (type == "vec3f") {
            auto output = std::make_unique<NumericObject>();
            if (json->json.contains(name)) {
                vec3f vec;
                vec[0] = float(json->json[name][0]);
                vec[1] = float(json->json[name][1]);
                vec[2] = float(json->json[name][2]);
                output->set<vec3f>(vec);
            }
            ZImpl(set_output("out", std::move(output)));
        }
        else if (type == "vec4f") {
            auto output = std::make_unique<NumericObject>();
            if (json->json.contains(name)) {
                vec4f vec;
                vec[0] = float(json->json[name][0]);
                vec[1] = float(json->json[name][1]);
                vec[2] = float(json->json[name][2]);
                vec[3] = float(json->json[name][3]);
                output->set<vec4f>(vec);
            }
            ZImpl(set_output("out", std::move(output)));
        }
    }
};
ZENDEFNODE(JsonGetChild, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "name"},
        ParamPrimitive("type", gParamType_String, "int", zeno::Combobox, std::vector<std::string>{"json", "int", "float", "string", "vec2f", "vec3f", "vec4f"}),
        //{"enum json int float string vec2f vec3f vec4f", "type", "json"},
    },
    {
        {gParamType_IObject, "out"},
    },
    {},
    {
        "deprecated"
    },
});

struct JsonHasKey : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto name = ZImpl(get_input2<std::string>("name"));
        ZImpl(set_output2("out", int(json->json.contains(name))));
    }
};
ZENDEFNODE(JsonHasKey, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "name"},
    },
    {
        {gParamType_Bool, "out"},
    },
    {},
    {
        "json"
    },
});
struct JsonGetInt : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        ZImpl(set_output2("value", int(json->json)));
    }
};
ZENDEFNODE(JsonGetInt, {
    {
        {gParamType_JsonObject, "json"},
    },
    {
        {gParamType_Int, "value"},
    },
    {},
    {
        "deprecated"
    },
});

struct JsonGetFloat : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        ZImpl(set_output2("value", float(json->json)));
    }
};
ZENDEFNODE(JsonGetFloat, {
    {
        {gParamType_JsonObject, "json"},
    },
    {
        {gParamType_Float, "value"},
    },
    {},
    {
        "deprecated"
    },
});

struct JsonGetString : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        set_output_string("string", stdString2zs(std::string(json->json)));
    }
};
ZENDEFNODE(JsonGetString, {
    {
        {gParamType_JsonObject, "json"},
    },
    {
        {gParamType_String, "string"},
    },
    {},
    {
        "deprecated"
    },
});

struct JsonGetKeys : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto list = std::make_unique<ListObject>();
        for (auto& [key, _] : json->json.items()) {
            list->push_back(std::make_unique<zeno::StringObject>(key));
        }
        ZImpl(set_output2("keys", std::move(list)));
    }
};
ZENDEFNODE(JsonGetKeys, {
    {
        {gParamType_JsonObject, "json"},
    },
    {
        {gParamType_List, "keys"},
    },
    {},
    {
        "json"
    },
});
struct JsonGetTypeName : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        set_output_string("string", stdString2zs(std::string(json->json.type_name())));
    }
};
ZENDEFNODE(JsonGetTypeName, {
    {
        {gParamType_JsonObject, "json"},
    },
    {
        {gParamType_String, "string"},
    },
    {},
    {
        "deprecated"
    },
});

#if 0
struct JsonData : zeno::INode {
    virtual void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto path = ZImpl(get_input2<std::string>("path"));
        auto strings = zeno::split_str(path, ':');
        auto type = strings[1];
        path = strings[0];
        auto names = split_str(path, '/');

        for (auto & name : names) {
            json->json = json->json[name];
        }


        if (type == "json") {
            auto out_json = std::make_unique<JsonObject>();
            out_json->json = json->json;
            set_output("out", std::move(out_json));
        }
        else if (type == "int") {
            set_output2("out", int(json->json));
        }
        else if (type == "float") {
            set_output2("out", float(json->json));
        }
        else if (type == "string") {
            set_output2("out", std::string(json->json));
        }
        else if (type == "vec2f") {
            float x = float(json->json["x"]);
            float y = float(json->json["y"]);
            set_output2("out", vec2f(x, y));
        }
        else if (type == "vec3f") {
            float x = float(json->json["x"]);
            float y = float(json->json["y"]);
            float z = float(json->json["z"]);
            set_output2("out", vec3f(x, y, z));
        }
        else if (type == "vec4f") {
            float x = float(json->json["x"]);
            float y = float(json->json["y"]);
            float z = float(json->json["z"]);
            float w = float(json->json["w"]);
            set_output2("out", vec4f(x, y, z, w));
        }
        else if (type == "vec2i") {
            auto x = int(json->json["x"]);
            auto y = int(json->json["y"]);
            set_output2("out", vec2i(x, y));
        }
        else if (type == "vec3i") {
            auto x = int(json->json["x"]);
            auto y = int(json->json["y"]);
            auto z = int(json->json["z"]);
            set_output2("out", vec3i(x, y, z));
        }
        else if (type == "vec4i") {
            auto x = int(json->json["x"]);
            auto y = int(json->json["y"]);
            auto z = int(json->json["z"]);
            auto w = int(json->json["w"]);
            set_output2("out", vec4i(x, y, z, w));
        }
    }
};
ZENDEFNODE(JsonData, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "path"},
    },
    {
        "out",
    },
    {},
    {
        "deprecated"
    },
});
#endif

struct JsonGetData : zeno::INode {
    virtual void apply() override {
        auto in_json = ZImpl(get_input<JsonObject>("json"));
        auto multi_path = ZImpl(get_input2<std::string>("paths"));
        std::istringstream iss(multi_path);
        std::vector<std::string> paths;
        std::string line;
        while (std::getline(iss, line)) {
            line = zeno::trim_string(line);
            if (line.size()) {
                paths.push_back(line);
            }
        }

        auto dict = std::make_unique<zeno::DictObject>();
        for (auto &path: paths) {
            auto json = std::make_unique<JsonObject>();
            json->json = in_json->json;
            auto strings = zeno::split_str(path, ':');
            auto type = strings[1];
            path = strings[0];
            std::string new_name = path;
            if (strings.size() == 3) {
                new_name = zeno::trim_string(strings[2]);
            }

            auto names = split_str(path, '/');
            bool missing = false;

            for (auto & name : names) {
                if (json->json.contains(name) == false) {
                    missing = true;
                    break;
                }
                if (json->json.is_array()) {
                    json->json = json->json[std::stoi(name)];
                }
                else {
                    json->json = json->json[name];
                }
            }
            if (missing) {
                if (ZImpl(get_input2<bool>("useDefaultValueWhenMissing"))) {
                    if (type == "json") {
                        dict->lut[new_name] = std::make_unique<JsonObject>();
                    }
                    else if (type == "int") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(int{});
                    }
                    else if (type == "float") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(float{});
                    }
                    else if (type == "string") {
                        dict->lut[new_name] = std::make_unique<StringObject>(std::string());
                    }
                    else if (type == "vec2f") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(vec2f{});
                    }
                    else if (type == "vec3f") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(vec3f{});
                    }
                    else if (type == "vec4f") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(vec4f{});
                    }
                    else if (type == "vec2i") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(vec2i{});
                    }
                    else if (type == "vec3i") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(vec3i{});
                    }
                    else if (type == "vec4i") {
                        dict->lut[new_name] = std::make_unique<NumericObject>(vec4i{});
                    }
                    continue;
                }
            }

            if (type == "json") {
                auto out_json = std::make_unique<JsonObject>();
                out_json->json = json->json;
                dict->lut[new_name] = std::move(out_json);
            }
            else if (type == "int") {
                dict->lut[new_name] = std::make_unique<NumericObject>(int(json->json));
            }
            else if (type == "float") {
                dict->lut[new_name] = std::make_unique<NumericObject>(float(json->json));
            }
            else if (type == "string") {
                dict->lut[new_name] = std::make_unique<StringObject>(std::string(json->json));
            }
            else if (type == "vec2f") {
                if (json->json.is_array()) {
                    float x = float(json->json[0]);
                    float y = float(json->json[1]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec2f(x, y));
                }
                else {
                    float x = float(json->json["x"]);
                    float y = float(json->json["y"]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec2f(x, y));
                }
            }
            else if (type == "vec3f") {
                if (json->json.is_array()) {
                    float x = float(json->json[0]);
                    float y = float(json->json[1]);
                    float z = float(json->json[2]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec3f(x, y, z));
                }
                else {
                    float x = float(json->json["x"]);
                    float y = float(json->json["y"]);
                    float z = float(json->json["z"]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec3f(x, y, z));
                }
            }
            else if (type == "vec4f") {
                if (json->json.is_array()) {
                    float x = float(json->json[0]);
                    float y = float(json->json[1]);
                    float z = float(json->json[2]);
                    float w = float(json->json[3]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec4f(x, y, z, w));
                }
                else {
                    float x = float(json->json["x"]);
                    float y = float(json->json["y"]);
                    float z = float(json->json["z"]);
                    float w = float(json->json["w"]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec4f(x, y, z, w));
                }
            }
            else if (type == "vec2i") {
                if (json->json.is_array()) {
                    auto x = int(json->json[0]);
                    auto y = int(json->json[1]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec2i(x, y));
                }
                else {
                    auto x = int(json->json["x"]);
                    auto y = int(json->json["y"]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec2i(x, y));
                }
            }
            else if (type == "vec3i") {
                if (json->json.is_array()) {
                    auto x = int(json->json[0]);
                    auto y = int(json->json[1]);
                    auto z = int(json->json[2]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec3i(x, y, z));
                }
                else {
                    auto x = int(json->json["x"]);
                    auto y = int(json->json["y"]);
                    auto z = int(json->json["z"]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec3i(x, y, z));
                }
            }
            else if (type == "vec4i") {
                if (json->json.is_array()) {
                    auto x = int(json->json[0]);
                    auto y = int(json->json[1]);
                    auto z = int(json->json[2]);
                    auto w = int(json->json[3]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec4i(x, y, z, w));
                }
                else {
                    auto x = int(json->json["x"]);
                    auto y = int(json->json["y"]);
                    auto z = int(json->json["z"]);
                    auto w = int(json->json["w"]);
                    dict->lut[new_name] = std::make_unique<NumericObject>(vec4i(x, y, z, w));
                }
            }
        }
        set_output("outs", std::move(dict));
    }
};
ZENDEFNODE(JsonGetData, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_Bool, "useDefaultValueWhenMissing", "0"},
        {gParamType_String, "paths", "json_path:vec3f:output_name", Socket_Primitve, zeno::Multiline}
    },
    {
        {gParamType_Dict, "outs"}
    },
    {},
    {
        "json"
    },
});

struct CreateJson : zeno::INode {
  virtual void apply() override {
      auto _json = std::make_unique<JsonObject>();
      set_output("json", std::move(_json));
  }
};
ZENDEFNODE(CreateJson, {
     {},
     {
         {gParamType_JsonObject, "json"},
     },
     {},
     {
         "json"
     },
 });
void access(Json &json, std::vector<std::string> &names, int index) {
        auto name = names[index];
        if (index == names.size() - 1) {
            json.erase(name);
            return;
        }
        if (json.is_array()) {
            access(json[std::stoi(name)], names, index + 1);
        }
        else {
            access(json[name], names, index + 1);
        }
    }



struct JsonErase : zeno::INode {

    void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto path = ZImpl(get_input2<std::string>("path"));
        auto names = split_str(path, '/');

        access(json->json, names, 0);

        set_output("json", std::move(json));
    }
};
ZENDEFNODE(JsonErase, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "path", "a/0/b"}
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});

struct JsonRenameKey : zeno::INode {
    void access(Json &json, std::vector<std::string> &names, int index, std::string &new_name) {
        auto name = names[index];
        if (index == names.size() - 1) {
            Json node = json[name];
            json.erase(name);
            json[new_name] = node;
            return;
        }
        if (json.is_array()) {
            access(json[std::stoi(name)], names, index + 1, new_name);
        }
        else {
            access(json[name], names, index + 1, new_name);
        }
    }
    void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto path = ZImpl(get_input2<std::string>("path"));
        auto new_name = ZImpl(get_input2<std::string>("new_name"));
        auto names = split_str(path, '/');

        access(json->json, names, 0, new_name);

        set_output("json", std::move(json));
    }
};
ZENDEFNODE(JsonRenameKey, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "path", "a/0/b"},
        {gParamType_String, "new_name", "new_name"},
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});

struct JsonInsertValue : zeno::INode {
    void access(Json &json, std::vector<std::string> &names, int index, const zany& iObject) {
        auto name = names[index];
        if (index == names.size() - 1) {
            json[name] = iobject_to_json(iObject);
            return;
        }
        if (json.is_array()) {
            access(json[std::stoi(name)], names, index + 1, iObject);
        }
        else {
            access(json[name], names, index + 1, iObject);
        }
    }
    void apply() override {
        auto json = ZImpl(get_input<JsonObject>("json"));
        auto path = ZImpl(get_input2<std::string>("path"));
        auto iObject = clone_input("iObject");
        auto names = split_str(path, '/');

        access(json->json, names, 0, iObject);

        set_output("json", std::move(json));
    }
};
ZENDEFNODE(JsonInsertValue, {
    {
        {gParamType_JsonObject, "json"},
        {gParamType_String, "path", "a/0/b"},
        {gParamType_IObject, "iObject"},
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "json"
    },
});

struct CreateRenderInstance : zeno::INode {
    virtual void apply() override {
        auto instID = ZImpl(get_input2<std::string>("instID"));
        auto Geom = ZImpl(get_input2<std::string>("Geom"));
        auto Matrix = ZImpl(get_input2<std::string>("Matrix"));
        auto Material = ZImpl(get_input2<std::string>("Material"));
        if (instID.empty()) {
            auto info = zeno::format("instID {} can not be empty!", instID);
            throw zeno::makeError(info);
        }

        auto out_json = std::make_unique<JsonObject>();
        out_json->json["BasicRenderInstances"][instID] = {
            {"Geom", Geom},
            {"Matrix", Matrix},
            {"Material", Material},
        };
        out_json->json["Root"] = instID;
        set_output("json", std::move(out_json));
    }
};

ZENDEFNODE( CreateRenderInstance, {
    {
        {gParamType_String, "instID", ""},
        {gParamType_String, "Geom", ""},
        {gParamType_String, "Matrix", "Identity"},
        {gParamType_String, "Material", "Default"},
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "shader",
    },
});

struct RenderGroup : zeno::INode {
    virtual void apply() override {
        auto RenderGroupID = ZImpl(get_input2<std::string>("RenderGroupID"));
        auto is_static = ZImpl(get_input2<bool>("static"));
        auto Matrix_string = ZImpl(get_input2<std::string>("Matrixes"));
        std::vector<std::string> Matrixes = zeno::split_str(Matrix_string, {' ', '\n'});
        auto items = ZImpl(get_input<ListObject>("items"))->m_impl->get<JsonObject>();

        std::set<std::string> rinst;
        std::map<std::string, int> id_checker;

        Json node = {};
        node["Objects"] = Json::array();
        for (const auto& item: items) {
            node["Objects"].push_back(item->json["Root"]);
        }
        node["Matrixes"] = Json::array();
        for (auto &matrix: Matrixes) {
            node["Matrixes"].push_back(matrix);
        }

        auto out_json = std::make_unique<JsonObject>();
        out_json->json["Root"] = RenderGroupID;

        for (const auto& item: items) {
            for (auto& [key, value] : item->json["BasicRenderInstances"].items()) {
                out_json->json["BasicRenderInstances"][key] = value;
                rinst.insert(key);
            }
            for (auto& [key, value] : item->json["DynamicRenderGroups"].items()) {
                out_json->json["DynamicRenderGroups"][key] = value;
                id_checker[key] += 1;
            }
            for (auto& [key, value] : item->json["StaticRenderGroups"].items()) {
                if (is_static) {
                    out_json->json["StaticRenderGroups"][key] = value;
                }
                else {
                    out_json->json["DynamicRenderGroups"][key] = value;
                }
                id_checker[key] += 1;
            }
        }

        if (is_static && !out_json->json.contains("DynamicRenderGroups")) {
            out_json->json["StaticRenderGroups"][RenderGroupID] = node;
        }
        else {
            out_json->json["DynamicRenderGroups"][RenderGroupID] = node;
        }
        id_checker[RenderGroupID] += 1;

        for (auto const &[GroupID, count]: id_checker) {
            if (count > 1) {
                auto info = zeno::format("Group ID {} is not unique!", GroupID);
                zeno::log_error(info);
                throw zeno::makeError(info);
            }
            if (rinst.count(GroupID)) {
                auto info = zeno::format("Group ID {} is not same with RenderInstance ID!", GroupID);
                zeno::log_error(info);
                throw zeno::makeError(info);
            }
        }

        set_output("json", std::move(out_json));
    }
};

ZENDEFNODE( RenderGroup, {
    {
        {gParamType_String, "RenderGroupID"},
        {gParamType_List, "items"},
        {gParamType_Bool, "static", "1"},
        {gParamType_String, "Matrixes", "Identity"},
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {
        "shader",
    },
});

struct PyJsonHost: INode {
    const std::string pre_process = R"(
import sys, json
sys.stderr = sys.stdout
from pathlib import Path
current_dir = Path(__file__).parent.resolve()
input_file_path = current_dir / 'input.json'
in_json = {}
with open(input_file_path, 'r', encoding='utf-8') as f:
    in_json = json.load(f)
out_json = {}
)";
    const std::string post_process = R"(
output_file_path = current_dir / 'output.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(out_json, f)
)";
    void apply() override {
        UUIDv4::UUIDGenerator<std::mt19937_64> uuidGenerator;
        std::string uuid_str = uuidGenerator.getUUID().str();
        fs::path temp_dir = fs::temp_directory_path();
        fs::path temp_node_dir = temp_dir / ("pyjsonhost-" + uuid_str);
        fs::create_directories(temp_node_dir);
        std::string input_json = "{}";
        {
            if (ZImpl(has_input2<std::string>("in_json"))) {
                input_json = ZImpl(get_input2<std::string>("in_json"));
            }
            else if (ZImpl(has_input2<JsonObject>("in_json"))) {
                input_json = ZImpl(get_input<JsonObject>("in_json"))->json.dump();
            }
            auto input_json_path = temp_node_dir / "input.json";
            file_put_content(input_json_path.string(), input_json);
        }
        {
            auto py_code = ZImpl(get_input2<std::string>("py_code"));
            py_code = pre_process + py_code + post_process;
            auto py_code_path = temp_node_dir / "run.py";
            file_put_content(py_code_path.string(), py_code);
            std::string cmd = "python " + py_code_path.string();
            system(cmd.c_str());
        }
        {
            auto output_json_path = temp_node_dir / "output.json";
            auto output_json = file_get_content(output_json_path.string());
            if (ZImpl(get_input2<bool>("output json as string"))) {
                ZImpl(set_output2("out_json", output_json));
            }
            else {
                auto json_obj = std::make_unique<JsonObject>();
                try {
                    json_obj->json = Json::parse(output_json);
                } catch (const std::exception& e) {
                    zeno::log_error(output_json);
                    throw std::runtime_error("Failed to parse output JSON: " + std::string(e.what()));
                }
                ZImpl(set_output2("out_json", std::move(json_obj)));
            }
        }
        fs::remove_all(temp_node_dir);
    }
};

ZENDEFNODE(PyJsonHost, {
    {
        {gParamType_String, "in_json"},
        {gParamType_String, "py_code", "out_json = in_json", Socket_Primitve, zeno::Multiline},
        {gParamType_Bool, "output json as string", "1"},
    },
    {
        {gParamType_JsonObject, "out_json"}
    },
    {},
    {"json"},
});

#ifdef ZENO_WITH_PYTHON
static PyObject * pycheck(PyObject *pResult) {
    if (pResult == nullptr) {
        PyErr_Print();
        throw zeno::makeError("python err");
    }
    return pResult;
}

static void pycheck(int result) {
    if (result != 0) {
        PyErr_Print();
        throw zeno::makeError("python err");
    }
}

#if 0
struct PyJson: INode {
    const std::string pre_process = "import sys, json\nsys.stderr = sys.stdout\nin_json = json.loads(input_json)\nout_json={}\n";
    const std::string post_process = "\noutput_json = json.dumps(out_json)";
    void apply() override {
        std::string input_json;
        if (has_input("in_json")) {
            input_json = ZImpl(get_input2<std::string>("in_json"));
        }
        else {
            input_json = ZImpl(get_input<JsonObject>("in_json"))->json.dump();
        }
        auto py_code = ZImpl(get_input2<std::string>("py_code"));
        Py_Initialize();
        zeno::scope_exit init_defer([=]{ Py_Finalize(); });
        PyObject* userGlobals = PyDict_New();
        PyObject* pyInnerValue = PyUnicode_DecodeUTF8(input_json.c_str(), input_json.size(), "strict");
        pycheck(PyDict_SetItemString(userGlobals, "input_json", pyInnerValue));
        std::string python_code = pre_process + py_code + post_process;
        pycheck(PyRun_String(python_code.c_str(), Py_file_input, userGlobals, nullptr));
        PyObject *result_value = pycheck(PyDict_GetItemString(userGlobals, "output_json"));
        std::string out = PyUnicode_AsUTF8(result_value);
        if (ZImpl(get_input2<bool>("output json as string"))) {
            ZImpl(set_output2("out_json", out));
        }
        else {
            auto json_obj = std::make_unique<JsonObject>();
            json_obj->json = Json::parse(out);
            ZImpl(set_output2("out_json", json_obj));
        }
    }
};

ZENDEFNODE(PyJson, {
    {
        "in_json",
        {"multiline_string", "py_code", "out_json = in_json"},
        {"bool", "output json as string", "1"},
    },
    {
        "out_json",
    },
    {},
    {"json"},
});

struct PyText: INode {
    const std::string pre_process = "import sys\nsys.stderr = sys.stdout\nout_text = ''\n";
    const std::string post_process = "";
    void apply() override {
        std::string input_json = get_input2<std::string>("in_text");
        auto py_code = get_input2<std::string>("py_code");
        Py_Initialize();
        zeno::scope_exit init_defer([=]{ Py_Finalize(); });
        PyObject* userGlobals = PyDict_New();
        PyObject* pyInnerValue = PyUnicode_DecodeUTF8(input_json.c_str(), input_json.size(), "strict");
        pycheck(PyDict_SetItemString(userGlobals, "in_text", pyInnerValue));
        std::string python_code = pre_process + py_code + post_process;
        pycheck(PyRun_String(python_code.c_str(), Py_file_input, userGlobals, nullptr));
        PyObject *result_value = pycheck(PyDict_GetItemString(userGlobals, "out_text"));
        std::string out = PyUnicode_AsUTF8(result_value);
        set_output2("out_text", out);
    }
};

ZENDEFNODE(PyText, {
    {
        {gParamType_String, "in_text", ""},
        {"multiline_string", "py_code", "out_text = in_text"},
    },
    {
        "out_text",
    },
    {},
    {"json"},
});
#endif
#endif

}