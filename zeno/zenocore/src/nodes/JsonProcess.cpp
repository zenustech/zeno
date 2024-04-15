//
// Created by zh on 2023/11/14.
//

#include "zeno/types/DictObject.h"
#include "zeno/types/ListObject.h"
#include "zeno/utils/fileio.h"
#include "zeno/utils/log.h"
#include "zeno/utils/string.h"
#include <iostream>
#include <sstream>
#include <string>
#include <tinygltf/json.hpp>
#include <zeno/zeno.h>

using Json = nlohmann::json;

namespace zeno {
struct JsonObject : IObject {
    Json json;
};
struct ReadJson : zeno::INode {
    virtual void apply() override {
        auto json = std::make_shared<JsonObject>();
        auto path = get_input2<std::string>("path");
        std::string native_path = std::filesystem::u8path(path).string();
        auto content = zeno::file_get_content(native_path);
        json->json = Json::parse(content);
        set_output("json", json);
    }
};
ZENDEFNODE(ReadJson, {
    {
        {"readpath", "path"},
    },
    {
        "json",
    },
    {},
    {
        "json"
    },
});
//P:/jielidou/EP028/HDRI/Light/ok_light.json
struct PrintJson : zeno::INode {
  virtual void apply() override {
    auto out_json = std::make_shared<JsonObject>();
    auto json = get_input<JsonObject>("json");
    std::cerr << "print json: " << to_string(json->json) << std::endl;
  }
};
ZENDEFNODE(PrintJson, {
     {
         "json",
     },
     {
     },
     {},
     {
         "json"
     },
 });
struct ReadJsonFromString : zeno::INode {
    virtual void apply() override {
        auto json = std::make_shared<JsonObject>();
        auto content = get_input2<std::string>("content");
        json->json = Json::parse(content);
        set_output("json", json);
    }
};
ZENDEFNODE(ReadJsonFromString, {
    {
        {"string", "content"},
    },
    {
        "json",
    },
    {},
    {
        "json"
    },
});
struct JsonGetArraySize : zeno::INode {
    virtual void apply() override {
        auto json = get_input<JsonObject>("json");
        set_output("size", std::make_shared<NumericObject>((int)json->json.size()));
    }
};
ZENDEFNODE(JsonGetArraySize, {
    {
        {"json"},
    },
    {
        "size",
    },
    {},
    {
        "json"
    },
});
struct JsonGetArrayItem : zeno::INode {
    virtual void apply() override {
        auto out_json = std::make_shared<JsonObject>();
        auto json = get_input<JsonObject>("json");
        auto index = get_input2<int>("index");
        out_json->json = json->json[index];
        set_output("json", out_json);
    }
};
ZENDEFNODE(JsonGetArrayItem, {
    {
        {"json"},
        {"int", "index"}
    },
    {
        "json",
    },
    {},
    {
        "json"
    },
});

struct JsonGetChild : zeno::INode {
    virtual void apply() override {
        auto json = get_input<JsonObject>("json");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");
        if (type == "json") {
            auto out_json = std::make_shared<JsonObject>();
            out_json->json = json->json[name];
            set_output("out", out_json);
        }
        else if (type == "int") {
            set_output2("out", int(json->json[name]));
        }
        else if (type == "float") {
            set_output2("out", float(json->json[name]));
        }
        else if (type == "string") {
            set_output2("out", std::string(json->json[name]));
        }
        else if (type == "vec2f") {
            float x = float(json->json[name][0]);
            float y = float(json->json[name][1]);
            set_output2("out", vec2f(x, y));
        }
        else if (type == "vec3f") {
            float x = float(json->json[name][0]);
            float y = float(json->json[name][1]);
            float z = float(json->json[name][2]);
            set_output2("out", vec3f(x, y, z));
        }
        else if (type == "vec4f") {
            float x = float(json->json[name][0]);
            float y = float(json->json[name][1]);
            float z = float(json->json[name][2]);
            float w = float(json->json[name][3]);
            set_output2("out", vec4f(x, y, z, w));
        }
    }
};
ZENDEFNODE(JsonGetChild, {
    {
        {"json"},
        {"string", "name"},
        {"enum json int float string vec2f vec3f vec4f", "type"},
    },
    {
        "out",
    },
    {},
    {
        "deprecated"
    },
});
struct JsonGetInt : zeno::INode {
    virtual void apply() override {
        auto json = get_input<JsonObject>("json");
        set_output2("value", int(json->json));
    }
};
ZENDEFNODE(JsonGetInt, {
    {
        {"json"},
    },
    {
        "value",
    },
    {},
    {
        "deprecated"
    },
});

struct JsonGetFloat : zeno::INode {
    virtual void apply() override {
        auto json = get_input<JsonObject>("json");
        set_output2("value", float(json->json));
    }
};
ZENDEFNODE(JsonGetFloat, {
    {
        {"json"},
    },
    {
        "value",
    },
    {},
    {
        "deprecated"
    },
});

struct JsonGetString : zeno::INode {
    virtual void apply() override {
        auto json = get_input<JsonObject>("json");
        set_output2("string", std::string(json->json));
    }
};
ZENDEFNODE(JsonGetString, {
    {
        {"json"},
    },
    {
        "string",
    },
    {},
    {
        "deprecated"
    },
});
struct JsonGetTypeName : zeno::INode {
    virtual void apply() override {
        auto json = get_input<JsonObject>("json");
        set_output2("string", std::string(json->json.type_name()));
    }
};
ZENDEFNODE(JsonGetTypeName, {
    {
        {"json"},
    },
    {
        "string",
    },
    {},
    {
        "deprecated"
    },
});

struct JsonData : zeno::INode {
    virtual void apply() override {
        auto json = get_input<JsonObject>("json");
        auto path = get_input2<std::string>("path");
        auto strings = zeno::split_str(path, ':');
        auto type = strings[1];
        path = strings[0];
        auto names = split_str(path, '/');

        for (auto & name : names) {
            json->json = json->json[name];
        }


        if (type == "json") {
            auto out_json = std::make_shared<JsonObject>();
            out_json->json = json->json;
            set_output("out", out_json);
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
    }
};
ZENDEFNODE(JsonData, {
    {
        {"json"},
        {"string", "path"},
    },
    {
        "out",
    },
    {},
    {
        "deprecated"
    },
});

struct JsonGetData : zeno::INode {
    virtual void apply() override {
        auto in_json = get_input<JsonObject>("json");
        auto multi_path = get_input2<std::string>("paths");
        std::istringstream iss(multi_path);
        std::vector<std::string> paths;
        std::string line;
        while (std::getline(iss, line)) {
            line = zeno::trim_string(line);
            if (line.size()) {
                paths.push_back(line);
            }
        }

        auto dict = std::make_shared<zeno::DictObject>();
        for (auto &path: paths) {
            auto json = std::make_shared<JsonObject>();
            json->json = in_json->json;
            auto strings = zeno::split_str(path, ':');
            auto type = strings[1];
            path = strings[0];
            std::string new_name = path;
            if (strings.size() == 3) {
                new_name = zeno::trim_string(strings[2]);
            }

            auto names = split_str(path, '/');

            for (auto & name : names) {
                if (json->json.is_array()) {
                    json->json = json->json[std::stoi(name)];
                }
                else {
                    json->json = json->json[name];
                }
            }

            if (type == "json") {
                auto out_json = std::make_shared<JsonObject>();
                out_json->json = json->json;
                dict->lut[new_name] = out_json;
            }
            else if (type == "int") {
                dict->lut[new_name] = std::make_shared<NumericObject>(int(json->json));
            }
            else if (type == "float") {
                dict->lut[new_name] = std::make_shared<NumericObject>(float(json->json));
            }
            else if (type == "string") {
                dict->lut[new_name] = std::make_shared<StringObject>(std::string(json->json));
            }
            else if (type == "vec2f") {
                if (json->json.is_array()) {
                    float x = float(json->json[0]);
                    float y = float(json->json[1]);
                    dict->lut[new_name] = std::make_shared<NumericObject>(vec2f(x, y));
                }
                else {
                    float x = float(json->json["x"]);
                    float y = float(json->json["y"]);
                    dict->lut[new_name] = std::make_shared<NumericObject>(vec2f(x, y));
                }
            }
            else if (type == "vec3f") {
                if (json->json.is_array()) {
                    float x = float(json->json[0]);
                    float y = float(json->json[1]);
                    float z = float(json->json[2]);
                    dict->lut[new_name] = std::make_shared<NumericObject>(vec3f(x, y, z));
                }
                else {
                    float x = float(json->json["x"]);
                    float y = float(json->json["y"]);
                    float z = float(json->json["z"]);
                    dict->lut[new_name] = std::make_shared<NumericObject>(vec3f(x, y, z));
                }
            }
            else if (type == "vec4f") {
                if (json->json.is_array()) {
                    float x = float(json->json[0]);
                    float y = float(json->json[1]);
                    float z = float(json->json[2]);
                    float w = float(json->json[3]);
                    dict->lut[new_name] = std::make_shared<NumericObject>(vec4f(x, y, z, w));
                }
                else {
                    float x = float(json->json["x"]);
                    float y = float(json->json["y"]);
                    float z = float(json->json["z"]);
                    float w = float(json->json["w"]);
                    dict->lut[new_name] = std::make_shared<NumericObject>(vec4f(x, y, z, w));
                }
            }
        }
        set_output("outs", dict);
    }
};
ZENDEFNODE(JsonGetData, {
    {
        {"json"},
        {"multiline_string", "paths"},
    },
    {
        {"DictObject", "outs"}
    },
    {},
    {
        "json"
    },
});

}