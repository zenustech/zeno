//
// Created by zh on 2023/11/14.
//

#include <zeno/zeno.h>
#include <tinygltf/json.hpp>
#include "zeno/utils/fileio.h"
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
    }
};
ZENDEFNODE(JsonGetChild, {
    {
        {"json"},
        {"string", "name"},
        {"enum json int float string vec2f vec3f", "type"},
    },
    {
        "out",
    },
    {},
    {
        "json"
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
        "json"
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
        "json"
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
        "json"
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
        "json"
    },
});

}