#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <filesystem>
#include "zeno/utils/log.h"
#include "zeno/types/ListObject.h"
#include "zeno/utils/string.h"
#include <cstdio>
#include <fstream>


namespace zeno {
namespace {

using DefaultValue = std::variant<
        int,
        vec2i,
        vec3i,
        vec4i,
        float,
        vec2f,
        vec3f,
        vec4f,
        std::string
        >;

struct ParamFormatInfo: IObject {
    std::string name;
    std::string _type;
    DefaultValue defaultValue;
};

struct ParamFormat : zeno::INode {
    virtual void apply() override {
        auto format = std::make_shared<zeno::ParamFormatInfo>();
        format->name = get_input2<std::string>("name");
        format->_type = get_input2<std::string>("type");
        auto defaultValue = get_input2<std::string>("defaultValue");
        auto items = zeno::split_str(defaultValue, ',');
        if (format->_type == "int") {
            format->defaultValue = std::stoi(defaultValue);
        }
        else if (format->_type == "vec2i") {
            format->defaultValue = vec2i(
                std::stoi(items[0]),
                std::stoi(items[1])
            );
        }
        else if (format->_type == "vec3i") {
            format->defaultValue = vec3i(
                    std::stoi(items[0]),
                    std::stoi(items[1]),
                    std::stoi(items[2])
            );
        }
        else if (format->_type == "vec4i") {
            format->defaultValue = vec4i(
                    std::stoi(items[0]),
                    std::stoi(items[1]),
                    std::stoi(items[2]),
                    std::stoi(items[3])
            );
        }
        else if (format->_type == "float") {
            format->defaultValue = std::stof(defaultValue);
        }
        else if (format->_type == "vec2f") {
            format->defaultValue = vec2f(
                    std::stof(items[0]),
                    std::stof(items[1])
            );
        }
        else if (format->_type == "vec3f") {
            format->defaultValue = vec3f(
                    std::stof(items[0]),
                    std::stof(items[1]),
                    std::stof(items[2])
            );
        }
        else if (format->_type == "vec4f") {
            format->defaultValue = vec4f(
                    std::stof(items[0]),
                    std::stof(items[1]),
                    std::stof(items[2]),
                    std::stof(items[3])
            );
        }
        else {
            format->defaultValue = defaultValue;
        }

        set_output("format", std::move(format));
    }
};

ZENDEFNODE(ParamFormat, {
    {
        {"string", "name"},
        {"enum float vec2f vec3f vec4f int vec2i vec3i vec4i string", "type", "string"},
        {"string", "defaultValue"},
    },
    {"format"},
    {},
    {"layout"},
});

struct ParamFileParser : zeno::INode {
    virtual void apply() override {
        auto formatList = get_input<zeno::ListObject>("formatList");
        auto params = std::make_shared<zeno::DictObject>();
        auto path = get_input2<std::string>("configFilePath");
        if (std::filesystem::exists(path)) {

            zeno::log_info("exists");
            auto is = std::ifstream(path);
            while (!is.eof()) {
                std::string line;
                std::getline(is, line);
                line = zeno::trim_string(line);
                if (line.empty()) {
                    continue;
                }
                auto items = zeno::split_str(line, ',');
                zany value;
                if (items[1] == "int") {
                    value = std::make_shared<NumericObject>(
                            std::stoi(items[2])
                    );
                }
                else if (items[1] == "vec2i") {
                    value = std::make_shared<NumericObject>(vec2i(
                            std::stoi(items[2]),
                            std::stoi(items[3])
                            )
                    );
                }
                else if (items[1] == "vec3i") {
                    value = std::make_shared<NumericObject>(vec3i(
                                std::stoi(items[2]),
                                std::stoi(items[3]),
                                std::stoi(items[4])
                        )
                    );
                }
                else if (items[1] == "vec4i") {
                    value = std::make_shared<NumericObject>(vec4i(
                            std::stoi(items[2]),
                            std::stoi(items[3]),
                            std::stoi(items[4]),
                            std::stoi(items[5])
                        )
                    );
                }
                else if (items[1] == "float") {
                    value = std::make_shared<NumericObject>(
                        std::stof(items[2])
                    );
                }
                else if (items[1] == "vec2f") {
                    value = std::make_shared<NumericObject>(vec2f(
                            std::stof(items[2]),
                            std::stof(items[3])
                        )
                    );
                }
                else if (items[1] == "vec3f") {
                    value = std::make_shared<NumericObject>(vec3f(
                            std::stof(items[2]),
                            std::stof(items[3]),
                            std::stof(items[4])
                        )
                    );
                }
                else if (items[1] == "vec4f") {
                    value = std::make_shared<NumericObject>(vec4f(
                            std::stof(items[2]),
                            std::stof(items[3]),
                            std::stof(items[4]),
                            std::stof(items[5])
                        )
                    );
                }
                else {
                    value = std::make_shared<StringObject>(items[2]);
                }
                params->lut[items[0]] = value;
            }
        }
        else {
            FILE *fp = fopen(path.c_str(), "w");
            if (!fp) {
                perror(path.c_str());
                abort();
            }
            for (auto &ptr: formatList->arr) {
                auto p = std::static_pointer_cast<ParamFormatInfo>(ptr);
                zany value;
                if (std::holds_alternative<int>(p->defaultValue)) {
                    auto v = std::get<int>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%d\n", p->name.c_str(), p->_type.c_str(), v);
                }
                else if (std::holds_alternative<vec2i>(p->defaultValue)) {
                    auto v = std::get<vec2i>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%d,%d\n", p->name.c_str(), p->_type.c_str(), v[0], v[1]);
                }
                else if (std::holds_alternative<vec3i>(p->defaultValue)) {
                    auto v = std::get<vec3i>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%d,%d,%d\n", p->name.c_str(), p->_type.c_str(), v[0], v[1], v[2]);
                }
                else if (std::holds_alternative<vec4i>(p->defaultValue)) {
                    auto v = std::get<vec4i>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%d,%d,%d,%d\n", p->name.c_str(), p->_type.c_str(), v[0], v[1], v[2], v[3]);
                }
                else if (std::holds_alternative<float>(p->defaultValue)) {
                    auto v = std::get<float>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%f\n", p->name.c_str(), p->_type.c_str(), v);
                }
                else if (std::holds_alternative<vec2f>(p->defaultValue)) {
                    auto v = std::get<vec2f>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%f,%f\n", p->name.c_str(), p->_type.c_str(), v[0], v[1]);
                }
                else if (std::holds_alternative<vec3f>(p->defaultValue)) {
                    auto v = std::get<vec3f>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%f,%f,%f\n", p->name.c_str(), p->_type.c_str(), v[0], v[1], v[2]);
                }
                else if (std::holds_alternative<vec4f>(p->defaultValue)) {
                    auto v = std::get<vec4f>(p->defaultValue);
                    value = std::make_shared<NumericObject>(v);
                    fprintf(fp, "%s,%s,%f,%f,%f,%f\n", p->name.c_str(), p->_type.c_str(), v[0], v[1], v[2], v[3]);
                }
                else if (std::holds_alternative<std::string>(p->defaultValue)) {
                    auto v = std::get<std::string>(p->defaultValue);
                    value = std::make_shared<StringObject>(v);
                    fprintf(fp, "%s,%s,%s\n", p->name.c_str(), p->_type.c_str(), v.c_str());
                }
                params->lut[p->name] = value;
            }
        }
        set_output("params", std::move(params));
    }
};

ZENDEFNODE(ParamFileParser, {
    {
        "formatList",
        {"writepath", "configFilePath"},
     },
    {
        {"DictObject", "params"},
    },
    {},
    {"layout"},
});

}
}
