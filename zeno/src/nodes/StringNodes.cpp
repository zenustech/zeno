#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/format.h>
#include <zeno/utils/fileio.h>
#include <zeno/extra/GlobalState.h>

namespace zeno {
namespace {

struct MakeWritePath : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::StringObject>();
        obj->set(get_param<std::string>("path"));
        set_output("path", std::move(obj));
    }
};

ZENDEFNODE(MakeWritePath, {
    {},
    {{"string", "path"}},
    {{"writepath", "path", ""}},
    {"string"},
});

struct MakeReadPath : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::StringObject>();
        obj->set(get_param<std::string>("path"));
        set_output("path", std::move(obj));
    }
};

ZENDEFNODE(MakeReadPath, {
    {},
    {{"string", "path"}},
    {{"readpath", "path", ""}},
    {"string"},
});

struct MakeString : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::StringObject>();
        obj->set(get_param<std::string>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(MakeString, {
    {},
    {{"string", "value"}},
    {{"string", "value", ""}},
    {"string"},
});

struct MakeMultilineString : MakeString {
};

ZENDEFNODE(MakeMultilineString, {
    {},
    {{"string", "value"}},
    {{"multiline_string", "value", ""}},
    {"string"},
});

struct StringEqual : zeno::INode {
    virtual void apply() override {
        auto lhs = get_input2<std::string>("lhs");
        auto rhs = get_input2<std::string>("rhs");
        set_output2("isEqual", lhs == rhs);
    }
};

ZENDEFNODE(StringEqual, {
    {{"string", "lhs"}, {"string", "rhs"}},
    {{"bool", "isEqual"}},
    {},
    {"string"},
});

struct PrintString : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        printf("PrintString: %s\n", str.c_str());
    }
};

ZENDEFNODE(PrintString, {
    {{"string", "str"}},
    {},
    {},
    {"string"},
});

struct FileWriteString
    : zeno::INode
{
    virtual void apply() override
    {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto str = get_input<zeno::StringObject>("str")->get();
        zeno::file_put_content(path, str);
    }
};

ZENDEFNODE(
    FileWriteString,
    {
        {
            {"string", "str", ""},
            {"writepath", "path", ""},
        },
        {},
        {},
        {"string"},
    });

struct FileReadString
    : zeno::INode
{
    virtual void apply() override
    {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto str = zeno::file_get_content(path);
        set_output2("str", std::move(str));
    }
};

ZENDEFNODE(
    FileReadString,
    {
        {
            {"readpath", "path"},
        },
        {
            {"string", "str"},
        },
        {},
        {"string"},
    });

struct StringFormat : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        for (int i = 0; i < str.size() - 1; i++) {
            if (str[i] == '$' && str[i + 1] == 'F') {
                str.replace(i, 2, std::to_string(getGlobalState()->frameid));
                break;
            }
        }
        set_output2("str", str);
    }
};

ZENDEFNODE(StringFormat, {
    {{"string", "str"}},
    {{"string", "str"}},
    {},
    {"deprecated"},
});

struct StringFormatNumber : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto num = get_input<zeno::NumericObject>("number");

        std::string output;
        std::visit([&](const auto &v) {
            //using T = std::decay_t<decltype(v)>;
            //if constexpr (std::is_same_v<T, int>) {
                //output = zeno::format(str, T(v));
                output = zeno::format(str, v);
            //}
            //else if constexpr (std::is_same_v<T, float>) {
                //output = zeno::format(str, T(v));
            //}
            //else {
                //output = str;
            //}
        }, num->value);
        set_output2("str", output);
    }
};

ZENDEFNODE(StringFormatNumber, {
    {
        {"string", "str", "{}"},
        {"number"},
    },
    {{"string", "str"}},
    {},
    {"deprecated"},
});

struct StringFormatNumStr : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto num_str = get_input<zeno::IObject>("num_str");
        std::string output;

        std::shared_ptr<zeno::NumericObject> num = std::dynamic_pointer_cast<zeno::NumericObject>(num_str);
        if (num) {
            std::visit([&](const auto &v) {
                output = zeno::format(str, v);
            }, num->value);
        }
        std::shared_ptr<zeno::StringObject> pStr = std::dynamic_pointer_cast<zeno::StringObject>(num_str);
        if (pStr) {
            output = zeno::format(str, pStr->get());
        }
        set_output2("str", output);
    }
};

ZENDEFNODE(StringFormatNumStr, {
    {
        {"string", "str", "{}"},
        {"num_str"},
    },
    {{"string", "str"}},
    {},
    {"string"},
});
/*static int objid = 0;

struct ExportPath : zeno::INode {  // deprecated
    virtual void apply() override {
        char buf[100];
        auto ext = get_param<std::string>("ext");
        sprintf(buf, "%06d", getGlobalState()->frameid);
        auto path = fs::path(getGlobalState()->iopath) / buf;
        if (!fs::is_directory(path)) {
            fs::create_directory(path);
        }
        sprintf(buf, "%06d.%s", objid++, ext.c_str());
        path /= buf;
        auto ret = std::make_unique<zeno::StringObject>();
        //printf("EXPORTPATH: %s\n", path.c_str());
        ret->set(path.string());
        set_output("path", std::move(ret));
    }
};

ZENDEFNODE(ExportPath, {
    {},
    {"path"},
    {{"string", "ext", "zpm"}},
    {"fileio"},
});

struct EndFrame : zeno::INode {  // deprecated
    virtual void apply() override {
        char buf[100];
        sprintf(buf, "%06d", getGlobalState()->frameid);
        auto path = fs::path(getGlobalState()->iopath) / buf;
        if (!fs::is_directory(path)) {
            fs::create_directory(path);
        }
        path /= "done.lock";
        std::ofstream ofs(path.string());
        ofs.write("DONE", 4);
        objid = 0;
    }
};

ZENDEFNODE(EndFrame, {
    {"chain"},
    {},
    {},
    {"fileio"},
});*/

struct StringToNumber : zeno::INode {

    virtual void apply() override {
        auto in_str = get_input2<std::string>("str");
        auto type = get_input2<std::string>("type");
        auto obj = std::make_unique<zeno::NumericObject>();
        if (type == "float") {
            float v = std::stof(in_str);
            obj->set(v);
        }
        else if (type == "int") {
            int v = std::stoi(in_str);
            obj->set(v);
        }
        else {
            throw zeno::makeError("Unknown type");
        }

        set_output("num_str", std::move(obj));
    }
};

ZENDEFNODE(StringToNumber, {{
                                /* inputs: */
                                {"enum float int", "type", "all"},
                                {"string", "str", "0"},
                            },

                            {
                                /* outputs: */
                                "num_str",
                            },

                            {
                                /* params: */

                            },

                            {
                                /* category: */
                                "string",
                            }});

}
}
