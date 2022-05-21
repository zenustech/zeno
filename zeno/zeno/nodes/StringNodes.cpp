#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#ifdef ZENO_GLOBALSTATE
#include <zeno/extra/GlobalState.h>
#endif
#include <iostream>
#include <fstream>
#include <spdlog/spdlog.h>

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
        std::ofstream ofs{path};
        ofs << str;
        ofs.close();
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
        std::ifstream ifs{path};
        std::string buffer;
        ifs >> buffer;
        ifs.close();

        auto str = std::make_unique<zeno::StringObject>();
        str->set(buffer);
        set_output("str", std::move(str));
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

#ifdef ZENO_GLOBALSTATE
struct StringFormat : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        for (int i = 0; i < str.size() - 1; i++) {
            if (str[i] == '$' && str[i + 1] == 'F') {
                str.replace(i, 2, std::to_string(zeno::state.frameid));
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
    {"string"},
});
#endif

struct StringFormatNumber : zeno::INode {
    virtual void apply() override {
        auto str = get_input2<std::string>("str");
        auto num = get_input<zeno::NumericObject>("number");

        std::string output;
        std::visit([&](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, int>) {
                output = fmt::format(str, T(v));
            }
            else if constexpr (std::is_same_v<T, float>) {
                output = fmt::format(str, T(v));
            }
            else {
                output = str;
            }
        }, num->value);
        set_output2("str", output);
    }
};

ZENDEFNODE(StringFormatNumber, {
    {
        {"string", "str"},
        {"number"},
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
        sprintf(buf, "%06d", zeno::state.frameid);
        auto path = fs::path(zeno::state.iopath) / buf;
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
        sprintf(buf, "%06d", zeno::state.frameid);
        auto path = fs::path(zeno::state.iopath) / buf;
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

}
