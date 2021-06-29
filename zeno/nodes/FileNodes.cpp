#include <zeno/zen.h>
#include <zeno/StringObject.h>
#include <zeno/GlobalState.h>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace fs = std::filesystem;

struct MakeString : zen::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zen::StringObject>();
        obj->set(get_param<std::string>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(MakeString, {
    {},
    {"value"},
    {{"string", "value", ""}},
    {"fileio"},
});

static int objid = 0;

struct ExportPath : zen::INode {  // TODO: deprecated
    virtual void apply() override {
        char buf[100];
        auto ext = get_param<std::string>("ext");
        sprintf(buf, "%06d", zen::state.frameid);
        auto path = fs::path(zen::state.iopath) / buf;
        if (!fs::is_directory(path)) {
            fs::create_directory(path);
        }
        sprintf(buf, "%06d.%s", objid++, ext.c_str());
        path /= buf;
        auto ret = std::make_unique<zen::StringObject>();
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

struct EndFrame : zen::INode {  // TODO: deprecated
    virtual void apply() override {
        char buf[100];
        sprintf(buf, "%06d", zen::state.frameid);
        auto path = fs::path(zen::state.iopath) / buf;
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
});
