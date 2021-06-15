#include <zen/zen.h>
#include <zen/StringObject.h>
#include <zen/GlobalState.h>
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

struct ExportPath : zen::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        char buf[100];
        sprintf(buf, "%06d", zen::state.frameid);
        auto path = fs::path(zen::state.iopath) / buf;
        if (!fs::is_directory(path)) {
            fs::create_directory(path);
        }
        path /= name;
        auto ret = std::make_unique<zen::StringObject>();
        ret->set(path.string());
        set_output("path", std::move(ret));
    }
};

ZENDEFNODE(ExportPath, {
    {},
    {"path"},
    {{"string", "name", "out.zpm"}},
    {"fileio"},
});

struct EndFrame : zen::INode {
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
    }
};

ZENDEFNODE(EndFrame, {
    {"chain"},
    {},
    {},
    {"fileio"},
});
