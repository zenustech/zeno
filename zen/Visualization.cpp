#include <zen/zen.h>
#include <zen/Visualization.h>
#include <zen/GlobalState.h>
#include <zen/filesystem.h>
#include <fstream>
#include <cstdio>

namespace zen::Visualization {

static int objid = 0;

ZENAPI std::string exportPath(std::string const &ext) {
    char buf[100];
    sprintf(buf, "%06d", zen::state.frameid);
    auto path = fs::path(zen::state.iopath) / buf;
    if (!fs::is_directory(path)) {
        fs::create_directory(path);
    }
    sprintf(buf, "%06d.%s", objid++, ext.c_str());
    path /= buf;
    //printf("EXPORTPATH: %s\n", path.c_str());
    return path.string();
}

ZENAPI void endFrame() {
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

}
