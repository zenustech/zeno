#ifdef ZENO_VISUALIZATION
#ifndef ZENO_GLOBALSTATE
#error "ZENO_GLOBALSTATE must be ON when ZENO_VISUALIZATION is ON"
#endif
#include <zeno/zeno.h>
#include <zeno/Visualization.h>
#include <zeno/GlobalState.h>
#include <zeno/filesystem.h>
#include <fstream>
#include <cstdio>

namespace zeno::Visualization {

static int objid = 0;

ZENAPI std::string exportPath() {
    char buf[100];
    sprintf(buf, "%06d", zeno::state.frameid);
    auto path = fs::path(zeno::state.iopath) / buf;
    if (!fs::is_directory(path)) {
        fs::create_directory(path);
    }
    sprintf(buf, "%06d", objid++);
    path /= buf;
    //printf("EXPORTPATH: %s\n", path.c_str());
    return path.string();
}

ZENAPI void endFrame() {
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

}
#endif
