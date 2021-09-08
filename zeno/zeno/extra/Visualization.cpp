#ifdef ZENO_VISUALIZATION
#ifndef ZENO_GLOBALSTATE
#error "ZENO_GLOBALSTATE must be ON when ZENO_VISUALIZATION is ON"
#endif
#include <zeno/zeno.h>
#include <zeno/extra/Visualization.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/filesystem.h>
#include <fstream>
#include <cstdio>

namespace zeno::Visualization {

static int objid = 0;

ZENO_API std::string exportPath() {
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

ZENO_API void endFrame() {
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
