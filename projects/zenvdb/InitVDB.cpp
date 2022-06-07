#include "zeno/InitVDB.h"
#include <openvdb/openvdb.h>
#include <cstdio>

namespace zeno {
    OpenvdbInitializer::OpenvdbInitializer() {
        printf("Initializing OpenVDB...\n");
        openvdb::initialize();
        printf("Initialized OpenVDB successfully!\n");
    }
#ifdef __linux__
    static OpenvdbInitializer g_openvdb_initializer{};
#endif
}
