#include "zstartup.h"
#ifdef _WIN32
#include "zeno/InitVDB.h"
#endif

void startUp()
{
#ifdef _WIN32
    static zeno::OpenvdbInitializer g_openvdb_initializer;
#endif
}