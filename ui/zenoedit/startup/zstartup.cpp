#include "zstartup.h"
#ifdef _WIN32
    #ifdef ZENO_WITH_zenvdb
        #include "zeno/InitVDB.h"
    #endif
#endif

void startUp()
{
#ifdef _WIN32
    #ifdef ZENO_WITH_zenvdb
        static zeno::OpenvdbInitializer g_openvdb_initializer;
    #endif
#endif
}