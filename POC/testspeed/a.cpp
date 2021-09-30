#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>

struct cls {
    int x;

    bool funny_function() {
        if (this->funny_function()) {
            auto a = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>();
        }
        return 0;
    }
};
