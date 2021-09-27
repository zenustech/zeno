module;

#include <cstdio>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>

export module helloworld__world;

export void world();

export inline void test() {
    auto p = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.1f, openvdb::Vec3f(0), 4.0f, 4);
    printf("Testing openvdb...\n");
}
