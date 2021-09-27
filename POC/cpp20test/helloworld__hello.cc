module;

#include <cstdio>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>

module helloworld__hello;

void hello() {
    printf("Hello, world!\n");
    auto p = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.1f, openvdb::Vec3f(0), 4.0f, 0.1f);
}
