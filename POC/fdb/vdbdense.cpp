#include <openvdb/tools/Dense.h>
#include <string_view>
#include "vec.h"

namespace zinc {

int writevdb
        ( std::string_view path
        , vec3i size
        )
{
    openvdb::CoordBBox bbox = openvdb::CoordBBox(size);
    openvdb::tools::Dense dens(bbox);
}

}
