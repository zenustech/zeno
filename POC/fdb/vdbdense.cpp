#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <string_view>
#include "vec.h"

namespace zinc {

void writevdb
        ( std::string_view path
        , std::function<float(vec3I)> sampler
        , vec3I size
        )
{
    openvdb::tools::Dense<float> dens(openvdb::Coord(size[0], size[1], size[2]));
    for (uint32_t z = 0; z < size[2]; z++) {
        for (uint32_t y = 0; y < size[1]; y++) {
            for (uint32_t x = 0; x < size[0]; x++) {
                auto val = sampler({x, y, z});
                dens.setValue(x, y, z, val);
            }
        }
    }
    auto grid = openvdb::FloatGrid::create();
    openvdb::FloatGrid::ValueType tolerance{0};
    openvdb::tools::copyFromDense(dens, grid->tree(), tolerance);
    openvdb::io::File((std::string)path).write({grid});
}

}
