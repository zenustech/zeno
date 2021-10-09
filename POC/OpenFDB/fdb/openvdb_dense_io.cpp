#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include "openvdb_dense_io.h"
#include "vec.h"

namespace fdb {

template <class GridT, class ValT>
void impl_write_dense_vdb
        ( std::string path
        , std::function<ValT(vec3i)> sampler
        , vec3i start
        , vec3i stop
        )
{
    openvdb::CoordBBox bbox(
            openvdb::Coord(start[0], start[1], start[2]),
            openvdb::Coord(stop[0], stop[1], stop[2]));
    openvdb::tools::Dense<typename GridT::ValueType> dens(bbox);
    for (int z = start[2]; z < stop[2]; z++) {
        for (int y = start[1]; y < stop[1]; y++) {
            for (int x = start[0]; x < stop[0]; x++) {
                auto val = sampler({x, y, z});
                dens.setValue(x - start[0], y - start[1], z - start[2],
                        vec_to_other<typename GridT::ValueType>(val));
            }
        }
    }
    auto grid = GridT::create();
    typename GridT::ValueType tolerance{0};
    openvdb::tools::copyFromDense(dens, grid->tree(), tolerance);
    openvdb::io::File((std::string)path).write({grid});
}

void write_dense_vdb
    ( std::string path
    , std::function<float(vec3i)> sampler
    , vec3i start
    , vec3i stop
    )
{
    return impl_write_dense_vdb<openvdb::FloatGrid>(path, sampler, start, stop);
}

void write_dense_vdb
    ( std::string path
    , std::function<vec3f(vec3i)> sampler
    , vec3i start
    , vec3i stop
    )
{
    return impl_write_dense_vdb<openvdb::Vec3fGrid>(path, sampler, start, stop);
}

}
