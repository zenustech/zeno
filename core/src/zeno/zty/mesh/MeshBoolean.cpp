#include <zeno/zty/mesh/MeshBoolean.h>
#include <mcut/mcut.h>


ZENO_NAMESPACE_BEGIN
namespace zty {



struct MeshCutter::Impl {
    McContext ctx = MC_NULL_HANDLE;
    std::vector<McConnectedComponent> connComps;
};

MeshCutter::MeshCutter(Mesh const &mesh1, Mesh const &mesh2)
    : impl(std::make_unique<Impl>())
{
    mcCreateContext
    ( &impl->ctx
    , MC_NULL_HANDLE
    );

    mcDispatch
    ( impl->ctx
    , MC_DISPATCH_VERTEX_ARRAY_FLOAT
    , (float *)mesh1.vert.data()
    , mesh1.loop.data()
    , mesh1.poly.data()
    , mesh1.vert.size()
    , mesh1.poly.size()
    , (float *)mesh2.vert.data()
    , mesh2.loop.data()
    , mesh2.poly.data()
    , mesh2.vert.size()
    , mesh2.poly.size()
    );

    uint32_t numConnComps = 0;
    mcGetConnectedComponents
    ( impl->ctx
    , MC_CONNECTED_COMPONENT_TYPE_ALL
    , 0
    , NULL
    , &numConnComps
    );
    impl->connComps.resize(numConnComps);
    mcGetConnectedComponents
    ( impl->ctx
    , MC_CONNECTED_COMPONENT_TYPE_ALL
    , numConnComps
    , impl->connComps.data()
    , NULL
    );
}


Mesh MeshCutter::getComponent(size_t i)
{
    McConnectedComponent connComp = impl->connComps[i];
    uint64_t numBytes;

    mcGetConnectedComponentData
    ( impl->ctx
    , connComp
    , MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT
    , 0
    , NULL
    , &numBytes
    );
    std::vector<math::vec3f> vertices(numBytes / sizeof(math::vec3f));
    mcGetConnectedComponentData
    ( impl->ctx
    , connComp
    , MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT
    , numBytes
    , vertices.data()
    , NULL
    );

    mcGetConnectedComponentData
    ( impl->ctx
    , connComp
    , MC_CONNECTED_COMPONENT_DATA_FACE
    , 0
    , NULL
    , &numBytes
    );
    std::vector<uint32_t> faces(numBytes / sizeof(uint32_t));
    mcGetConnectedComponentData
    ( impl->ctx
    , connComp
    , MC_CONNECTED_COMPONENT_DATA_FACE
    , numBytes
    , faces.data()
    , NULL
    );

    mcGetConnectedComponentData
    ( impl->ctx
    , connComp
    , MC_CONNECTED_COMPONENT_DATA_FACE_SIZE
    , 0
    , NULL
    , &numBytes
    );
    std::vector<uint32_t> faceSizes(numBytes / sizeof(uint32_t));
    mcGetConnectedComponentData
    ( impl->ctx
    , connComp
    , MC_CONNECTED_COMPONENT_DATA_FACE_SIZE
    , numBytes
    , faceSizes.data()
    , NULL
    );

    Mesh retMesh;
    retMesh.vert = std::move(vertices);
    retMesh.loop = std::move(faces);
    retMesh.poly = std::move(faceSizes);
    return retMesh;
}


MeshCutter::~MeshCutter()
{
    mcReleaseConnectedComponents
    ( impl->ctx
    , 0
    , NULL
    );
    mcReleaseContext
    ( impl->ctx
    );
}


}
ZENO_NAMESPACE_END
