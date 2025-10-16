#include "geotopology.h"
#include "halfedgetopo.h"
#include "indicemeshtopo.h"


namespace zeno
{
    std::shared_ptr<IGeomTopology> create_halfedge_topo(bool bTriangle, int nPoints, int nFaces, bool bInitFaces) {
        auto topo = std::make_shared<HalfEdgeTopology>(bTriangle, nPoints, nFaces, bInitFaces);
        return topo;
    }

    std::shared_ptr<IGeomTopology> create_halfedge_topo(bool bTriangle, int nPoints, const std::vector<std::vector<int>>& faces) {
        auto topo = std::make_shared<HalfEdgeTopology>(bTriangle, nPoints, faces.size(), false);
        for (const auto& face : faces) {
            topo->add_face(face, true);
        }
        return topo;
    }

    std::shared_ptr<IGeomTopology> create_indicemesh_topo(PrimitiveObject* prim) {
        auto topo = std::make_shared<IndiceMeshTopology>(prim);
        return topo;
    }

    std::shared_ptr<IGeomTopology> create_indicemesh_topo(bool bTriangle, int nPoints, int nFaces, bool bInitFaces) {
        auto topo = std::make_shared<IndiceMeshTopology>(bTriangle, nPoints, nFaces, bInitFaces);
        return topo;
    }

    std::shared_ptr<IGeomTopology> create_indicemesh_topo(bool bTriangle, int nPoints, const std::vector<std::vector<int>>& faces) {
        auto topo = std::make_shared<IndiceMeshTopology>(bTriangle, nPoints, faces);
        return topo;
    }

    std::shared_ptr<IGeomTopology> create_indicemesh_by_halfedge(std::shared_ptr<IGeomTopology> halfedge) {
        if (halfedge->type() != Topo_HalfEdge) return nullptr;

        std::shared_ptr<HalfEdgeTopology> halfedge_topo = std::static_pointer_cast<HalfEdgeTopology>(halfedge);
        std::unique_ptr<PrimitiveObject> spPrim = std::make_unique<PrimitiveObject>();
        halfedge_topo->toPrimitive(spPrim.get());
        return create_indicemesh_topo(spPrim.get());
    }

    std::shared_ptr<IGeomTopology> create_halfedge_by_indicemesh(int n_points, std::shared_ptr<IGeomTopology> indicemesh) {
        if (indicemesh->type() != Topo_IndiceMesh) return nullptr;

        std::shared_ptr<IndiceMeshTopology> indice_topo = std::static_pointer_cast<IndiceMeshTopology>(indicemesh);
        std::shared_ptr<HalfEdgeTopology> halfedge_topo = std::make_shared<HalfEdgeTopology>();
        halfedge_topo->initFromPrim(n_points, indice_topo->toPrimitiveObject().get());
        return halfedge_topo;
    }

    std::shared_ptr<IGeomTopology> clone_topology(std::shared_ptr<IGeomTopology> topology) {
        return topology->clone();
    }

    //从拓扑中获得PrimitiveObject的表达，如果topo是indiceMesh，直接返回内建的prim，如果是halfedge，就构造一个新的返回
    std::unique_ptr<PrimitiveObject> get_primitive_topo(std::shared_ptr<IGeomTopology> topology) {
        GeomTopoType type = topology->type();
        if (Topo_IndiceMesh == type) {
            std::shared_ptr<IndiceMeshTopology> indice_topo = std::static_pointer_cast<IndiceMeshTopology>(topology);
            return indice_topo->toPrimitiveObject();
        }
        else if (Topo_HalfEdge == type) {
            std::shared_ptr<HalfEdgeTopology> halfedge_topo = std::static_pointer_cast<HalfEdgeTopology>(topology);
            std::unique_ptr<PrimitiveObject> spPrim = std::make_unique<PrimitiveObject>();
            halfedge_topo->toPrimitive(spPrim.get());
            return spPrim;
        }
        else {
            return nullptr;
        }
    }
}