#ifndef __IGEOMETRY_TOPOLOGY_H__
#define __IGEOMETRY_TOPOLOGY_H__

#include <vector>
#include <string>
#include <zeno/core/common.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno
{
    struct IGeomTopology
    {
        IGeomTopology() = default;
        virtual ~IGeomTopology() = default;
        virtual GeomTopoType type() const = 0;
        virtual std::shared_ptr<IGeomTopology> clone() = 0;

        virtual std::vector<vec3i> tri_indice() const = 0;
        virtual std::vector<std::vector<int>> face_indice() const = 0;
        virtual std::vector<int> edge_list() const = 0;
        virtual bool is_base_triangle() const = 0;
        virtual bool is_line() const = 0;

        /* 添加元素 */
        virtual int add_face(const std::vector<int>& points, bool bClose) = 0;
        virtual void set_face(int idx, const std::vector<int>& points, bool bClose) = 0;
        virtual int add_point() = 0;
        virtual int add_vertex(int face_id, int point_id) = 0;

        /* 移除元素相关 */
        virtual bool remove_faces(const std::set<int>& faces, bool includePoints, std::vector<int>& removedPtnum) = 0;
        virtual bool remove_point(int ptnum) = 0;
        virtual bool remove_vertex(int face_id, int vert_id) = 0;

        /* 返回元素个数 */
        virtual int npoints() const = 0;
        virtual int nfaces() const = 0;
        virtual int nvertices() const = 0;
        virtual int nvertices(int face_id) const = 0;

        /* 点相关 */
        virtual std::vector<int> point_faces(int point_id) const = 0;
        virtual int point_vertex(int point_id) const = 0;
        virtual std::vector<int> point_vertices(int point_id) const = 0;

        /* 面相关 */
        virtual int face_point(int face_id, int vert_id) const = 0;
        virtual std::vector<int> face_points(int face_id) const = 0;
        virtual int face_vertex(int face_id, int vert_id) const = 0;
        virtual int face_vertex_count(int face_id) const = 0;
        virtual std::vector<int> face_vertices(int face_id) const = 0;

        /* Vertex相关 */
        virtual int vertex_index(int face_id, int vertex_id) const = 0;
        virtual int vertex_next(int linear_vertex_id) const = 0;
        virtual int vertex_prev(int linear_vertex_id) const = 0;
        virtual std::tuple<int, int, int>  vertex_info(int linear_vertex_id) const = 0;
        virtual int vertex_point(int linear_vertex_id) const = 0;
        virtual int vertex_face(int linear_vertex_id) const = 0;
        virtual int vertex_face_index(int linear_vertex_id) const = 0;
    };

    std::shared_ptr<IGeomTopology> create_halfedge_topo(bool bTriangle, int nPoints, int nFaces, bool bInitFaces = false);
    std::shared_ptr<IGeomTopology> create_halfedge_topo(bool bTriangle, int nPoints, const std::vector<std::vector<int>>& faces);
    std::shared_ptr<IGeomTopology> create_halfedge_by_indicemesh(int n_points, std::shared_ptr<IGeomTopology> indicemesh);

    std::shared_ptr<IGeomTopology> create_indicemesh_topo(PrimitiveObject* prim);
    std::shared_ptr<IGeomTopology> create_indicemesh_topo(bool bTriangle, int nPoints, int nFaces, bool bInitFaces = false);
    std::shared_ptr<IGeomTopology> create_indicemesh_topo(bool bTriangle, int nPoints, const std::vector<std::vector<int>>& faces);
    std::shared_ptr<IGeomTopology> create_indicemesh_by_halfedge(std::shared_ptr<IGeomTopology> halfedge);

    std::shared_ptr<IGeomTopology> clone_topology(std::shared_ptr<IGeomTopology> topology);

    //从拓扑中获得PrimitiveObject的表达，如果topo是indiceMesh，直接返回内建的prim，如果是halfedge，就构造一个新的返回
    std::unique_ptr<PrimitiveObject> get_primitive_topo(std::shared_ptr<IGeomTopology> topology);
}

#endif