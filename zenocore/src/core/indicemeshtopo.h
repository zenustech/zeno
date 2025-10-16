#pragma once

#include "geotopology.h"

namespace zeno
{
    class IndiceMeshTopology : public IGeomTopology
    {
    public:
        IndiceMeshTopology() = default;
        IndiceMeshTopology(const IndiceMeshTopology& rhs) = delete;
        IndiceMeshTopology(PrimitiveObject* prim);
        IndiceMeshTopology(bool bTriangle, int nPoints, int nFaces, bool bInitFaces);
        IndiceMeshTopology(bool bTriangle, int nPoints, const std::vector<std::vector<int>>& faces);

        GeomTopoType type() const override;
        std::shared_ptr<IGeomTopology> clone() override;
        std::unique_ptr<PrimitiveObject> toPrimitiveObject() const;

        std::vector<vec3i> tri_indice() const override;
        std::vector<std::vector<int>> face_indice() const override;
        std::vector<int> edge_list() const override;
        bool is_base_triangle() const override;
        bool is_line() const override;

        /* 添加元素 */
        int add_face(const std::vector<int>& points, bool bClose) override;
        void set_face(int idx, const std::vector<int>& points, bool bClose) override;
        int add_point() override;
        int add_vertex(int face_id, int point_id) override;

        /* 移除元素相关 */
        bool remove_faces(const std::set<int>& faces, bool includePoints, std::vector<int>& removedPtnum) override;
        bool remove_point(int ptnum) override;
        bool remove_vertex(int face_id, int vert_id) override;

        /* 返回元素个数 */
        int npoints() const override;
        int nfaces() const override;
        int nvertices() const override;
        int nvertices(int face_id) const override;

        /* 点相关 */
        std::vector<int> point_faces(int point_id) const  override;
        int point_vertex(int point_id) const override;
        std::vector<int> point_vertices(int point_id) const override;

        /* 面相关 */
        int face_point(int face_id, int vert_id) const override;
        std::vector<int> face_points(int face_id) const override;
        int face_vertex(int face_id, int vert_id) const override;
        int face_vertex_count(int face_id) const override;
        std::vector<int> face_vertices(int face_id) const override;

        /* Vertex相关 */
        int vertex_index(int face_id, int vertex_id) const override;
        int vertex_next(int linear_vertex_id) const override;
        int vertex_prev(int linear_vertex_id) const override;
        std::tuple<int, int, int>  vertex_info(int linear_vertex_id) const override;
        int vertex_point(int linear_vertex_id) const override;
        int vertex_face(int linear_vertex_id) const override;
        int vertex_face_index(int linear_vertex_id) const override;

    private:
        std::unique_ptr<PrimitiveObject> m_indiceMesh_topo;
        int m_point_size;
    };
}