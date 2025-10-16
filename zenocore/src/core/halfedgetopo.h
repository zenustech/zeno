#pragma once

#include "geotopology.h"

namespace zeno
{
    struct HEdge;
    struct HF_Face;
    struct HF_Point;

    struct HEdge {
        std::string id;
        HEdge* pair = 0, * next = 0;
        int point = -1;
        int point_from = -1;
        int face = -1;

        int find_from() {
            if (point_from != -1) return point_from;
            if (pair) return pair->point;
            HEdge* h = this;
            while (h->next != this) {
                h = h->next;
            }
            return h->point;
        }
    };

    struct HF_Face {
        int start_linearIdx;  //the vertex index of start vertex.
        HEdge* h = 0;      //应该是起始边
    };

    struct HF_Point {
        std::set<HEdge*> edges;    //all h-edge starting from this point.
    };


    struct HalfEdgeTopology : IGeomTopology
    {
        HalfEdgeTopology() = default;
        HalfEdgeTopology(const HalfEdgeTopology& rhs);
        HalfEdgeTopology(bool bTriangle, int nPoints, int nFaces, bool bInitFaces = false);
        virtual ~HalfEdgeTopology();

        GeomTopoType type() const override;
        std::shared_ptr<IGeomTopology> clone() override;

        void initFromPrim(int n_points, PrimitiveObject* prim);
        void toPrimitive(PrimitiveObject* spPrim);

        HEdge* checkHEdge(int fromPoint, int toPoint);
        std::tuple<HF_Point*, HEdge*, HEdge*> getPrev(HEdge* outEdge);
        int getNextOutEdge(int fromPoint, int currentOutEdge);
        int getPointTo(HEdge* hedge) const;

        std::vector<vec3i> tri_indice() const override;
        std::vector<std::vector<int>> face_indice() const override;
        std::vector<int> edge_list() const override;
        bool is_base_triangle() const override;
        bool is_line() const override;

        int npoints_in_face(HF_Face* face) const;
        void geomTriangulate(zeno::TriangulateInfo& info);

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
        std::vector<int> point_faces(int point_id) const override;
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
        void update_linear_vertex();

    private:
        std::vector<std::shared_ptr<HF_Point>> m_points;
        std::vector<std::shared_ptr<HF_Face>> m_faces;
        std::unordered_map<std::string, std::shared_ptr<HEdge>> m_hEdges;
        bool m_bTriangle = true;    //所有面都是三角面
    };
}