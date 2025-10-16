#include "indicemeshtopo.h"
#include <zeno/para/parallel_reduce.h>

namespace zeno
{
    IndiceMeshTopology::IndiceMeshTopology(PrimitiveObject* prim)
        : m_indiceMesh_topo(std::make_unique<PrimitiveObject>(*prim))
        , m_point_size(prim->size())
    {
        //不需要顶点数据和属性，是记录在外面的
        m_indiceMesh_topo->verts.clear();
        m_indiceMesh_topo->verts.attrs.clear();
    }

    IndiceMeshTopology::IndiceMeshTopology(bool bTriangle, int nPoints, int nFaces, bool bInitFaces)
        : m_point_size(nPoints)
    {
        m_indiceMesh_topo = std::make_unique<PrimitiveObject>();
        if (bTriangle) {
            m_indiceMesh_topo->tris->resize(nFaces);
        }
        else {
            //loops目前还没初始化，需要知道拓扑
            m_indiceMesh_topo->polys->resize(nFaces);
        }
    }

    IndiceMeshTopology::IndiceMeshTopology(bool bTriangle, int nPoints, const std::vector<std::vector<int>>& faces)
        : m_point_size(nPoints)
    {
        m_indiceMesh_topo = std::make_unique<PrimitiveObject>();
        int nFace = faces.size();
        if (bTriangle) {
            m_indiceMesh_topo->tris->resize(nFace);
            for (auto i = 0; i < nFace; i++) {
                const auto& face = faces[i];
                if (face.size() != 3) throw makeError<UnimplError>("");
                m_indiceMesh_topo->tris[i] = { face[0], face[1], face[2] };
            }
        }
        else {
            m_indiceMesh_topo->polys->resize(nFace);
            int start_offset_in_loop = 0;
            for (auto i = 0; i < nFace; i++) {
                const auto& face = faces[i];
                for (auto pt : face) {
                    m_indiceMesh_topo->loops->push_back(pt);
                }
                m_indiceMesh_topo->polys[i] = { start_offset_in_loop, (int)face.size() };
                start_offset_in_loop += (int)face.size();
            }
        }
    }

    GeomTopoType IndiceMeshTopology::type() const {
        return Topo_IndiceMesh;
    }

    std::shared_ptr<IGeomTopology> IndiceMeshTopology::clone() {
        return std::make_shared<IndiceMeshTopology>(m_indiceMesh_topo.get());
    }

    std::unique_ptr<PrimitiveObject> IndiceMeshTopology::toPrimitiveObject() const {
        return std::make_unique<PrimitiveObject>(*m_indiceMesh_topo);
    }

    std::vector<vec3i> IndiceMeshTopology::tri_indice() const {
        throw makeError<UnimplError>("");
    }

    std::vector<std::vector<int>> IndiceMeshTopology::face_indice() const {
        throw makeError<UnimplError>("");
    }

    std::vector<int> IndiceMeshTopology::edge_list() const {
        throw makeError<UnimplError>("");
    }

    bool IndiceMeshTopology::is_base_triangle() const {
        return !m_indiceMesh_topo->tris->empty();
    }

    bool IndiceMeshTopology::is_line() const {
        return false;
    }

    /* 添加元素 */
    int IndiceMeshTopology::add_face(const std::vector<int>& points, bool bClose) {
        throw makeError<UnimplError>("");
    }

    void IndiceMeshTopology::set_face(int idx, const std::vector<int>& points, bool bClose) {
        throw makeError<UnimplError>("");
    }

    int IndiceMeshTopology::add_point() {
        throw makeError<UnimplError>("");
    }

    int IndiceMeshTopology::add_vertex(int face_id, int point_id) {
        throw makeError<UnimplError>("");
    }

    /* 移除元素相关 */
    bool IndiceMeshTopology::remove_faces(const std::set<int>& faces, bool includePoints, std::vector<int>& removedPtnum) {
        throw makeError<UnimplError>("");
    }

    bool IndiceMeshTopology::remove_point(int ptnum) {
        throw makeError<UnimplError>("");
    }

    bool IndiceMeshTopology::remove_vertex(int face_id, int vert_id) {
        throw makeError<UnimplError>("");
    }

    /* 返回元素个数 */
    int IndiceMeshTopology::npoints() const {
        //verts其实只存在Geom层面，拓扑层面后续会自定义，而不是沿用PrimitiveObject
        return m_point_size;
    }

    int IndiceMeshTopology::nfaces() const {
        if (is_base_triangle()) {
            return m_point_size;
        }
        else {
            return m_indiceMesh_topo->polys->size();
        }
    }

    int IndiceMeshTopology::nvertices() const {
        if (is_base_triangle()) {
            return m_indiceMesh_topo->tris->size() * 3;
        }
        else {
            return parallel_reduce_sum(m_indiceMesh_topo->polys->begin(), m_indiceMesh_topo->polys->end(), [&](vec2i const& ind) {
                return ind[1];
            });
        }
    }

    int IndiceMeshTopology::nvertices(int face_id) const {
        if (is_base_triangle()) {
            return 3;
        }
        else {
            if (face_id < 0 || face_id >= m_indiceMesh_topo->polys->size()) {
                throw makeError<UnimplError>("face_id invalid");
            }
            return m_indiceMesh_topo->polys[face_id][1];
        }
    }

    /* 点相关 */
    std::vector<int> IndiceMeshTopology::point_faces(int point_id) const {
        throw makeError<UnimplError>("");
    }

    int IndiceMeshTopology::point_vertex(int point_id) const {
        throw makeError<UnimplError>("");
    }

    std::vector<int> IndiceMeshTopology::point_vertices(int point_id) const {
        throw makeError<UnimplError>("");
    }

    /* 面相关 */
    int IndiceMeshTopology::face_point(int face_id, int vert_id) const {
        throw makeError<UnimplError>("");
    }

    std::vector<int> IndiceMeshTopology::face_points(int face_id) const {
        const auto& face_summary = m_indiceMesh_topo->polys[face_id];
        int startIdx = face_summary[0];
        int count = face_summary[1];
        std::vector<int> face;
        for (int vert = startIdx; vert < startIdx + count; vert++) {
            face.push_back(m_indiceMesh_topo->loops[vert]);
        }
        return face;
    }

    int IndiceMeshTopology::face_vertex(int face_id, int vert_id) const {
        throw makeError<UnimplError>("");
    }

    int IndiceMeshTopology::face_vertex_count(int face_id) const {
        throw makeError<UnimplError>("");
    }

    std::vector<int> IndiceMeshTopology::face_vertices(int face_id) const {
        throw makeError<UnimplError>("");
    }

    /* Vertex相关 */
    int IndiceMeshTopology::vertex_index(int face_id, int vertex_id) const {
        throw makeError<UnimplError>("");
    }

    int IndiceMeshTopology::vertex_next(int linear_vertex_id) const {
        throw makeError<UnimplError>("");
    }

    int IndiceMeshTopology::vertex_prev(int linear_vertex_id) const {
        throw makeError<UnimplError>("");
    }

    std::tuple<int, int, int> IndiceMeshTopology::vertex_info(int linear_vertex_id) const {
        //throw makeError<UnimplError>("");
        return { -1,-1,-1 };
    }

    int IndiceMeshTopology::vertex_point(int linear_vertex_id) const {
        //throw makeError<UnimplError>("");
        return -1;
    }

    int IndiceMeshTopology::vertex_face(int linear_vertex_id) const {
        //throw makeError<UnimplError>("");
        return -1;
    }

    int IndiceMeshTopology::vertex_face_index(int linear_vertex_id) const {
        //throw makeError<UnimplError>("");
        return -1;
    }
}