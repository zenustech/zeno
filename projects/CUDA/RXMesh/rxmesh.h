#pragma once
#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include "patch_info.h"
#include "context.h"
#include "types.h"
#include "patcher/patcher.h"


namespace zeno::rxmesh {

/**
 * @brief The main class for creating RXMesh data structure.
 * Not mean to be used directly by the user. Users should use RXMeshStatic instead.
 */
class RXMesh {
   public:
    uint32_t get_num_vertices() const {
        return m_num_vertices;
    }
    uint32_t get_num_edges() const {
        return m_num_edges;
    }
    uint32_t get_num_faces() const {
        return m_num_faces;
    }
    const zeno::rxmesh::Context& get_context() const {
        return m_rxmesh_context;
    }
    bool is_edge_manifold() const {
        return m_is_input_edge_manifold;
    }
    bool is_closed() const {
        return m_is_input_closed;
    }
    uint32_t get_num_patches() const{
        return m_num_patches;
    }

    uint32_t get_edge_id(const uint32_t v0, const uint32_t v1) const;

   protected:
    virtual ~RXMesh();

    RXMesh(const RXMesh&) = delete;

    RXMesh();

    void init(const std::vector<std::vector<uint32_t>>& fv);

    /**
     * @brief Set the number of vertices, edges, and faces, populate edge_map,
     * build face-incident-faces data structure.
     */
    void build_supporting_structures(
        const std::vector<std::vector<uint32_t>>& fv,
        std::vector<std::vector<uint32_t>>&       ef,
        std::vector<uint32_t>&                    ff_offset,
        std::vector<uint32_t>&                    ff_values);

    /**
     * @brief Calculate various statistics for the input mesh:
     * if the input is closed, if the input is edge manifold, and max number of
     * vertices/edges/faces per patch.
     */
    void calc_statistics(const std::vector<std::vector<uint32_t>>& fv,
                         const std::vector<std::vector<uint32_t>>& ef);

    void calc_max_not_owned_elements();

    void build(const std::vector<std::vector<uint32_t>>& fv);

    void build_single_patch_ltog(const std::vector<std::vector<uint32_t>>& fv,
                                 const uint32_t patch_id);

    void build_single_patch_topology(
        const std::vector<std::vector<uint32_t>>& fv,
        const uint32_t                            patch_id);

    void build_device();

    uint32_t get_edge_id(const std::pair<uint32_t, uint32_t>& edge) const {
        return get_edge_id(edge.first, edge.second);
    }

    template <typename T>
    friend class VertexAttribute;
    template <typename T>
    friend class EdgeAttribute;
    template <typename T>
    friend class FaceAttribute;

    zeno::rxmesh::Context m_rxmesh_context;

    uint32_t m_num_edges, m_num_faces, m_num_vertices;

    uint32_t m_max_vertices_per_patch, m_max_edges_per_patch,
        m_max_faces_per_patch;

    uint32_t m_max_not_owned_vertices, m_max_not_owned_edges,
        m_max_not_owned_faces;

    uint32_t       m_num_patches;
    const uint32_t m_patch_size;
    bool           m_is_input_edge_manifold;
    bool           m_is_input_closed;

    // Edge hash map that takes two vertices and return their edge id
    std::unordered_map<std::pair<uint32_t, uint32_t>,
                       uint32_t,
                       zeno::rxmesh::detail::edge_key_hash>
        m_edges_map;

    // pointer to the patcher class responsible for everything related to
    // patching the mesh into small pieces
    std::unique_ptr<zeno::rxmesh::Patcher> m_patcher;

    //** main incident relations
    std::vector<std::vector<uint16_t>> m_h_patches_ev;
    std::vector<std::vector<uint16_t>> m_h_patches_fe;

    // the number of owned mesh elements per patch
    std::vector<uint16_t> m_h_num_owned_f, m_h_num_owned_e, m_h_num_owned_v;

    // local to global map for (v)ertices (e)dges and (f)aces
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_v;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_e;
    std::vector<std::vector<uint32_t>> m_h_patches_ltog_f;

    zeno::rxmesh::PatchInfo *m_d_patches_info, *m_h_patches_info;
};
}  // namespace rxmesh
