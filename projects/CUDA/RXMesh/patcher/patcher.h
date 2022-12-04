#pragma once

#include <functional>
#include <unordered_map>
#include "../utils/util.h"

namespace zeno::rxmesh {

/**
 * @brief Takes an input mesh and partition it to patches using Lloyd algorithm
 * on the gpu
 */
class Patcher {
   public:
    Patcher () = default;
    Patcher(uint32_t                                  patch_size,
            const std::vector<uint32_t>&              ff_offset,
            const std::vector<uint32_t>&              ff_values,
            const std::vector<std::vector<uint32_t>>& fv,
            const std::unordered_map<std::pair<uint32_t, uint32_t>,
                                     uint32_t,
                                     zeno::rxmesh::detail::edge_key_hash> edges_map,
            const uint32_t num_vertices,
            const uint32_t num_edges);

    ~Patcher();

    uint32_t get_num_patches() const {
        return m_num_patches;
    }
    uint32_t get_patch_size() const {
        return m_patch_size;
    }

    std::vector<uint32_t>& get_face_patch() {
        return m_face_patch;
    }

    uint32_t* get_device_face_patch() {
        return m_d_face_patch;
    }

    uint32_t* get_device_vertex_patch() {
        return m_d_vertex_patch;
    }

    uint32_t* get_device_edge_patch() {
        return m_d_edge_patch;
    }

    std::vector<uint32_t>& get_vertex_patch() {
        return m_vertex_patch;
    }

    std::vector<uint32_t>& get_edge_patch() {
        return m_edge_patch;
    }

    uint32_t* get_patches_val() {
        return m_patches_val.data();
    }

    uint32_t* get_patches_offset() {
        return m_patches_offset.data();
    }

    std::vector<uint32_t>& get_external_ribbon_val() {
        return m_ribbon_ext_val;
    }

    std::vector<uint32_t>& get_external_ribbon_offset() {
        return m_ribbon_ext_offset;
    }

    uint32_t get_face_patch_id(const uint32_t fid) const {
        return m_face_patch[fid];
    }

    uint32_t get_vertex_patch_id(const uint32_t vid) const {
        return m_vertex_patch[vid];
    }

    uint32_t get_edge_patch_id(const uint32_t eid) const {
        return m_edge_patch[eid];
    }
    uint32_t get_num_ext_ribbon_faces() const {
        return m_ribbon_ext_offset[m_num_patches - 1];
    }
    uint32_t get_num_components() const {
        return m_num_components;
    }

    double get_ribbon_overhead() const {
        return 100.0 * double(get_num_ext_ribbon_faces()) / double(m_num_faces);
    }

   private:
    /**
     * @brief Allocate various auxiliary memory needed to store patches info on
     * the host
     */
    void allocate_memory();

    /**
     * @brief Allocate various temporarily memory on the device needed to
     * compute patches on the device
     */
    void allocate_device_memory(const std::vector<uint32_t>& ff_offset,
                                const std::vector<uint32_t>& ff_values);

    void assign_patch(
        const std::vector<std::vector<uint32_t>>&                 fv,
        const std::unordered_map<std::pair<uint32_t, uint32_t>,
                                 uint32_t,
                                 zeno::rxmesh::detail::edge_key_hash> edges_map);

    void initialize_random_seeds(const std::vector<uint32_t>& ff_offset,
                                 const std::vector<uint32_t>& ff_values);

    void get_multi_components(std::vector<std::vector<uint32_t>>& components,
                              const std::vector<uint32_t>&        ff_offset,
                              const std::vector<uint32_t>&        ff_values);

    void initialize_random_seeds_single_component();
    void generate_random_seed_from_component(std::vector<uint32_t>& component,
                                             uint32_t               num_seeds);

    void postprocess(const std::vector<std::vector<uint32_t>>& fv,
                     const std::vector<uint32_t>&              ff_offset,
                     const std::vector<uint32_t>&              ff_values);

    uint32_t construct_patches_compressed_format();

    void run_lloyd();


    uint32_t m_patch_size, m_num_patches, m_num_vertices, m_num_edges,
        m_num_faces, m_num_seeds, m_max_num_patches, m_num_components,
        m_num_lloyd_run;

    // store the face, vertex, edge patch
    std::vector<uint32_t> m_face_patch, m_vertex_patch, m_edge_patch;
    uint32_t *            m_d_face_patch, *m_d_vertex_patch, *m_d_edge_patch;


    // Stores the patches in compressed format
    std::vector<uint32_t> m_patches_val, m_patches_offset;

    // deallocated immediately after computing patches
    uint32_t *m_d_patches_offset, *m_d_patches_size, *m_d_patches_val;

    // Stores ribbon in compressed format
    std::vector<uint32_t> m_ribbon_ext_val, m_ribbon_ext_offset;

    // caching the time taken to construct the patches
    float m_patching_time_ms;

    std::vector<uint32_t> m_seeds;

    // (deallocated immediately after computing patches)
    uint32_t* m_d_seeds;

    // stores ff on the device (deallocated immediately after computing patches)
    uint32_t *m_d_ff_values, *m_d_ff_offset;

    // utility used during creating patches (deallocated immediately after
    // computing patches)
    uint32_t *m_d_queue, *m_d_queue_ptr, *m_d_new_num_patches,
        *m_d_max_patch_size;

    // CUB temp memory(deallocated immediately after computing patches)
    void * m_d_cub_temp_storage_scan, *m_d_cub_temp_storage_max;
    size_t m_cub_scan_bytes, m_cub_max_bytes;
};
}  // namespace zeno