#pragma once
#include <string>
#include <utility>
#include "utils/util.h"

namespace zeno::rxmesh {

/**
 * @brief Stores the information needed for query operations in a patch
 */
struct ALIGN(16) PatchInfo {
    using LocalVertexT = LocalIndexT;
    using LocalEdgeT = LocalIndexT;
    using LocalFaceT = LocalIndexT;

    // Edge incident vertices and face incident edges
    LocalVertexT* ev;
    LocalEdgeT*   fe;

    // Non-owned mesh elements patch ID
    uint32_t* not_owned_patch_v;
    uint32_t* not_owned_patch_e;
    uint32_t* not_owned_patch_f;

    // Non-owned mesh elements local ID
    LocalVertexT* not_owned_id_v;
    LocalEdgeT*   not_owned_id_e;
    LocalFaceT*   not_owned_id_f;

    // Number of mesh elements in the patch
    uint16_t num_vertices, num_edges, num_faces;

    // Number of mesh elements owned by this patch
    uint16_t num_owned_vertices, num_owned_edges, num_owned_faces;

    // The index of this patch (relative to all other patches)
    uint32_t patch_id;
};
}  // namespace rxmesh