#pragma once
#include <assert.h>
#include <fstream>
#include <memory>
#include "attribute.h"
#include "kernels/for_each.cuh"


namespace zeno::rxmesh {

/**
 * @brief Parameters needed to launch kernels.
 * Meant to be calculated by RXMeshStatic and then used by the user.
 */
template <uint32_t blockThreads>
struct LaunchBox {
    uint32_t       blocks, num_registers_per_thread;
    size_t         smem_bytes_dyn, smem_bytes_static;
    const uint32_t num_threads = blockThreads;
};

/**
 * @brief responsible for query operations of static meshes. It extends
 * RXMesh with methods for launching kernel and do computation on the
 * mesh as well as managing mesh attributes.
 */
class RXMeshStatic : public RXMesh {
   public:
    RXMeshStatic(const RXMeshStatic&) = delete;

    RXMeshStatic(std::vector<std::vector<uint32_t>>& fv)
        : RXMesh(), m_input_vertex_coordinates(nullptr) {
        this->init(fv);
        m_attr_container = std::make_shared<AttributeContainer>();
    };

    virtual ~RXMeshStatic() {}


    /**
     * @brief Apply a lambda function on all vertices in the mesh
     */
    template <typename LambdaT>
    void for_each_vertex(locationT    location,
                         LambdaT      apply,
                         cudaStream_t stream = NULL) {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();
#pragma omp parallel for
            for (int p = 0; p < num_patches; ++p) {
                for (uint16_t v = 0;
                     v < this->m_h_patches_info[p].num_owned_vertices;
                     ++v) {
                    const ElementHandle v_handle(static_cast<uint32_t>(p), v);
                    apply(v_handle);
                }
            }
        }

        if ((location & DEVICE) == DEVICE) {
            if constexpr (IS_HD_LAMBDA(LambdaT) || IS_D_LAMBDA(LambdaT)) {
                const int num_patches = this->get_num_patches();
                const int threads     = 256;
                detail::for_each_vertex<<<num_patches, threads, 0, stream>>>(
                    num_patches, this->m_d_patches_info, apply);
            } else {
                zeno::log_error(
                    "RXMeshStatic::for_each_vertex() Input lambda function "
                    "should be annotated with  __device__ for execution on "
                    "device");
            }
        }
    }

    /**
     * @brief Apply a lambda function on all edges in the mesh
     */
    template <typename LambdaT>
    void for_each_edge(locationT    location,
                       LambdaT      apply,
                       cudaStream_t stream = NULL) {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();
#pragma omp parallel for
            for (int p = 0; p < num_patches; ++p) {
                for (uint16_t e = 0;
                     e < this->m_h_patches_info[p].num_owned_edges;
                     ++e) {
                    const ElementHandle e_handle(static_cast<uint32_t>(p), e);
                    apply(e_handle);
                }
            }
        }

        if ((location & DEVICE) == DEVICE) {
            if constexpr (IS_HD_LAMBDA(LambdaT) || IS_D_LAMBDA(LambdaT)) {
                const int num_patches = this->get_num_patches();
                const int threads     = 256;
                detail::for_each_edge<<<num_patches, threads, 0, stream>>>(
                    num_patches, this->m_d_patches_info, apply);
            } else {
                zeno::log_error(
                    "RXMeshStatic::for_each_edge() Input lambda function "
                    "should be annotated with  __device__ for execution on "
                    "device");
            }
        }
    }

    /**
     * @brief Apply a lambda function on all faces in the mesh
     */
    template <typename LambdaT>
    void for_each_face(locationT    location,
                       LambdaT      apply,
                       cudaStream_t stream = NULL) {
        if ((location & HOST) == HOST) {
            const int num_patches = this->get_num_patches();
#pragma omp parallel for
            for (int p = 0; p < num_patches; ++p) {
                for (int f = 0; f < this->m_h_patches_info[p].num_owned_faces;
                     ++f) {
                    const ElementHandle f_handle(static_cast<uint32_t>(p), f);
                    apply(f_handle);
                }
            }
        }

        if ((location & DEVICE) == DEVICE) {
            if constexpr (IS_HD_LAMBDA(LambdaT) || IS_D_LAMBDA(LambdaT)) {
                const int num_patches = this->get_num_patches();
                const int threads     = 256;
                detail::for_each_face<<<num_patches, threads, 0, stream>>>(
                    num_patches, this->m_d_patches_info, apply);
            } else {
                zeno::log_error(
                    "RXMeshStatic::for_each_face() Input lambda function "
                    "should be annotated with  __device__ for execution on "
                    "device");
            }
        }
    }

    /**
     * @brief populate the launch_box with grid size and dynamic shared memory
     * needed for kernel launch
     * @param op List of query operations done inside this the kernel
     * @param launch_box input launch box to be populated
     * @param kernel The kernel to be launched
     * @param oriented if the query is oriented. Valid only for Op::VV queries
     */
    template <uint32_t blockThreads>
    void prepare_launch_box(const std::vector<Op>    op,
                            LaunchBox<blockThreads>& launch_box,
                            const void*              kernel,
                            const bool               oriented = false) const {
        launch_box.blocks         = this->m_num_patches;
        launch_box.smem_bytes_dyn = 0;

        for (auto o : op) {
            launch_box.smem_bytes_dyn = std::max(
                launch_box.smem_bytes_dyn,
                this->template calc_shared_memory<blockThreads>(o, oriented));
        }

        check_shared_memory(launch_box.smem_bytes_dyn,
                            launch_box.smem_bytes_static,
                            launch_box.num_registers_per_thread,
                            kernel);
    }

    template <class T>
    std::shared_ptr<FaceAttribute<T>> add_face_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA) {
        return m_attr_container->template add<FaceAttribute<T>>(
            name.c_str(),
            this->m_h_num_owned_f,
            num_attributes,
            location,
            layout,
            this);
    }

    template <class T>
    std::shared_ptr<FaceAttribute<T>> add_face_attribute(
        const std::vector<std::vector<T>>& f_attributes,
        const std::string&                 name,
        layoutT                            layout = SoA) {
        if (f_attributes.size() != get_num_faces()) {
            zeno::log_error(
                "RXMeshStatic::add_face_attribute() input attribute size ({}) "
                "is not the same as number of faces in the input mesh ({})",
                f_attributes.size(),
                get_num_faces());
        }

        uint32_t num_attributes = f_attributes[0].size();

        auto ret = m_attr_container->template add<FaceAttribute<T>>(
            name.c_str(),
            this->m_h_num_owned_f,
            num_attributes,
            LOCATION_ALL,
            layout,
            this);

        const int num_patches = this->get_num_patches();
#pragma omp parallel for
        for (int p = 0; p < num_patches; ++p) {
            for (uint16_t f = 0; f < this->m_h_num_owned_f[p]; ++f) {

                const ElementHandle f_handle(static_cast<uint32_t>(p), f);

                uint32_t global_f = m_h_patches_ltog_f[p][f];

                for (uint32_t a = 0; a < num_attributes; ++a) {
                    (*ret)(f_handle, a) = f_attributes[global_f][a];
                }
            }
        }

        ret->move(rxmesh::HOST, rxmesh::DEVICE);
        return ret;
    }

    template <class T>
    std::shared_ptr<EdgeAttribute<T>> add_edge_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA) {
        return m_attr_container->template add<EdgeAttribute<T>>(
            name.c_str(),
            this->m_h_num_owned_e,
            num_attributes,
            location,
            layout,
            this);
    }

    template <class T>
    std::shared_ptr<VertexAttribute<T>> add_vertex_attribute(
        const std::string& name,
        uint32_t           num_attributes,
        locationT          location = LOCATION_ALL,
        layoutT            layout   = SoA) {
        return m_attr_container->template add<VertexAttribute<T>>(
            name.c_str(),
            this->m_h_num_owned_v,
            num_attributes,
            location,
            layout,
            this);
    }

    template <class T>
    std::shared_ptr<VertexAttribute<T>> add_vertex_attribute(
        const std::vector<std::vector<T>>& v_attributes,
        const std::string&                 name,
        layoutT                            layout = SoA) {
        if (v_attributes.size() != get_num_vertices()) {
            zeno::log_error(
                "RXMeshStatic::add_vertex_attribute() input attribute size "
                "({}) is not the same as number of vertices in the input mesh "
                "({})",
                v_attributes.size(),
                get_num_vertices());
        }

        uint32_t num_attributes = v_attributes[0].size();

        auto ret = m_attr_container->template add<VertexAttribute<T>>(
            name.c_str(),
            this->m_h_num_owned_v,
            num_attributes,
            LOCATION_ALL,
            layout,
            this);

        const int num_patches = this->get_num_patches();
#pragma omp parallel for
        for (int p = 0; p < num_patches; ++p) {
            for (uint16_t v = 0; v < this->m_h_num_owned_v[p]; ++v) {

                const ElementHandle v_handle(static_cast<uint32_t>(p), v);

                uint32_t global_v = m_h_patches_ltog_v[p][v];

                for (uint32_t a = 0; a < num_attributes; ++a) {
                    (*ret)(v_handle, a) = v_attributes[global_v][a];
                }
            }
        }

        ret->move(rxmesh::HOST, rxmesh::DEVICE);
        return ret;
    }

    void remove_attribute(const std::string& name) {
        if (m_attr_container->does_exist(name.c_str())) {
            m_attr_container->remove(name.c_str());
        }
    }

    uint32_t map_to_global_v(const ElementHandle vh) const {
        auto pl = vh.unpack();
        return m_h_patches_ltog_v[pl.first][pl.second];
    }

    uint32_t map_to_global_e(const ElementHandle eh) const {
        auto pl = eh.unpack();
        return m_h_patches_ltog_e[pl.first][pl.second];
    }

    uint32_t map_to_global_f(const ElementHandle fh) const {
        auto pl = fh.unpack();
        return m_h_patches_ltog_f[pl.first][pl.second];
    }

   protected:
    template <uint32_t blockThreads>
    size_t calc_shared_memory(const Op op, const bool oriented = false) const {
        if (op == Op::VV || op == Op::VE) {
            if (2 * this->m_max_edges_per_patch >
                blockThreads * TRANSPOSE_ITEM_PER_THREAD) {
                zeno::log_error(
                    "RXMeshStatic::calc_shared_memory() "
                    "TRANSPOSE_ITEM_PER_THREAD needs "
                    "to be increased.");
            }
        } else if (op == Op::VE || op == Op::EF || op == Op::FF) {
            if (3 * this->m_max_faces_per_patch >
                blockThreads * TRANSPOSE_ITEM_PER_THREAD) {
                zeno::log_error(
                    "RXMeshStatic::calc_shared_memory() "
                    "TRANSPOSE_ITEM_PER_THREAD needs "
                    "to be increased.");
            }
        }

        if (oriented && op != Op::VV) {
            zeno::log_error(
                "RXMeshStatic::calc_shared_memory() Oriented is only "
                "allowed on VV.");
        }

        if (oriented && op == Op::VV && !this->m_is_input_closed) {
            zeno::log_error(
                "RXMeshStatic::calc_shared_memory() Can't generate oriented "
                "output (VV) for input with boundaries");
        }

        size_t dynamic_smem = 0;
                
        if (op == Op::FE) {
            // only FE will be loaded
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            dynamic_smem += this->m_max_not_owned_edges * sizeof(uint32_t);
            dynamic_smem += this->m_max_not_owned_edges * sizeof(uint16_t);
        } else if (op == Op::EV) {
            dynamic_smem = 2 * this->m_max_edges_per_patch * sizeof(uint16_t);
            dynamic_smem += this->m_max_not_owned_vertices * sizeof(uint32_t);
            dynamic_smem += this->m_max_not_owned_vertices * sizeof(uint16_t);
        } else if (op == Op::FV) {
            dynamic_smem = 2 * this->m_max_edges_per_patch * sizeof(uint16_t);
            dynamic_smem += 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            const uint32_t not_owned_v_bytes =
                this->m_max_not_owned_vertices *
                (sizeof(uint16_t) + sizeof(uint32_t));
            const uint32_t edges_bytes =
                2 * this->m_max_edges_per_patch * sizeof(uint16_t);
            if (not_owned_v_bytes > edges_bytes) {
                zeno::log_error("RXMeshStatic::calc_shared_memory() FV query might fail!");
            }
        } else if (op == Op::VE) {
            dynamic_smem = (2 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);
            dynamic_smem += this->m_max_not_owned_edges * sizeof(uint32_t);
            dynamic_smem += this->m_max_not_owned_edges * sizeof(uint16_t);
        } else if (op == Op::EF) {
            dynamic_smem =
                (2 * 3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
                sizeof(uint16_t) + sizeof(uint16_t);
            dynamic_smem += this->m_max_not_owned_faces * sizeof(uint32_t);
            dynamic_smem += this->m_max_not_owned_faces * sizeof(uint16_t);
        } else if (op == Op::VF) {
            dynamic_smem = 3 * this->m_max_faces_per_patch * sizeof(uint16_t);
            dynamic_smem += std::max(3 * this->m_max_faces_per_patch,
                                     2 * this->m_max_edges_per_patch) *
                                sizeof(uint16_t) +
                            sizeof(uint16_t);
            dynamic_smem += this->m_max_not_owned_faces * sizeof(uint32_t);
            dynamic_smem += this->m_max_not_owned_faces * sizeof(uint16_t);
        } else if (op == Op::VV) {
            dynamic_smem =
                (2 * 2 * this->m_max_edges_per_patch) * sizeof(uint16_t);
            dynamic_smem +=
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);
            if (this->m_max_not_owned_vertices *
                    (sizeof(uint16_t) + sizeof(uint32_t)) >
                (2 * this->m_max_edges_per_patch) * sizeof(uint16_t)) {
                zeno::log_error("RXMeshStatic::calc_shared_memory() VV query might fail!");
            }
        } else if (op == Op::FF) {
            dynamic_smem = (3 * this->m_max_faces_per_patch +        // FE
                            2 * (3 * this->m_max_faces_per_patch) +  // EF
                            4 * this->m_max_faces_per_patch) *       // FF
                           sizeof(uint16_t);
        }
        if (op == Op::VV && oriented) {
            dynamic_smem +=
                (3 * this->m_max_faces_per_patch) * sizeof(uint16_t);
        }
        return dynamic_smem;
    }

    void check_shared_memory(const uint32_t smem_bytes_dyn,
                             size_t&        smem_bytes_static,
                             uint32_t&      num_reg_per_thread,
                             const void*    kernel) const {
        cudaFuncAttributes func_attr = cudaFuncAttributes();
        CUDA_ERROR(cudaFuncGetAttributes(&func_attr, kernel));

        smem_bytes_static  = func_attr.sharedSizeBytes;
        num_reg_per_thread = static_cast<uint32_t>(func_attr.numRegs);
        int device_id;
        CUDA_ERROR(cudaGetDevice(&device_id));
        cudaDeviceProp devProp;
        CUDA_ERROR(cudaGetDeviceProperties(&devProp, device_id));

        if (smem_bytes_static + smem_bytes_dyn > devProp.sharedMemPerBlock) {
            zeno::log_error(
                " RXMeshStatic::check_shared_memory() shared memory needed for"
                " input function ({} bytes) exceeds the max shared memory "
                "per block on the current device ({} bytes)",
                smem_bytes_static + smem_bytes_dyn,
                devProp.sharedMemPerBlock);
            exit(EXIT_FAILURE);
        }
    }

    std::shared_ptr<AttributeContainer>     m_attr_container;
    std::shared_ptr<VertexAttribute<float>> m_input_vertex_coordinates;
};
}  // namespace rxmesh