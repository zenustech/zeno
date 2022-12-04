#pragma once

#include <utility>
#include <cstring>
#include <zeno/utils/log.h>
#include "handle.h"
#include "rxmesh.h"
#include "kernels/attribute.cuh"
#include "utils/util.cuh"

namespace zeno::rxmesh {
/**
 * @brief Base untyped attributes used as an interface for attribute container
 */
class AttributeBase {
   public:
    AttributeBase() = default;

    virtual const char* get_name() const = 0;

    virtual void release(locationT location = LOCATION_ALL) = 0;

    virtual ~AttributeBase() = default;
};

/**
 * @brief  Manages the attributes attached to element on top of the mesh.
 */
template <class T>
class Attribute : public AttributeBase {

   public:
    Attribute()
        : AttributeBase(),
          m_name(nullptr),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_h_ptr_on_device(nullptr),
          m_d_attr(nullptr),
          m_num_patches(0),
          m_d_element_per_patch(nullptr),
          m_h_element_per_patch(nullptr),
          m_layout(AoS) {

        this->m_name    = (char*)malloc(sizeof(char) * 1);
        this->m_name[0] = '\0';
    }

    Attribute(const char* name)
        : AttributeBase(),
          m_name(nullptr),
          m_num_attributes(0),
          m_allocated(LOCATION_NONE),
          m_h_attr(nullptr),
          m_h_ptr_on_device(nullptr),
          m_d_attr(nullptr),
          m_num_patches(0),
          m_d_element_per_patch(nullptr),
          m_h_element_per_patch(nullptr),
          m_layout(AoS) {
        if (name != nullptr) {
            this->m_name = (char*)malloc(sizeof(char) * (strlen(name) + 1));
            strcpy(this->m_name, name);
        }
    }

    Attribute(const Attribute& rhs) = default;

    virtual ~Attribute() = default;

    const char* get_name() const {
        return m_name;
    }

    __host__ __device__ __forceinline__ uint32_t get_num_attributes() const {
        return this->m_num_attributes;
    }

    __host__ __device__ __forceinline__ locationT get_allocated() const {
        return this->m_allocated;
    }

    __host__ __device__ __forceinline__ bool is_device_allocated() const {
        return ((m_allocated & DEVICE) == DEVICE);
    }

    __host__ __device__ __forceinline__ bool is_host_allocated() const {
        return ((m_allocated & HOST) == HOST);
    }

    /**
     * @brief Reset attribute to certain value
     */
    void reset(const T value, locationT location, cudaStream_t stream = NULL) {
        if ((location & DEVICE) == DEVICE) {

            assert((m_allocated & DEVICE) == DEVICE);

            const int threads = 256;
            zeno::rxmesh::template memset_attribute<T>
                <<<m_num_patches, threads, 0, stream>>>(*this,
                                                        value,
                                                        m_d_element_per_patch,
                                                        m_num_patches,
                                                        m_num_attributes);
        }

        if ((location & HOST) == HOST) {
            assert((m_allocated & HOST) == HOST);
#pragma omp parallel for
            for (int p = 0; p < static_cast<int>(m_num_patches); ++p) {
                for (int e = 0; e < m_h_element_per_patch[p]; ++e) {
                    m_h_attr[p][e] = value;
                }
            }
        }
    }

    /**
     * @brief Allocate memory for attribute. Meant to be used by RXMeshStatic.
     */
    void init(const std::vector<uint16_t>& element_per_patch,
              const uint32_t               num_attributes,
              locationT                    location = LOCATION_ALL,
              const layoutT                layout   = AoS) {
        release();
        m_num_patches    = element_per_patch.size();
        m_num_attributes = num_attributes;
        m_layout         = layout;

        if (m_num_patches == 0) {
            return;
        }

        allocate(element_per_patch.data(), location);
    }

    /**
     * @brief Copy memory from one location to another. If target is not
     * allocated, it will be allocated first before copying the memory.
     * TODO it is better to launch a kernel that do the memcpy than relying on
     * the host API from CUDA since all these small memcpy will be enqueued in
     * the same stream and so serialized
     */
    void move(locationT source, locationT target, cudaStream_t stream = NULL) {
        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            zeno::log_error(
                "Attribute::move() moving source is not valid"
                " because it was not allocated on source i.e., {}",
                source);
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & m_allocated) != target)) {
            allocate(m_h_element_per_patch, target);
        }

        if (this->m_num_patches == 0) {
            return;
        }

        if (source == HOST && target == DEVICE) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    m_h_attr[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
        } else if (source == DEVICE && target == HOST) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_attr[p],
                    m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyDeviceToHost,
                    stream));
            }
        }
    }

    /**
     * @brief Release allocated memory in certain location
     */
    void release(locationT location = LOCATION_ALL) {
        if (((location & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                free(m_h_attr[p]);
            }
            free(m_h_attr);
            m_h_attr = nullptr;
            free(m_h_element_per_patch);
            m_h_element_per_patch = nullptr;
            m_allocated           = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            for (uint32_t p = 0; p < m_num_patches; ++p) {
                GPU_FREE(m_h_ptr_on_device[p]);
            }
            GPU_FREE(m_d_attr);
            GPU_FREE(m_d_element_per_patch);
            m_allocated = m_allocated & (~DEVICE);
        }
    }

    /**
     * @brief Deep copy from a source attribute. If source_flag and dst_flag are
     * both set to LOCATION_ALL, then we copy what is on host to host, and what
     * on device to device. If sourc_flag is set to HOST (or DEVICE) and
     * dst_flag is set to LOCATION_ALL, then we copy source's HOST (or
     * DEVICE) to both HOST and DEVICE. Setting source_flag to
     * LOCATION_ALL while dst_flag is NOT set to LOCATION_ALL is invalid
     * because we don't know which source to copy from
     */
    void copy_from(Attribute<T>& source,
                   locationT     source_flag,
                   locationT     dst_flag,
                   cudaStream_t  stream = NULL) {
        if (source.m_layout != m_layout) {
            zeno::log_error(
                "Attribute::copy_from() does not support copy from "
                "source of different layout!");
        }

        if ((source_flag & LOCATION_ALL) == LOCATION_ALL &&
            (dst_flag & LOCATION_ALL) != LOCATION_ALL) {
            zeno::log_error("Attribute::copy_from() Invalid configuration!");
        }

        if (m_num_attributes != source.get_num_attributes()) {
            zeno::log_error(
                "Attribute::copy_from() number of attributes is "
                "different!");
        }

        if (this->is_empty() || this->m_num_patches == 0) {
            return;
        }

        // 1) copy from HOST to HOST
        if ((source_flag & HOST) == HOST && (dst_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                zeno::log_error(
                    "Attribute::copy() copying src is not allocated on host");
            }
            if ((dst_flag & m_allocated) != dst_flag) {
                zeno::log_error(
                    "Attribute::copy() copying dst is not allocated on host");
            }

            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                std::memcpy(
                    m_h_ptr_on_device[p],
                    source.m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes);
            }
        }


        // 2) copy from DEVICE to DEVICE
        if ((source_flag & DEVICE) == DEVICE && (dst_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                zeno::log_error(
                    "Attribute::copy() copying src is not allocated on device");
            }
            if ((dst_flag & m_allocated) != dst_flag) {
                zeno::log_error(
                    "Attribute::copy() copying dst is not allocated on device");
            }

            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    source.m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }

        // 3) copy from DEVICE to HOST
        if ((source_flag & DEVICE) == DEVICE && (dst_flag & HOST) == HOST) {
            if ((source_flag & source.m_allocated) != source_flag) {
                zeno::log_error(
                    "Attribute::copy() copying src is not allocated on device");
            }
            if ((dst_flag & m_allocated) != dst_flag) {
                zeno::log_error(
                    "Attribute::copy() copying dst is not allocated on host");
            }

            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_attr[p],
                    source.m_h_ptr_on_device[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyDeviceToHost,
                    stream));
            }
        }


        // 4) copy from HOST to DEVICE
        if ((source_flag & HOST) == HOST && (dst_flag & DEVICE) == DEVICE) {
            if ((source_flag & source.m_allocated) != source_flag) {
                zeno::log_error(
                    "Attribute::copy() copying src is not allocated on host");
            }
            if ((dst_flag & m_allocated) != dst_flag) {
                zeno::log_error(
                    "Attribute::copy() copying dst is not allocated on device");
            }


            for (uint32_t p = 0; p < m_num_patches; ++p) {
                assert(m_h_element_per_patch[p] ==
                       source.m_h_element_per_patch[p]);
                CUDA_ERROR(cudaMemcpyAsync(
                    m_h_ptr_on_device[p],
                    source.m_h_attr[p],
                    sizeof(T) * m_h_element_per_patch[p] * m_num_attributes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
        }
    }

    /**
     * @brief Access the attribute value using patch and local index in the
     * patch. This is meant to be used by XXAttribute not directly by the user
     */
    __host__ __device__ __forceinline__ T& operator()(const uint32_t patch_id,
                                                      const uint16_t local_id,
                                                      const uint32_t attr) const {
        assert(patch_id < m_num_patches);
        assert(attr < m_num_attributes);

        const uint32_t pitch_x = (m_layout == AoS) ? m_num_attributes : 1;
#ifdef __CUDA_ARCH__
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_d_element_per_patch[patch_id];
        return m_d_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#else
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_h_element_per_patch[patch_id];
        return m_h_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#endif
    }

    /**
     * @brief Access the attribute value using patch and local index in the
     * patch. This is meant to be used by XXAttribute not directly by the user
     */
    __host__ __device__ __forceinline__ T& operator()(const uint32_t patch_id,
                                                      const uint16_t local_id,
                                                      const uint32_t attr) {
        assert(patch_id < m_num_patches);
        assert(attr < m_num_attributes);

        const uint32_t pitch_x = (m_layout == AoS) ? m_num_attributes : 1;
#ifdef __CUDA_ARCH__
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_d_element_per_patch[patch_id];
        return m_d_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#else
        const uint32_t pitch_y =
            (m_layout == AoS) ? 1 : m_h_element_per_patch[patch_id];
        return m_h_attr[patch_id][local_id * pitch_x + attr * pitch_y];
#endif
    }

    __host__ __device__ __forceinline__ bool is_empty() const {
        return m_num_patches == 0;
    }


   private:
    void allocate(const uint16_t* element_per_patch, locationT location) {

        if (m_num_patches != 0) {

            if ((location & HOST) == HOST) {
                release(HOST);
                m_h_element_per_patch = static_cast<uint16_t*>(
                    malloc(sizeof(uint16_t) * m_num_patches));

                m_h_attr = static_cast<T**>(malloc(sizeof(T*) * m_num_patches));

                std::memcpy(m_h_element_per_patch,
                            element_per_patch,
                            sizeof(uint16_t) * m_num_patches);

                for (uint32_t p = 0; p < m_num_patches; ++p) {
                    m_h_attr[p] = static_cast<T*>(malloc(
                        sizeof(T) * element_per_patch[p] * m_num_attributes));
                }

                m_allocated = m_allocated | HOST;
            }

            if ((location & DEVICE) == DEVICE) {
                release(DEVICE);

                m_h_element_per_patch = static_cast<uint16_t*>(
                    malloc(sizeof(uint16_t) * m_num_patches));

                std::memcpy(m_h_element_per_patch,
                            element_per_patch,
                            sizeof(uint16_t) * m_num_patches);

                CUDA_ERROR(cudaMalloc((void**)&(m_d_element_per_patch),
                                      sizeof(uint16_t) * m_num_patches));


                CUDA_ERROR(cudaMalloc((void**)&(m_d_attr),
                                      sizeof(T*) * m_num_patches));
                m_h_ptr_on_device =
                    static_cast<T**>(malloc(sizeof(T*) * m_num_patches));

                CUDA_ERROR(cudaMemcpy(m_d_element_per_patch,
                                      element_per_patch,
                                      sizeof(uint16_t) * m_num_patches,
                                      cudaMemcpyHostToDevice));

                for (uint32_t p = 0; p < m_num_patches; ++p) {
                    CUDA_ERROR(cudaMalloc((void**)&(m_h_ptr_on_device[p]),
                                          sizeof(T) * m_h_element_per_patch[p] *
                                              m_num_attributes));
                }
                CUDA_ERROR(cudaMemcpy(m_d_attr,
                                      m_h_ptr_on_device,
                                      sizeof(T*) * m_num_patches,
                                      cudaMemcpyHostToDevice));
                m_allocated = m_allocated | DEVICE;
            }
        }
    }

    char*     m_name;
    uint32_t  m_num_attributes;
    locationT m_allocated;
    T**       m_h_attr;
    T**       m_h_ptr_on_device;
    T**       m_d_attr;
    uint32_t  m_num_patches;
    uint16_t* m_d_element_per_patch;
    uint16_t* m_h_element_per_patch;
    layoutT   m_layout;
};

template <class T>
class FaceAttribute : public Attribute<T> {
   public:
    FaceAttribute() = default;

    FaceAttribute(const char*                  name,
                  const std::vector<uint16_t>& face_per_patch,
                  const uint32_t               num_attributes,
                  locationT                    location,
                  const layoutT                layout,
                  const RXMesh*                rxmesh)
        : Attribute<T>(name), m_rxmesh(rxmesh) {
        this->init(face_per_patch, num_attributes, location, layout);
    }

    __host__ __device__ __forceinline__ T& operator()(const ElementHandle f_handle,
                                                      const uint32_t   attr = 0) const {
        auto                 pl = f_handle.unpack();
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

    __host__ __device__ __forceinline__ T& operator()(const ElementHandle f_handle,
                                                      const uint32_t   attr = 0) {
        auto                 pl = f_handle.unpack();
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

   private:
    const RXMesh* m_rxmesh;
};

template <class T>
class EdgeAttribute : public Attribute<T> {
   public:
    EdgeAttribute() = default;

    EdgeAttribute(const char*                  name,
                  const std::vector<uint16_t>& edge_per_patch,
                  const uint32_t               num_attributes,
                  locationT                    location,
                  const layoutT                layout,
                  const RXMesh*                rxmesh)
        : Attribute<T>(name), m_rxmesh(rxmesh) {
        this->init(edge_per_patch, num_attributes, location, layout);
    }

    __host__ __device__ __forceinline__ T& operator()(const ElementHandle e_handle,
                                                      const uint32_t   attr = 0) const {
        auto                 pl = e_handle.unpack();
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }
    __host__ __device__ __forceinline__ T& operator()(const ElementHandle e_handle,
                                                      const uint32_t   attr = 0) {
        auto                 pl = e_handle.unpack();
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

   private:
    const RXMesh* m_rxmesh;
};

template <class T>
class VertexAttribute : public Attribute<T> {
   public:
    VertexAttribute() = default;

    VertexAttribute(const char*                  name,
                    const std::vector<uint16_t>& vertex_per_patch,
                    const uint32_t               num_attributes,
                    locationT                    location,
                    const layoutT                layout,
                    const RXMesh*                rxmesh)
        : Attribute<T>(name), m_rxmesh(rxmesh) {
        this->init(vertex_per_patch, num_attributes, location, layout);
    }

    __host__ __device__ __forceinline__ T& operator()(const ElementHandle v_handle,
                                                      const uint32_t     attr = 0) const {
        auto                 pl = v_handle.unpack();
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }
    __host__ __device__ __forceinline__ T& operator()(const ElementHandle v_handle,
                                                      const uint32_t     attr = 0) {
        auto                 pl = v_handle.unpack();
        return Attribute<T>::operator()(pl.first, pl.second, attr);
    }

   private:
    const RXMesh* m_rxmesh;
};

/**
 * @brief Manages a collection of attributes by RXMeshStatic
 */
class AttributeContainer {
   public:
    AttributeContainer() = default;

    virtual ~AttributeContainer() {
        while (!m_attr_container.empty()) {
            m_attr_container.back()->release();
            m_attr_container.pop_back();
        }
    }

    /**
     * @brief add a new attribute to be managed by this container
     */
    template <typename AttrT>
    std::shared_ptr<AttrT> add(const char*            name,
                               std::vector<uint16_t>& element_per_patch,
                               uint32_t               num_attributes,
                               locationT              location,
                               layoutT                layout,
                               const RXMesh*          rxmesh) {
        if (does_exist(name)) {
            zeno::log_warn(
                "AttributeContainer::add() adding an attribute with "
                "name {} already exists!",
                std::string(name));
        }

        auto new_attr = std::make_shared<AttrT>(
            name, element_per_patch, num_attributes, location, layout, rxmesh);
        m_attr_container.push_back(
            std::dynamic_pointer_cast<AttributeBase>(new_attr));

        return new_attr;
    }

    bool does_exist(const char* name) {
        for (size_t i = 0; i < m_attr_container.size(); ++i) {
            if (!strcmp(m_attr_container[i]->get_name(), name)) {
                return true;
            }
        }
        return false;
    }

    void remove(const char* name) {
        for (auto it = m_attr_container.begin(); it != m_attr_container.end(); ++it) {
            if (!strcmp((*it)->get_name(), name)) {
                (*it)->release(LOCATION_ALL);
                m_attr_container.erase(it);
                break;
            }
        }
    }

   private:
    std::vector<std::shared_ptr<AttributeBase>> m_attr_container;
};
}  // namespace rxmesh