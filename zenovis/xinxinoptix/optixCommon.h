#pragma once
#include "XAS.h"

#include <memory>
#include <new>
#include <type_traits>
#include <utility>

typedef enum GeoChange {
    VoidChange = 0,
    AttrChange = 1 << 0,
    ShapChange = 1 << 1,//and omm change
    TopoChange = ShapChange | AttrChange,
    MateChange = 2 << 1,
    FullChange = 255,
} GeoChange;

template <typename T>
struct CudaPinnedAllocator {
    using value_type = T;
    using is_pinned_host_allocator = std::true_type;

    using pointer = T*;
    using const_pointer = const T*;

    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    CudaPinnedAllocator() = default;
    template <typename U> CudaPinnedAllocator(const CudaPinnedAllocator<U>&) {}

    T* allocate(std::size_t n) {
        T* ptr;
        cudaError_t err = cudaMallocHost((void**)&ptr, n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* p, std::size_t n) {
        cudaFreeHost(p);
    }
    
    template <typename U>
    bool operator==(const CudaPinnedAllocator<U>&) const { return true; }
    template <typename U>
    bool operator!=(const CudaPinnedAllocator<U>&) const { return false; }
};

// A pinned upload remains owned by the host until this event completes. Keep
// the fence per owner so reusing one upload buffer does not drain unrelated
// work on the CUDA stream.
struct CudaUploadFence {
    cudaEvent_t event{};

    CudaUploadFence() = default;
    CudaUploadFence(const CudaUploadFence&) = delete;
    CudaUploadFence& operator=(const CudaUploadFence&) = delete;

    CudaUploadFence(CudaUploadFence&& other) noexcept
        : event(std::exchange(other.event, nullptr)) {}

    CudaUploadFence& operator=(CudaUploadFence&& other) noexcept {
        if (this == &other) return *this;
        waitNoThrow();
        if (event != nullptr) cudaEventDestroy(event);
        event = std::exchange(other.event, nullptr);
        return *this;
    }

    ~CudaUploadFence() {
        waitNoThrow();
        if (event != nullptr) cudaEventDestroy(event);
    }

    void wait() const {
        if (event == nullptr) return;
        const auto status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) {
            CUDA_CHECK(cudaEventSynchronize(event));
        } else {
            CUDA_CHECK(status);
        }
    }

    void record(cudaStream_t stream) {
        if (event == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(event, stream));
    }

private:
    void waitNoThrow() const noexcept {
        if (event != nullptr) cudaEventSynchronize(event);
    }
};

template <template <class> class ALLOC>
struct SceneNodeT {
    xinxinoptix::raii<CUdeviceptr> buffer;
    IASBuildWorkspace buildWorkspace;
    std::vector<OptixInstance, ALLOC<OptixInstance>> hostInstances;
    CudaUploadFence hostInstancesUpload;
    std::vector<OptixTraversableHandle> childHandles;
    std::vector<OptixTraversableHandle> childHandlesScratch;
    // Keep child topology and the scalar build-policy signature so UPDATE
    // cannot cross an incompatible IAS layout or traversable sequence.
    unsigned int buildFlags{};
    OptixTraversableHandle handle{};
    uint32_t count{};
    uint32_t frame{UINT32_MAX};
    uint8_t depth{};
};

// OptiX IAS UPDATE permits instance-field changes, but not a changed sequence
// of child traversable handles. Record the sequence while assembling the IAS.
template <template <class> class ALLOC>
struct InstanceCollectorT {
    SceneNodeT<ALLOC>& node;
    bool childrenUnchanged{true};

    explicit InstanceCollectorT(SceneNodeT<ALLOC>& scene_node) : node(scene_node)
    {
        node.hostInstancesUpload.wait();
        node.hostInstances.clear();
        node.childHandlesScratch.clear();
    }

    void reserveAdditional(size_t additional)
    {
        if (additional == 0) return;
        const size_t required = node.hostInstances.size() + additional;
        reserveForAppend(node.hostInstances, required);
        // Both handle vectors alternate roles at commit(), so keep capacity in
        // both instead of paying the same growth again on the next frame.
        reserveForAppend(node.childHandles, required);
        reserveForAppend(node.childHandlesScratch, required);
    }

    void append(const OptixInstance& instance)
    {
        const size_t index = node.hostInstances.size();
        childrenUnchanged &= index < node.childHandles.size()
            && node.childHandles[index] == instance.traversableHandle;
        node.hostInstances.push_back(instance);
        node.childHandlesScratch.push_back(instance.traversableHandle);
    }

    bool canUpdate() const
    {
        return childrenUnchanged && node.childHandles.size() == node.childHandlesScratch.size();
    }

    void commit()
    {
        node.childHandles.swap(node.childHandlesScratch);
    }

private:
    template <class Vector>
    static void reserveForAppend(Vector& vector, size_t required)
    {
        if (required <= vector.capacity()) return;
        const size_t grown = vector.capacity() + vector.capacity() / 2;
        vector.reserve(std::max(required, grown));
    }
};

template <template <class> class ALLOC>
struct SceneTypes {
    using Node = SceneNodeT<ALLOC>;
    using Collector = InstanceCollectorT<ALLOC>;
};

using DefaultSceneTypes = SceneTypes<std::allocator>;
using SceneNode = DefaultSceneTypes::Node;
using InstanceCollector = DefaultSceneTypes::Collector;
