#pragma once
#include "XAS.h"
#include <optional>

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