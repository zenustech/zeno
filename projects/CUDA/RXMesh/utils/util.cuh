#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace zeno::rxmesh {
    
template <typename attrT>
__global__ void memcpy(attrT* d_dest, const attrT* d_src, const uint32_t length) {
    const uint32_t stride = blockDim.x * gridDim.x;
    uint32_t       i      = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length) {
        d_dest[i] = d_src[i];
        i += stride;
    }
}


template <typename attrT>
__global__ void memset(attrT* d_dest, const attrT val, const uint32_t length) {
    const uint32_t stride = blockDim.x * gridDim.x;
    uint32_t       i      = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length) {
        d_dest[i] = val;
        i += stride;
    }
}

__device__ __forceinline__ uint16_t atomicAdd(uint16_t* address, uint16_t val) {
    // https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh#L36
    size_t    offset        = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool      is_32_align   = offset;
    uint32_t  old           = *address_as_ui;
    uint32_t  old_bytes;
    uint32_t  newval;
    uint32_t  assumed;

    do {
        assumed   = old;
        old_bytes = is_32_align ? old >> 16 : old & 0xffff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint16_t>(val + old_bytes);
        newval = is_32_align ? (old & 0xffff) | (newval << 16) :
                               (old & 0xffff0000) | newval;
        old    = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
    return (is_32_align) ? uint16_t(old >> 16) : uint16_t(old & 0xffff);
}


__device__ __forceinline__ uint8_t atomicAdd(uint8_t* address, uint8_t val) {
    // https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh#L14
    size_t    offset        = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t  old           = *address_as_ui;
    uint32_t  shift         = offset * 8;
    uint32_t  old_byte;
    uint32_t  newval;
    uint32_t  assumed;

    do {
        assumed  = old;
        old_byte = (old >> shift) & 0xff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint8_t>(val + old_byte);
        newval = (old & ~(0x000000ff << shift)) | (newval << shift);
        old    = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);

    return uint8_t((old >> shift) & 0xff);
}


/**
 * atomicCAS() on unsigned short int for SM < 7.0
 */
__device__ __forceinline__ unsigned short int atomicCAS(
    unsigned short int* address,
    unsigned short int  compare,
    unsigned short int  val) {
#if __CUDA_ARCH__ >= 700
    return ::atomicCAS(address, compare, val);
#else
    // https://github.com/rapidsai/cudf/blob/89b802e6cecffe2425048f1f70cd682b865730b8/cpp/include/cudf/detail/utilities/device_atomics.cuh
    using T_int       = unsigned int;
    using T_int_short = unsigned short int;

    bool   is_32_align = (reinterpret_cast<size_t>(address) & 2) ? false : true;
    T_int* address_uint32 = reinterpret_cast<T_int*>(
        reinterpret_cast<size_t>(address) - (is_32_align ? 0 : 2));

    T_int       old = *address_uint32;
    T_int       assumed;
    T_int_short target_value;
    uint16_t    u_val = *(reinterpret_cast<uint16_t*>(&val));

    do {
        assumed = old;
        target_value =
            (is_32_align) ? T_int_short(old & 0xffff) : T_int_short(old >> 16);
        if (target_value != compare)
            break;

        T_int new_value = (is_32_align) ? (old & 0xffff0000) | u_val :
                                          (old & 0xffff) | (T_int(u_val) << 16);
        old             = ::atomicCAS(address_uint32, assumed, new_value);
    } while (assumed != old);

    return target_value;

#endif
}
}  // namespace rxmesh