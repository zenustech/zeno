#ifndef __MAPPING_KERNELS_CUH_
#define __MAPPING_KERNELS_CUH_
#include <driver_types.h>

namespace zs {

  template <typename EntryType, typename MarkerType, typename BinaryOp>
  __global__ void mark_boundary(int num, const EntryType *_entries, MarkerType *_marks,
                                BinaryOp op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    /// mark the tails of all segments
    _marks[idx] = idx != num - 1 ? op(_entries[idx], _entries[idx + 1]) : 1;
  }

  template <typename IndexType>
  __global__ void set_inverse(int num, const IndexType *_map, IndexType *_mapInv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    auto mapIdx = _map[idx];
    if (idx == 0 || mapIdx != _map[idx - 1]) {
      _mapInv[mapIdx] = idx;
#if 0
			if (mapIdx < 5)
				printf("%d-th block starts at %d\n", mapIdx, _mapInv[mapIdx]);
#endif
    }
    if (idx == num - 1) {
      _mapInv[mapIdx + 1] = num;
    }
#if 0
		if (idx < 5)
			printf("%d-th particle belongs to block %d\n", idx, mapIdx);
#endif
  }

  template <typename CounterType, typename IndexType>
  __global__ void exclusive_scan_inverse(CounterType num, const IndexType *_map,
                                         IndexType *_mapInv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    auto mapIdx = _map[idx];
    if (mapIdx != _map[idx + 1]) _mapInv[mapIdx] = idx;
  }

  template <typename IndexType>
  __global__ void map_inverse(int num, const IndexType *_map, IndexType *_mapInv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    _mapInv[_map[idx]] = idx;
  }

  template <typename MappingType, typename CounterType>
  __global__ void set_range_inverse(int count, const MappingType *_toPackedRangeMap,
                                    const MappingType *_toRangeMap, CounterType *_numPackedRange,
                                    MappingType *_rangeIds, MappingType *_rangeLeftBound,
                                    MappingType *_rangeRightBound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    auto packedRangeIdx = _toPackedRangeMap[idx];
    auto rangeIdx = _toRangeMap[idx];
    bool lTag = idx == 0 || rangeIdx != _toRangeMap[idx - 1];
    bool rTag = idx == count - 1 || rangeIdx != _toRangeMap[idx + 1];
    /// left bound
    if (lTag) {
      _rangeLeftBound[packedRangeIdx] = idx;
      _rangeIds[packedRangeIdx] = rangeIdx;
#if 0
		if (packedRangeIdx == 0)
			printf("%d-th block st (%d): %d\n", packedRangeIdx, rangeIdx, _rangeLeftBound[packedRangeIdx]);
#endif
    }
    /// right bound
    if (rTag) {
      _rangeRightBound[packedRangeIdx] = idx + 1;
#if 0
		if (packedRangeIdx == 0)
			printf("%d-th block ed: %d\n", packedRangeIdx, _rangeRightBound[packedRangeIdx]);
#endif
    }
    if (idx == count - 1) *_numPackedRange = packedRangeIdx + 1;
  }

}  // namespace zs

#endif