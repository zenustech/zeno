#include "Allocators.cuh"
#include "Cuda.h"

namespace zs {

  /// customized memory allocators
  MonotonicAllocator::MonotonicAllocator(std::size_t totalMemBytes, std::size_t textureAlignBytes)
      : stack_allocator{&raw_allocator<device_mem_tag>::instance(), totalMemBytes,
                        textureAlignBytes} {
    printf("device memory allocator alignment: %llu bytes\tsize: %llu MB\n", (u64)textureAlignBytes,
           (u64)(totalMemBytes / 1024.0 / 1024.0));
  }
  auto MonotonicAllocator::borrow(std::size_t bytes) -> void * { return allocate(bytes); }
  void MonotonicAllocator::reset() {
    std::size_t usedBytes = _head - _data;
    std::size_t totalBytes = _tail - _data;
    if (usedBytes >= totalBytes * 3 / 4) {
      this->resource()->deallocate((void *)this->_data, totalBytes);
      std::size_t totalMemBytes = totalBytes * 3 / 2;
      this->_data = (char *)(this->resource()->allocate(totalMemBytes));
      this->_head = this->_data;
      this->_tail = this->_data + totalMemBytes;
    } else {
      this->_head = this->_data;
    }
  }

  MonotonicVirtualAllocator::MonotonicVirtualAllocator(int devId, std::size_t totalMemBytes,
                                                       std::size_t textureAlignBytes)
      : stack_allocator{&raw_allocator<um_mem_tag>::instance(), totalMemBytes, textureAlignBytes} {
    if (Cuda::context(devId).supportConcurrentUmAccess)
      cudaMemAdvise(_data, totalMemBytes, cudaMemAdviseSetPreferredLocation, devId);
    printf("unified memory allocator alignment: %llu bytes\tsize: %llu MB\n",
           (u64)textureAlignBytes, (u64)(totalMemBytes / 1024.0 / 1024.0));
  }
  auto MonotonicVirtualAllocator::borrow(std::size_t bytes) -> void * { return allocate(bytes); }
  void MonotonicVirtualAllocator::reset() {
    std::size_t usedBytes = _head - _data;
    std::size_t totalBytes = _tail - _data;
    if (usedBytes >= totalBytes * 3 / 4) {
      this->resource()->deallocate((void *)this->_data, totalBytes);
      std::size_t totalMemBytes = totalBytes * 3 / 2;
      this->_data = (char *)(this->resource()->allocate(totalMemBytes));
      this->_head = this->_data;
      this->_tail = this->_data + totalMemBytes;
    } else {
      this->_head = this->_data;
    }
  }

}  // namespace zs
