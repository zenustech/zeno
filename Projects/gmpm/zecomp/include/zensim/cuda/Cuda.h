#pragma once

/// kokkos/core/src/setup/Kokkos_Setup_Cuda.hpp
#if !ZS_ENABLE_CUDA
#  error "ZS_ENABLE_CUDA was not enabled, but Cuda.h was included anyway."
#endif

#if ZS_ENABLE_CUDA && !defined(__CUDACC__)
#  error "ZS_ENABLE_CUDA defined but the compiler is not defining the __CUDACC__ macro as expected"
// Some tooling environments will still function better if we do this here.
#  define __CUDACC__
#endif

// #include <driver_types.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../DynamicLoader.h"
#include "Allocators.cuh"
#include "CudaFunction.cuh"
#include "CudaLaunchConfig.cuh"
#include "zensim/Reflection.h"
#include "zensim/types/Tuple.h"

namespace zs {

  class Cuda : public Singleton<Cuda> {
  public:
    Cuda();
    ~Cuda();

    /// kernel launching
    enum class StreamIndex { Compute = 0, H2DCopy, D2HCopy, D2DCopy, Spare, Total = 32 };
    enum class EventIndex { Compute = 0, H2DCopy, D2HCopy, D2DCopy, Spare, Total = 32 };

    static auto &driver() { return instance(); }
    static auto &context(int devid) { return driver().contexts[devid]; }
    static auto alignment() { return driver().textureAlignment; }

    struct CudaContext {
      auto &driver() const noexcept { return Cuda::driver(); }
      CudaContext(int devId = 0, int device = 0, void *contextIn = nullptr)
          : devid{devId}, dev{device}, context{contextIn} {}
      auto getDevId() const noexcept { return devid; }
      auto getDevice() const noexcept { return dev; }
      auto getContext() const noexcept { return context; }

      /// only use after Cuda system initialization
      void setContext() const;
      /// stream & event
      // stream
      template <StreamIndex sid> auto stream() const {
        return streams[static_cast<unsigned int>(sid)];
      }
      auto stream(unsigned sid) const { return streams[sid]; }
      auto streamCompute() const {
        return streams[static_cast<unsigned int>(StreamIndex::Compute)];
      }
      auto streamSpare(unsigned sid = 0) const {
        return streams[static_cast<unsigned int>(StreamIndex::Spare) + sid];
      }
      
      // event
      auto eventCompute() const { return events[static_cast<unsigned int>(EventIndex::Compute)]; }
      auto eventSpare(unsigned eid = 0) const {
        return events[static_cast<unsigned int>(EventIndex::Spare) + eid];
      }

      // record
      void recordEventCompute();
      void recordEventSpare(unsigned id = 0);
      // sync
      void syncStream(unsigned sid) const;
      void syncCompute() const;
      template <StreamIndex sid> void syncStream() const { syncStream(stream<sid>()); }
      void syncStreamSpare(unsigned sid = 0) const;
      // stream-event sync
      void computeStreamWaitForEvent(void *event);
      void spareStreamWaitForEvent(unsigned sid, void *event);

      /// kernel launch
      // kernel execution
      void checkError() const;
      void launchKernel(const void *f, unsigned int gx, unsigned int gy, unsigned int gz, 
        unsigned int bx, unsigned int by, unsigned int bz, void **args, std::size_t shmem, void *stream) const;
      void launchCooperativeKernel(const void *f, unsigned int gx, unsigned int gy, unsigned int gz, 
        unsigned int bx, unsigned int by, unsigned int bz, void **args, std::size_t shmem, void *stream) const;
      void launchCallback(void *stream, void *f, void *data) const;

      template <typename... Arguments> void launchCompute(LaunchConfig &&lc,
                                                          void (*f)(remove_vref_t<Arguments>...),
                                                          const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          void *kernelArgs[] = {(void *)&args...};
          // driver().launch((void *)f, lc.dg, lc.db, kernelArgs, lc.shmem, streamCompute());
          launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, kernelArgs, lc.shmem,
            streamCompute());
          checkError();
        }
      }

      // https://docs.nvidia.com/cuda/archive/10.2/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER
      template <typename... Arguments> void launchSpare(StreamID sid, LaunchConfig &&lc,
                                                        void (*f)(remove_vref_t<Arguments>...),
                                                        const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          void *kernelArgs[] = {(void*)&args...};
#if 0
          // driver api
          driver().launchCuKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, lc.shmem,
                                       streamSpare(sid), kernelArgs, nullptr);
#else
          // f<<<lc.dg, lc.db, lc.shmem, (cudaStream_t)streamSpare(sid)>>>(args...);
          launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, kernelArgs, lc.shmem, 
            streamSpare(sid));
#endif
          checkError();
        }
      }

      template <typename... Arguments> void launch(void *stream, LaunchConfig &&lc,
                                                   void (*f)(remove_vref_t<Arguments>...),
                                                   const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          // f<<<lc.dg, lc.db, lc.shmem, (cudaStream_t)stream>>>(args...);
          void *kernelArgs[] = {(void *)&args...};
          launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, kernelArgs, lc.shmem, stream);
          checkError();
        }
      }

      /// allocator initialization on use
      void initDeviceMemory();
      void initUnifiedMemory();

      auto borrow(std::size_t bytes) -> void *;
      void resetMem();

      auto borrowVirtual(std::size_t bytes) -> void *;
      void resetVirtualMem();

    public:
      int devid;
      int dev;                      ///< CUdevice (4 bytes)
      void *context;                ///< CUcontext
      std::vector<void *> streams;  ///< CUstream
      std::vector<void *> events;   ///< CUevents
      bool supportConcurrentUmAccess;
      std::unique_ptr<MonotonicAllocator> deviceMem;
      std::unique_ptr<MonotonicVirtualAllocator> unifiedMem;
    };  //< [end] struct CudaContext

#define PER_CUDA_FUNCTION(name, symbol_name, ...) CudaDriverApi<__VA_ARGS__> name;
#include "cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

#if 0
#define PER_CUDA_FUNCTION(name, symbol_name, ...) CudaRuntimeApi<__VA_ARGS__> name;
#include "cuda_runtime_functions.inc.h"
#undef PER_CUDA_FUNCTION
#endif
    void (*get_cu_error_name)(uint32_t, const char **);
    void (*get_cu_error_string)(uint32_t, const char **);
    //const char *(*get_cuda_error_name)(uint32_t);
    //const char *(*get_cuda_error_string)(uint32_t);

  private:
    int numTotalDevice;

    std::vector<CudaContext> contexts;  ///< generally one per device
    int textureAlignment;
    std::unique_ptr<DynamicLoader> driverLoader, runtimeLoader;

    int _iDevID;  ///< need changing
  };

}  // namespace zs
