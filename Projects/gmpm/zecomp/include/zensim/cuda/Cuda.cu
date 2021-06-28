#include <utility>

#include "../Platform.hpp"
#include "Cuda.h"
#include "CudaConstants.inc"
#include "zensim/tpls/fmt/format.h"

#define MEM_POOL_CTRL 3

namespace zs {

  std::string get_cu_error_message(uint32_t err) {
    const char *err_name_ptr;
    const char *err_string_ptr;
    Cuda::instance().get_cu_error_name(err, &err_name_ptr);
    Cuda::instance().get_cu_error_string(err, &err_string_ptr);
    return fmt::format("CUDA Driver Error {}: {}", err_name_ptr, err_string_ptr);
  }

#if 0
  std::string get_cuda_error_message(uint32_t err) {
    return fmt::format("CUDA Runtime Error {}: {}", Cuda::instance().get_cuda_error_name(err),
                       Cuda::instance().get_cuda_error_string(err));
  }
#endif

  // record
  void Cuda::CudaContext::recordEventCompute() {
    cudaEventRecord((cudaEvent_t)eventCompute(), (cudaStream_t)streamCompute());
  }
  void Cuda::CudaContext::recordEventSpare(unsigned id) {
    cudaEventRecord((cudaEvent_t)eventSpare(id), (cudaStream_t)streamSpare(id));
  }
  // sync
  void Cuda::CudaContext::syncStream(unsigned sid) const {
    cudaStreamSynchronize((cudaStream_t)stream(sid));
  }
  void Cuda::CudaContext::syncCompute() const {
    cudaStreamSynchronize((cudaStream_t)streamCompute());
  }
  void Cuda::CudaContext::syncStreamSpare(unsigned sid) const {
    cudaStreamSynchronize((cudaStream_t)streamSpare(sid));
  }
  // stream-event sync
  void Cuda::CudaContext::computeStreamWaitForEvent(void *event) {
    cudaStreamWaitEvent((cudaStream_t)streamCompute(), (cudaEvent_t)event, 0);
  }
  void Cuda::CudaContext::spareStreamWaitForEvent(unsigned sid, void *event) {
    cudaStreamWaitEvent((cudaStream_t)streamSpare(sid), (cudaEvent_t)event, 0);
  }

  void Cuda::CudaContext::checkError() const {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      fmt::print("Last Error on [Dev {}]: {}\n", devid, cudaGetErrorString(error));
  }
  void Cuda::CudaContext::launchKernel(const void *f, unsigned int gx, unsigned int gy,
                                       unsigned int gz, unsigned int bx, unsigned int by,
                                       unsigned int bz, void **args, std::size_t shmem,
                                       void *stream) const {
    cudaLaunchKernel(f, dim3{gx, gy, gz}, dim3{bx, by, bz}, args, shmem, (cudaStream_t)stream);
  }
  void Cuda::CudaContext::launchCooperativeKernel(const void *f, unsigned int gx, unsigned int gy,
                                                  unsigned int gz, unsigned int bx, unsigned int by,
                                                  unsigned int bz, void **args, std::size_t shmem,
                                                  void *stream) const {
    cudaLaunchCooperativeKernel(f, dim3{gx, gy, gz}, dim3{bx, by, bz}, args, shmem,
                                (cudaStream_t)stream);
  }
  void Cuda::CudaContext::launchCallback(void *stream, void *f, void *data) const {
    cudaLaunchHostFunc((cudaStream_t)stream, (cudaHostFn_t)f, data);
  }
  void Cuda::CudaContext::setContext() const { cudaSetDevice(devid); }

  Cuda::Cuda() {
    fmt::print("[Init -- Begin] Cuda\n");

    {  // cuda driver api (for JIT)
#if defined(ZS_PLATFORM_LINUX)
      driverLoader = std::make_unique<DynamicLoader>("libcuda.so.1");
#elif defined(ZS_PLATFORM_WINDOWS)
      driverLoader = std::make_unique<DynamicLoader>("nvcuda.dll");
#else
      static_assert(false, "CUDA driver supports only Windows and Linux.");
#endif
      driverLoader->load_function("cuGetErrorName", get_cu_error_name);
      driverLoader->load_function("cuGetErrorString", get_cu_error_string);

#define PER_CUDA_FUNCTION(name, symbol_name, ...)      \
  name.set(driverLoader->load_function(#symbol_name)); \
  name.set_names(#name, #symbol_name);
#include "cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION
    }

    init(0);

#if 0
    { // cuda runtime api
#  if defined(ZS_PLATFORM_LINUX)
      runtimeLoader.reset(new DynamicLoader("libcudart.so"));
#  elif defined(ZS_PLATFORM_WINDOWS)
      int version{0};
      getDriverVersion(&version);
      auto suf = std::to_string(version / 100);
      auto cudaDllName = std::string("cudart64_") + suf + ".dll";
      fmt::print("loading cuda runtime dll: {}\n", cudaDllName);
      runtimeLoader.reset(new DynamicLoader(cudaDllName.c_str())); //nvcudart.dll"));
#  else
      static_assert(false, "CUDA library supports only Windows and Linux.");
#  endif
      runtimeLoader->load_function("cudaGetErrorName", get_cuda_error_name);
      runtimeLoader->load_function("cudaGetErrorString", get_cuda_error_string);

#  define PER_CUDA_FUNCTION(name, symbol_name, ...)       \
    name.set(runtimeLoader->load_function(#symbol_name)); \
    name.set_names(#name, #symbol_name);
#  include "cuda_runtime_functions.inc.h"
#  undef PER_CUDA_FUNCTION
    }
#endif

    numTotalDevice = 0;
    getDeviceCount(&numTotalDevice);
    if (numTotalDevice == 0)
      fmt::print(
          "\t[InitInfo -- DevNum] There are no available device(s) that "
          "support CUDA\n");
    else
      fmt::print("\t[InitInfo -- DevNum] Detected {} CUDA Capable device(s)\n", numTotalDevice);

    contexts.resize(numTotalDevice);
    for (int i = 0; i < numTotalDevice; i++) {
      auto &context = contexts[i];
      int dev{};
      {
        void *c{nullptr};
        cudaSetDevice(i);
        getDevice(&dev, i);
        // fmt::print("device ordinal {} is {}\n", i, dev);

        // getContext(&c);
        retainPrimaryCtx(&c, dev);
        // createContext(&c, 4, dev); // CU_CTX_SCHED_BLOCKING_SYNC (0x04) | CU_CTX_SCHED_SPIN
        // (0x01)
        context = CudaContext{i, dev, c};
        // setContext(context.getContext());
      }

      context.streams.resize((int)StreamIndex::Total);
      for (auto &stream : context.streams)
        cudaStreamCreateWithFlags((cudaStream_t *)&stream, cudaStreamNonBlocking);
      context.events.resize((int)EventIndex::Total);
      for (auto &event : context.events)
        cudaEventCreateWithFlags((cudaEvent_t *)&event, cudaEventBlockingSync);

      /// device properties
      int major, minor, multiGpuBoardGroupID, multiProcessorCount, sharedMemPerBlock, regsPerBlock;
      int supportUnifiedAddressing, supportUm, supportConcurrentUmAccess;
      getDeviceAttribute(&sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
      getDeviceAttribute(&regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
      getDeviceAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
      getDeviceAttribute(&multiGpuBoardGroupID, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, dev);
      getDeviceAttribute(&textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev);
      getDeviceAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
      getDeviceAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
      getDeviceAttribute(&supportUnifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
      getDeviceAttribute(&supportUm, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev);
      getDeviceAttribute(&supportConcurrentUmAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                         dev);

      context.supportConcurrentUmAccess = supportConcurrentUmAccess;

      fmt::print(
          "\t[InitInfo -- Dev Property] GPU device {} ({}-th group on "
          "board)\n\t\tshared memory per block: {} bytes,\n\t\tregisters per SM: "
          "{},\n\t\tMulti-Processor count: {},\n\t\tSM compute capabilities: "
          "{}.{}.\n\t\tTexture alignment: {} bytes\n\t\tUVM support: allocation({}), unified "
          "addressing({}), concurrent access({})\n",
          i, multiGpuBoardGroupID, sharedMemPerBlock, regsPerBlock, multiProcessorCount, major,
          minor, textureAlignment, supportUm, supportUnifiedAddressing, supportConcurrentUmAccess);
    }

    /// enable peer access if feasible
    for (int i = 0; i < numTotalDevice; i++) {
      // setContext(contexts[i].getContext());
      cudaSetDevice(i);
      for (int j = 0; j < numTotalDevice; j++) {
        if (i != j) {
          int iCanAccessPeer = 0;
          canAccessPeer(&iCanAccessPeer, contexts[i].getDevice(), contexts[j].getDevice());
          if (iCanAccessPeer) enablePeerAccess(contexts[j].getContext(), 0);
          fmt::print("\t[InitInfo -- Peer Access] Peer access status {} -> {}: {}\n", i, j,
                     iCanAccessPeer ? "Inactive" : "Active");
        }
      }
    }
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
    /* GPUs with SM architecture 6.x or higher (Pascal class or newer) provide additional
    Unified Memory features such as on-demand page migration and GPU memory oversubscription
    that are outlined throughout this document. Note that currently these features are only
    supported on Linux operating systems. Applications running on Windows (whether in TCC
    or WDDM mode) will use the basic Unified Memory model as on pre-6.x architectures even
    when they are running on hardware with compute capability 6.x or higher. */

    fmt::print("\n[Init -- End] == Finished \'Cuda\' initialization\n\n");
  }

  Cuda::~Cuda() {
    for (int i = 0; i < numTotalDevice; i++) {
      auto &context = contexts[i];
      context.setContext();
      for (auto stream : context.streams) cudaStreamDestroy((cudaStream_t)stream);
      for (auto event : context.events) cudaEventDestroy((cudaEvent_t)event);
      context.deviceMem.reset(nullptr);
      context.unifiedMem.reset(nullptr);

      // destroyContext(context.getContext());
      cudaDeviceReset();
    }
    fmt::print("  Finished \'Cuda\' termination\n");
  }

  void Cuda::CudaContext::initDeviceMemory() {
    /// memory
    std::size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    deviceMem = std::make_unique<MonotonicAllocator>(free_byte >> MEM_POOL_CTRL,
                                                     driver().textureAlignment);
    fmt::print(
        "\t[InitInfo -- memory] device {}\n\t\tfree bytes/total bytes: "
        "{}/{},\n\t\tpre-allocated device memory: {} bytes\n\n",
        getDevId(), free_byte, total_byte, (free_byte >> MEM_POOL_CTRL));
  }
  void Cuda::CudaContext::initUnifiedMemory() {
#if defined(_WIN32)
    throw std::runtime_error("unified virtual memory manually disabled on windows!");
    return;
#endif
    std::size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    unifiedMem = std::make_unique<MonotonicVirtualAllocator>(getDevId(), total_byte * 4,
                                                             driver().textureAlignment);
    fmt::print(
        "\t[InitInfo -- memory] device {}\n\t\tfree bytes/total bytes: "
        "{}/{},\n\t\tpre-allocated unified memory: {} bytes\n\n",
        getDevId(), free_byte, total_byte, total_byte * 4);
  }

  auto Cuda::CudaContext::borrow(std::size_t bytes) -> void * {
    if (!deviceMem) initDeviceMemory();
    return deviceMem->borrow(bytes);
  }
  void Cuda::CudaContext::resetMem() {
    if (!deviceMem) initDeviceMemory();
    deviceMem->reset();
  }

  auto Cuda::CudaContext::borrowVirtual(std::size_t bytes) -> void * {
#if defined(_WIN32)
    throw std::runtime_error("unified virtual memory manually disabled on windows!");
    return nullptr;
#endif
    if (!unifiedMem) initUnifiedMemory();
    return unifiedMem->borrow(bytes);
  }
  void Cuda::CudaContext::resetVirtualMem() {
    if (!unifiedMem) initUnifiedMemory();
    unifiedMem->reset();
  }

}  // namespace zs
