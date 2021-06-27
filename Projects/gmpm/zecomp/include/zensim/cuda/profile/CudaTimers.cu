#include "CudaTimers.cuh"
#include "zensim/cuda/CudaConstants.inc"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/core.h"

namespace zs {

  CudaTimer::CudaTimer(stream_t sid) : streamId{sid} {
    cudaEventCreateWithFlags((cudaEvent_t *)&last, cudaEventBlockingSync);
    cudaEventCreateWithFlags((cudaEvent_t *)&cur, cudaEventBlockingSync);
  }
  CudaTimer::~CudaTimer() {
    cudaEventDestroy((cudaEvent_t)last);
    cudaEventDestroy((cudaEvent_t)cur);
  }
  float CudaTimer::elapsed() {
    float duration;
    cudaEventSynchronize((cudaEvent_t)cur);
    cudaEventElapsedTime(&duration, (cudaEvent_t)last, (cudaEvent_t)cur);
    return duration;
  }
  void CudaTimer::tick() { cudaEventRecord((cudaEvent_t)last, (cudaStream_t)streamId); }
  void CudaTimer::tock() { cudaEventRecord((cudaEvent_t)cur, (cudaStream_t)streamId); }
  void CudaTimer::tock(std::string_view tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag, elapsed());
  }

}  // namespace zs