#pragma once

#include <string_view>

namespace zs {

  struct CudaTimer {
    using event_t = void *;
    using stream_t = void *;
    explicit CudaTimer(stream_t sid);
    ~CudaTimer();
    void tick();
    void tock();
    float elapsed();
    void tock(std::string_view tag);

  private:
    stream_t streamId;
    event_t last, cur;
  };

}  // namespace zs
