#include "ExecutionPolicy.hpp"
#include <thread>

namespace zs {

  uint get_hardware_concurrency() noexcept { return std::thread::hardware_concurrency(); }

}  // namespace zs