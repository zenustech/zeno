#pragma once

#include <ctime>
#include <string>

namespace zs {

  /// wall time clock for now
  struct CppTimer {
    using TimeStamp = struct timespec;

    void tick();
    void tock();
    float elapsed() const noexcept;
    void tock(std::string_view tag);

  private:
    // clockid_t clock;
    TimeStamp last, cur;
  };

}  // namespace zs
