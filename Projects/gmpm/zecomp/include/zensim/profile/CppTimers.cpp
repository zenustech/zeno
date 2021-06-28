#include "CppTimers.hpp"

#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/core.h"
#if defined(ZS_PLATFORM_WINDOWS)
#  include <processthreadsapi.h>  // cputime: GetProcessTimes
#  include <sysinfoapi.h>         // walltime: long long int GetTickCount64()

// https://levelup.gitconnected.com/8-ways-to-measure-execution-time-in-c-c-48634458d0f9
// https://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows/17440673#17440673
static double get_cpu_time() {
  FILETIME a, b, c, d;
  if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
    //  Returns total user time.
    //  Can be tweaked to include kernel times as well.
    return (double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
  } else {
    //  Handle error
    return 0;
  }
}
#elif defined(ZS_PLATFORM_LINUX)
#  include <time.h>
#endif

namespace zs {

  void CppTimer::tick() { std::timespec_get(&last, TIME_UTC); }
  void CppTimer::tock() { std::timespec_get(&cur, TIME_UTC); }
  void CppTimer::tock(std::string_view tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag, elapsed());
  }

  float CppTimer::elapsed() const noexcept {
    long seconds = cur.tv_sec - last.tv_sec;
    long nanoseconds = cur.tv_nsec - last.tv_nsec;
    double elapsed = seconds * 1e3 + nanoseconds * 1e-6;
    return elapsed;
  }

}  // namespace zs