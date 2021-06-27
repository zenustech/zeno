#include "ConcurrencyPrimitive.hpp"

#include <time.h>

#include <any>
#include <mutex>
#include <shared_mutex>

#if defined(_WIN32)
#  include <synchapi.h>
#elif defined(__linux__)
#  include <linux/futex.h>
#  include <sys/syscall.h> /* Definition of SYS_* constants */
#  include <unistd.h>
#endif

namespace zs {

  decltype(auto) as_mutex(std::any &m) { return std::any_cast<std::mutex &>(m); }
  decltype(auto) as_mutex(const std::any &m) { return std::any_cast<const std::mutex &>(m); }

  decltype(auto) as_shared_mutex(std::any &m) { return std::any_cast<std::shared_mutex &>(m); }
  decltype(auto) as_shared_mutex(const std::any &m) {
    return std::any_cast<const std::shared_mutex &>(m);
  }

  decltype(auto) as_condition_variable(std::any &m) {
    return std::any_cast<std::shared_mutex &>(m);
  }
  decltype(auto) as_condition_variable(const std::any &m) {
    return std::any_cast<const std::shared_mutex &>(m);
  }

  // int futex(int *uaddr, int op, int val, const struct timespec *timeout, int *uaddr2, int val3);
  // long syscall(SYS_futex, u32 *uaddr, int op, u32 val, const timespec *, u32 *uaddr2, u32 val3);
  // bool WaitOnAddress(volatile void *addr, void *compareAddress, size_t, addressSize, dword dwMs)
  Futex::result_t Futex::wait(u32 expected, u32 waitMask) {
    return waitUntil(expected, (i64)-1, waitMask);
  }
  Futex::result_t Futex::waitUntil(u32 expected, i64 deadline, u32 waitMask) {
#if defined(_WIN32)
    /// windows
    u32 undesired = expected;
    bool rc = WaitOnAddress((void *)this, &undesired, sizeof(u32),
                            deadline == (i64)-1 ? INFINITE : deadline);
    if (rc) return result_t::awoken;
    if (undesired != expected) return result_t::value_changed;
    if (GetLastError() == ERROR_TIMEOUT) return result_t::timedout;
    return result_t::interrupted;

#elif defined(__linux__)
    /// linux
    struct timespec tm {};
    struct timespec *timeout = nullptr;
    if (deadline > -1) {
      // seconds, nanoseconds
      tm = timespec{deadline / 1000, (deadline % 1000) * 1000000};
      timeout = &tm;
    }
    int const op = FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG;
    long rc = syscall(SYS_futex, (u32 *)this, op, expected, timeout, nullptr, waitMask);
    if (rc == 0)
      return result_t::awoken;
    else {
      switch (rc) {
        case ETIMEDOUT:
          return result_t::timedout;
        case EINTR:
          return result_t::interrupted;
        case EWOULDBLOCK:
          return result_t::value_changed;
        default:
          return result_t::value_changed;
      }
    }
#endif
  }
  // wake up the thread if (wakeMask & waitMask == true)
  // WakeByAddressSingle/All
  int Futex::wake(int count, u32 wakeMask) {
#if defined(_WIN32)
    if (count == std::numeric_limits<int>::max()) {
      WaitByAddressAll((void *)this);
      return std::numeric_limits<int>::max();
    } else {
      for (int i = 0; i < count; ++i) WakeByAddressSingle((void *)this);
      return count;
    }

#elif defined(__linux__)
    int const op = FUTEX_WAKE_BITSET | FUTEX_PRIVATE_FLAG;
    long rc = syscall(SYS_futex, (u32 *)this, op, count, nullptr, nullptr, wakeMask);
    if (rc < 0) return 0;
    return rc;
#endif
  }

}  // namespace zs