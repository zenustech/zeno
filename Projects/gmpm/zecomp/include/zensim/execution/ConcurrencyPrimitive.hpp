#pragma once
#include <any>
#include <atomic>
#include <limits>

#include "zensim/TypeAlias.hpp"

namespace zs {

  /// shared within a single process
  struct Futex : std::atomic<u32> {
    enum struct result_t {
      value_changed,  // when expected != atomic value
      awoken,         // awoken from 'wake' (success state)
      interrupted,    //
      timedout        // not applicable to wait, applicable to wait until
    };
    // put the current thread to sleep if the expected value matches the value in the atomic
    // waitmask will be saved and compared to the wakemask later in the wake call
    // to check if you wanna wake up this thread or keep it sleeping
    result_t wait(u32 expected, u32 waitMask = ~(u32)0);
    result_t waitUntil(u32 expected, i64 deadline, u32 waitMask = ~(u32)0);
    // wake up the thread if (wakeMask & waitMask == true)
    int wake(int count = std::numeric_limits<int>::max(), u32 wakeMask = ~(u32)0);
  };

#if 0
  struct Mutex : std::atomic<u8> {
    void lock() noexcept {
      u8 state = this->load(std::memory_order_relaxed);
      if (likely((state & kIsLocked) == 0
                 && this->compare_exchange_weak(&state, state | kIsLocked,
                                                std::memory_order_acquire,
                                                std::memory_order_relaxed)))
        return;
      lockImpl(state);
    }
    bool try_lock() noexcept {
      u8 state = this->load(std::memory_order_relaxed);
      do {
        if (state & kIsLocked) return false;
      } while (!this->compare_exchange_weak(&state, state | kIsLocked, std::memory_order_acquire,
                                            std::memory_order_relaxed));
      return true;
    }
    void unlock() noexcept {
      u8 oldState = this->load(std::memory_order_relaxed), newState;
      do {
        newState = oldState & ~(kIsLocked | kIsParked);
      } while (!this->compare_exchange_weak(&oldState, newState, std::memory_order_acquire,
                                            std::memory_order_relaxed));
      if (oldState & kIsParked) unlockImpl();
    }

  private:
    void lockImpl(u8 oldState) noexcept {
      size_t spinCount = 0;
      static constexpr size_t spinLimit = 1000;
      static constexpr size_t yieldLimit = 1;
      u8 newState;
      u8 needPark = 0;
    retry:
      if ((oldState & kIsLocked) != 0) {
        ++spinCount;
        if (spinCount > spinLimit + yieldLimit) {
          newState = oldState | kIsParked;
          if (newState != oldState) {
            if (!this->compare_exchange_weak(&oldState, newState, std::memory_order_acquire,
                                             std::memory_order_relaxed))
              goto retry;
          }
        } else if (spinCount > spinLimit) {
          THREAD_YIELD();
        } else {
          THREAD_PAUSE();
        }
        oldState = this->load(std::memory_order_relaxed);
        goto retry;
      }
      newState = oldState | kIsLocked | needPark;
      if (!this->compare_exchange_weak(&oldState, newState, std::memory_order_acquire,
                                       std::memory_order_relaxed))
        goto retry;
    }
    void unlockImpl() noexcept { ; }
  };
#endif

#if 0

  class condition_variable {
    using steady_clock = chrono::steady_clock;
    using system_clock = chrono::system_clock;
#  ifdef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
    using __clock_t = steady_clock;
#  else
    using __clock_t = system_clock;
#  endif
    typedef __gthread_cond_t __native_type;

#  ifdef __GTHREAD_COND_INIT
    __native_type _M_cond = __GTHREAD_COND_INIT;
#  else
    __native_type _M_cond;
#  endif

  public:
    typedef __native_type* native_handle_type;

    condition_variable() noexcept;
    ~condition_variable() noexcept;

    condition_variable(const condition_variable&) = delete;
    condition_variable& operator=(const condition_variable&) = delete;

    void notify_one() noexcept;

    void notify_all() noexcept;

    void wait(unique_lock<mutex>& __lock) noexcept;

    template <typename _Predicate> void wait(unique_lock<mutex>& __lock, _Predicate __p) {
      while (!__p()) wait(__lock);
    }

#  ifdef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
    template <typename _Duration>
    cv_status wait_until(unique_lock<mutex>& __lock,
                         const chrono::time_point<steady_clock, _Duration>& __atime) {
      return __wait_until_impl(__lock, __atime);
    }
#  endif

    template <typename _Duration>
    cv_status wait_until(unique_lock<mutex>& __lock,
                         const chrono::time_point<system_clock, _Duration>& __atime) {
      return __wait_until_impl(__lock, __atime);
    }

    template <typename _Clock, typename _Duration>
    cv_status wait_until(unique_lock<mutex>& __lock,
                         const chrono::time_point<_Clock, _Duration>& __atime) {
#  if __cplusplus > 201703L
      static_assert(chrono::is_clock_v<_Clock>);
#  endif
      const typename _Clock::time_point __c_entry = _Clock::now();
      const __clock_t::time_point __s_entry = __clock_t::now();
      const auto __delta = __atime - __c_entry;
      const auto __s_atime = __s_entry + __delta;

      if (__wait_until_impl(__lock, __s_atime) == cv_status::no_timeout)
        return cv_status::no_timeout;
      // We got a timeout when measured against __clock_t but
      // we need to check against the caller-supplied clock
      // to tell whether we should return a timeout.
      if (_Clock::now() < __atime) return cv_status::no_timeout;
      return cv_status::timeout;
    }

    template <typename _Clock, typename _Duration, typename _Predicate>
    bool wait_until(unique_lock<mutex>& __lock,
                    const chrono::time_point<_Clock, _Duration>& __atime, _Predicate __p) {
      while (!__p())
        if (wait_until(__lock, __atime) == cv_status::timeout) return __p();
      return true;
    }

    template <typename _Rep, typename _Period>
    cv_status wait_for(unique_lock<mutex>& __lock, const chrono::duration<_Rep, _Period>& __rtime) {
      using __dur = typename steady_clock::duration;
      auto __reltime = chrono::duration_cast<__dur>(__rtime);
      if (__reltime < __rtime) ++__reltime;
      return wait_until(__lock, steady_clock::now() + __reltime);
    }

    template <typename _Rep, typename _Period, typename _Predicate>
    bool wait_for(unique_lock<mutex>& __lock, const chrono::duration<_Rep, _Period>& __rtime,
                  _Predicate __p) {
      using __dur = typename steady_clock::duration;
      auto __reltime = chrono::duration_cast<__dur>(__rtime);
      if (__reltime < __rtime) ++__reltime;
      return wait_until(__lock, steady_clock::now() + __reltime, std::move(__p));
    }

    native_handle_type native_handle() { return &_M_cond; }

  private:
#  ifdef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
    template <typename _Dur>
    cv_status __wait_until_impl(unique_lock<mutex>& __lock,
                                const chrono::time_point<steady_clock, _Dur>& __atime) {
      auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      auto __ns = chrono::duration_cast<chrono::nanoseconds>(__atime - __s);

      __gthread_time_t __ts = {static_cast<std::time_t>(__s.time_since_epoch().count()),
                               static_cast<long>(__ns.count())};

      pthread_cond_clockwait(&_M_cond, __lock.mutex()->native_handle(), CLOCK_MONOTONIC, &__ts);

      return (steady_clock::now() < __atime ? cv_status::no_timeout : cv_status::timeout);
    }
#  endif

    template <typename _Dur>
    cv_status __wait_until_impl(unique_lock<mutex>& __lock,
                                const chrono::time_point<system_clock, _Dur>& __atime) {
      auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      auto __ns = chrono::duration_cast<chrono::nanoseconds>(__atime - __s);

      __gthread_time_t __ts = {static_cast<std::time_t>(__s.time_since_epoch().count()),
                               static_cast<long>(__ns.count())};

      __gthread_cond_timedwait(&_M_cond, __lock.mutex()->native_handle(), &__ts);

      return (system_clock::now() < __atime ? cv_status::no_timeout : cv_status::timeout);
    }
  };
#endif

}  // namespace zs