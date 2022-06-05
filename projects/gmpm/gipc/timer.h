#pragma once
#ifndef _TIMER_H_
#define _TIMER_H_
// #include <windows.h>
#include <ctime>

class HighResolutionTimer {
public:
  virtual void set_start() = 0;
  virtual void set_end() = 0;
  virtual float get_millisecond() = 0;
};

#if 0
class HighResolutionTimerForWin : public HighResolutionTimer {
public:
  HighResolutionTimerForWin() {
    QueryPerformanceFrequency(&freq_);
    start_.QuadPart = 0;
    end_.QuadPart = 0;
  }

  void set_start() { QueryPerformanceCounter(&start_); }

  void set_end() { QueryPerformanceCounter(&end_); }

  float get_millisecond() {
    return static_cast<float>((end_.QuadPart - start_.QuadPart) * 1000 /
                              (float)freq_.QuadPart);
  }

private:
  LARGE_INTEGER freq_;
  LARGE_INTEGER start_, end_;
};
#else
class HighResolutionTimerForWin : public HighResolutionTimer {
public:
  void set_start() {
    struct timespec t;
    std::timespec_get(&t, TIME_UTC);
    last = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
  }

  void set_end() {
    struct timespec t;
    std::timespec_get(&t, TIME_UTC);
    cur = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
  }

  float get_millisecond() { return (cur - last); }

private:
  double last, cur;
};
#endif

#endif
