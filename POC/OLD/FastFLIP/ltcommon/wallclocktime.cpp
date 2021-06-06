

#ifdef _MSC_VER

#  include <Windows.h>

namespace LosTopos {
  static unsigned __int64 timer_frequency;
  static unsigned __int64 base_time;

  void set_time_base(void) {
    QueryPerformanceFrequency((LARGE_INTEGER*)&timer_frequency);
    QueryPerformanceCounter((LARGE_INTEGER*)&base_time);
  }

  double get_time_in_seconds(void) {
    unsigned __int64 newtime;
    QueryPerformanceCounter((LARGE_INTEGER*)&newtime);
    return (double)(newtime - base_time) / (double)timer_frequency;
  }
}  // namespace LosTopos
#else

#  include <sys/time.h>
#  include <wallclocktime.h>

namespace LosTopos {

  static long base_seconds = 0;

  void set_time_base(void) {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    base_seconds = tv.tv_sec;
  }

  double get_time_in_seconds(void) {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return 0.000001 * tv.tv_usec + (tv.tv_sec - base_seconds);
  }
}  // namespace LosTopos
#endif
