#pragma once

namespace hg {

class FPSCounter {
  double m_last_time = 0;
  double *m_intervals;
  int m_count = 0;

  const int N;
  double (*get_time)();

public:
  FPSCounter(double (*get_time)(), int N)
    : N(N), get_time(get_time)
  {
    m_intervals = new double[N];
    for (int i = 0; i < N; i++) {
      m_intervals[i] = 0.0;
    }
  }

  ~FPSCounter() {
    delete m_intervals;
  }

  void tick() {
    double curr_time = get_time();
    if (m_last_time == 0) {
      m_last_time = curr_time;
    }
    double interval = std::max(0.0, curr_time - m_last_time);
    m_intervals[m_count++ % N] = interval;
    m_last_time = curr_time;
  }

  int ready() const {
    return m_count % N == 0;
  }

  double fps() const {
    double itv = interval();
    return itv == 0.0 ? 0.0 : 1.0 / itv;
  }

  double interval() const {
    double ret = 0.0;
    for (int i = 0; i < std::min(N, m_count); i++) {
      ret += m_intervals[i];
    }
    return ret / N;
  }
};

}
