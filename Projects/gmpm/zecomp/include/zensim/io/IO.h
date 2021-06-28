#pragma once
#include "Utility.hpp"
#include "zensim/Singleton.h"
#include "zensim/execution/Concurrency.h"

namespace zs {

  struct IO : Singleton<IO> {
  private:
    void wait() {
      std::unique_lock<std::mutex> lk{mut};
      cv.wait(lk, [this]() { return !this->bRunning || !this->jobs.empty(); });
    };
    void worker() {
      while (bRunning) {
        wait();
        auto job = jobs.try_pop();
        if (job) (*job)();
      }
    }

  public:
    IO() : bRunning{true} {
      th = std::thread([this]() { this->worker(); });
    }
    ~IO() {
      while (!jobs.empty()) cv.notify_all();
      bRunning = false;
      th.join();
    }

    static void flush() {
      while (!instance().jobs.empty()) instance().cv.notify_all();
    }
    static void insert_job(std::function<void()> job) {
      std::unique_lock<std::mutex> lk{instance().mut};
      instance().jobs.push(job);
      lk.unlock();
      instance().cv.notify_all();
    }

  private:
    bool bRunning;
    std::mutex mut;
    std::condition_variable cv;
    threadsafe_queue<std::function<void()>> jobs;
    std::thread th;
  };

}  // namespace zs
