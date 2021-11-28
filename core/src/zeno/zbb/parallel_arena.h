#pragma once


#include <zeno/zbb/blocked_range.h>
#include <functional>
#include <thread>
#include <vector>
#include <mutex>
#include <deque>


ZENO_NAMESPACE_BEGIN
namespace zbb {


struct arena {
    using task = std::function<void(std::size_t)>;

    std::size_t _nprocs;
    std::mutex _tasks_mtx;
    std::deque<task> _tasks;
    std::vector<std::jthread> _procs;

    explicit arena(std::size_t nprocs)
        : _nprocs(nprocs)
    {}

    void submit(task func) {
        _tasks.push_front(std::move(func));
    }

    void start() {
        for (std::size_t tid = 0; tid < _nprocs; tid++) {
            std::jthread thr{[tid, this] {
                task func;
                {
                    std::lock_guard _(_tasks_mtx);
                    func = std::move(_tasks.back());
                    _tasks.pop_back();
                }
                func(tid);
            }};
            _procs.push_back(std::move(thr));
        }
    }

    void wait() {
        for (auto &thr: _procs) {
            thr.join();
        }
    }
};


template <class T>
static void parallel_arena(blocked_range<T> const &r, auto const &kern) {
    std::size_t nprocs = r.num_procs();
    std::size_t ngrain = r.grain_size();

    T itb = r.begin(), ite = r.end();
    std::mutex mtx;
    arena a(nprocs);
    for (T it = r.begin(); it != r.end();) {
        T b = it;
        T e = it + ngrain;
        if (e >= r.end())
            e = r.end();
        a.submit([&kern, b, e, ngrain, nprocs] (std::size_t tid) {
            kern([&] (auto const &body) {
                blocked_range<T> const r{b, e, tid, ngrain, nprocs};
                body(r);
            }, tid);
        });
        it = e;
    }
    a.start();
    a.wait();
}


}
ZENO_NAMESPACE_END
