#pragma once


#include <zeno/zbb/blocked_range.h>
#include <thread>
#include <vector>
#include <mutex>


ZENO_NAMESPACE_BEGIN
namespace zbb {


static void parallel_arena(std::size_t nprocs, auto const &body) {
    std::vector<std::jthread> pool;
    for (std::size_t tid = 0; tid < nprocs; tid++) {
        pool.emplace_back([tid, &body] {
            body(tid);
        });
    }
    for (auto &&thr: pool) {
        thr.join();
    }
}


template <class T>
static void parallel_for(blocked_range<T> const &r, auto const &body) {
    std::size_t nprocs = get_num_procs();
    std::size_t ngrain = r.grain();

    T it = r.begin();
    std::mutex mtx;
    parallel_arena(nprocs, [&] (std::size_t tid) {
        mtx.lock();
        blocked_range<T> const r(it, it + ngrain, ngrain);
        it = r.end();
        mtx.unlock();
        body(r);
    });
}


template <class T>
static void parallel_for(T i0, T i1, auto const &body) {
    parallel_for(blocked_range<T>(i0, i1), [&] (blocked_range<T> const &r) {
        for (T it = r.begin(); it != r.end(); ++it) {
            body(std::as_const(it));
        }
    });
}


}
ZENO_NAMESPACE_END
