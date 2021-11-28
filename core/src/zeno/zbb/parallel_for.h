#pragma once


#include <zeno/zbb/blocked_range.h>
#include <thread>
#include <vector>
#include <mutex>


ZENO_NAMESPACE_BEGIN
namespace zbb {

template <class ...Tls>
static void parallel_arena(std::size_t nprocs, auto const &body, Tls const &...tls) {
    std::vector<std::jthread> pool;
    for (std::size_t tid = 0; tid < nprocs; tid++) {
        pool.emplace_back([tid, tls..., &body] {
            body(tid, tls...);
        });
    }
    for (auto &&thr: pool) {
        thr.join();
    }
}


template <class T, class ...Tls>
static void parallel_for(blocked_range<T> const &r, auto const &body, Tls const &...tls) {
    std::size_t nprocs = get_num_procs();
    std::size_t ngrain = r.grain();

    T it = r.begin(), end = r.end();
    std::mutex mtx;
    parallel_arena<Tls...>(nprocs, [&] (std::size_t tid, Tls &...tls) {
        bool flag = true;
        do {
            mtx.lock();
            auto ie = it + ngrain;
            [[unlikely]] if (ie >= end) {
                it = ie;
                if (ie == end) {
                    mtx.unlock();
                    break;
                }
                ie = end;
                flag = false;
            }
            blocked_range<T> const r(it, ie, ngrain);
            it = ie;
            mtx.unlock();
            body(r, tls...);
        } while (flag);
    }, tls...);
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
