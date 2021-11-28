#pragma once


#include <zeno/zbb/blocked_range.h>
#include <zeno/zbb/auto_profiler.h>
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


template <class Func>
struct _arena_item
{
    std::size_t _tid;
    std::size_t _nprocs;
    Func _func;

    inline constexpr decltype(auto) operator()(auto &&...args) const {
        return _func(std::forward<decltype(args)>(args)...);
    }

    inline constexpr std::size_t proc_id() const noexcept {
        return _tid;
    }

    inline constexpr std::size_t num_procs() const noexcept {
        return _nprocs;
    }
};


template <class T, class ...Tls>
static void parallel_arena(blocked_range<T> const &r, auto const &kern) {
    std::size_t nprocs = r.num_procs();
    std::size_t ngrain = r.grain_size();

    T itb = r.begin(), ite = r.end();
    std::mutex mtx;
    parallel_arena(nprocs, [&] (std::size_t tid, Tls &...tls) {
        kern(_arena_item{tid, nprocs, [&] (auto const &body) {
            bool flag = true;
            do {
                mtx.lock();
                T ib = itb;
                T ie = ib + ngrain;
                [[unlikely]] if (ie >= ite) {
                    ie = ite;
                    flag = false;
                }
                blocked_range<T> const r{ib, ie, tid, ngrain, nprocs};
                itb = ie;
                mtx.unlock();

                [[likely]] if (ib != ie) {
                    body(r);
                }
            } while (flag);
        }});
    });
}


}
ZENO_NAMESPACE_END
