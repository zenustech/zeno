#pragma once

#include <zeno/zycl/core.h>
#include <zeno/zycl/instance.h>
#include <zeno/zycl/helper.h>


#ifndef ZENO_WITH_SYCL
ZENO_NAMESPACE_BEGIN
namespace zycl {
inline namespace ns_parallel_scan {

#error "TODO: implement parallel_scan for no-sycl"

}
}
ZENO_NAMESPACE_END
#else

ZENO_NAMESPACE_BEGIN
namespace zycl {
inline namespace ns_parallel_scan {

template <size_t blksize, class T>
auto _M_make_scanner(handler &cgh) {
    auto lxr_blk = local_access<access::mode::read_write, T>(cgh, range<1>(blksize));

    return [=] (nd_item<1> const &it, T &value, T &partial) {
        size_t tid = it.get_local_linear_id();

        lxr_blk[tid] = value;

        for (size_t offset = 1, stride = blksize >> 1; stride > 0; offset <<= 1, stride >>= 1) {
            it.barrier(access::fence_space::local_space);
            if (tid < stride) {
                size_t si = offset * (2 * tid + 1) - 1;
                size_t di = si + offset;
                lxr_blk[di] += lxr_blk[si];
            }
        }

        if (tid == 0) {
            size_t di = blksize - 1;
            partial = lxr_blk[di];
            lxr_blk[di] = T{};
        }

        for (size_t offset = blksize >> 1, stride = 1; stride < blksize; offset >>= 1, stride <<= 1) {
            it.barrier(access::fence_space::local_space);
            if (tid < stride) {
                size_t si = offset * (2 * tid + 1) - 1;
                size_t di = si + offset;
                auto tmp = lxr_blk[si];
                lxr_blk[si] = lxr_blk[di];
                lxr_blk[di] += tmp;
            }
        }

        it.barrier(access::fence_space::local_space);

        value = lxr_blk[tid];
    };
}

template <size_t blksize, class T>
void _M_parallel_scan(vector<T> &buf, size_t bufsize) {
    if (!bufsize)
        return;

    size_t partsize = (bufsize + blksize - 1) / blksize;
    vector<T> part(partsize);

    default_queue().submit([&] (handler &cgh) {
        auto axr_buf = make_access<access::mode::discard_read_write>(cgh, buf, range<1>(bufsize));
        auto axr_part = make_access<access::mode::discard_write>(cgh, part);
        auto scanner = _M_make_scanner<blksize, T>(cgh);

        cgh.parallel_for(nd_range<1>(partsize * blksize, blksize), [=] (nd_item<1> it) {
            size_t id = it.get_global_linear_id();
            size_t gid = it.get_group_linear_id();
            T val = T{};
            [[likely]] if (id < bufsize)
                val = axr_buf[id];
            scanner(it, val, axr_part[gid]);
            [[likely]] if (id < bufsize)
                axr_buf[id] = val;
        });
    });

    if (bufsize > blksize) {
        _M_parallel_scan<blksize, T>(part, partsize);

        default_queue().submit([&] (handler &cgh) {
            auto axr_buf = make_access<access::mode::discard_read_write>(cgh, buf, range<1>(bufsize));
            auto axr_part = make_access<access::mode::read>(cgh, part);

            cgh.parallel_for(nd_range<1>(partsize * blksize, blksize), [=] (nd_item<1> it) {
                size_t id = it.get_global_linear_id();
                size_t gid = it.get_group_linear_id();
                [[likely]] if (id < bufsize)
                    axr_buf[id] += axr_part[gid];
            });
        });
    }
}


template <size_t blksize, class T>
vector<T> parallel_scan(vector<T> &buf, size_t bufsize) {
    vector<T> sum(1);

    default_queue().submit([&] (handler &cgh) {
        auto axr_buf = make_access<access::mode::read>(cgh, buf);
        auto axr_sum = make_access<access::mode::discard_write>(cgh, buf);

        cgh.single_task([=] {
            axr_sum[0] = axr_buf[bufsize - 1];
        });
    });

    _M_parallel_scan<blksize, T>(buf, bufsize);

    default_queue().submit([&] (handler &cgh) {
        auto axr_buf = make_access<access::mode::read>(cgh, buf);
        auto axr_sum = make_access<access::mode::discard_read_write>(cgh, buf);

        cgh.single_task([=] {
            axr_sum[0] += axr_buf[bufsize - 1];
        });
    });

    return sum;
}


}
}
ZENO_NAMESPACE_END
#endif
