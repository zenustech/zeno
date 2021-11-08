#pragma once

#include <zeno/zycl/core.h>
#include <zeno/zycl/instance.h>
#include <zeno/zycl/helper.h>


#ifndef ZENO_WITH_SYCL

ZENO_NAMESPACE_BEGIN
namespace zycl {
inline namespace ns_parallel_scan {

template <class T>
auto make_scanner(sycl::handler &cgh, size_t blksize) {
    sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>
        lxr_blk(sycl::range<1>(blksize), cgh);

    return [=] (sycl::nd_item<1> const &it, T &value, T &partial) {
        size_t tid = it.get_local_linear_id();

        lxr_blk[tid] = value;

        for (size_t offset = 1, stride = blksize >> 1; stride > 0; offset <<= 1, stride >>= 1) {
            it.barrier(sycl::access::fence_space::local_space);
            if (tid < stride) {
                size_t si = offset * (2 * tid + 1) - 1;
                size_t di = si + offset;
                lxr_blk[di] += lxr_blk[si];
            }
        }

        if (tid == 0) {
            size_t di = blksize - 1;
            partial = lxr_blk[di];
            lxr_blk[di] = 0;
        }

        for (size_t offset = blksize >> 1, stride = 1; stride < blksize; offset >>= 1, stride <<= 1) {
            it.barrier(sycl::access::fence_space::local_space);
            if (tid < stride) {
                size_t si = offset * (2 * tid + 1) - 1;
                size_t di = si + offset;
                auto tmp = lxr_blk[si];
                lxr_blk[si] = lxr_blk[di];
                lxr_blk[di] += tmp;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        value = lxr_blk[tid];
    };
}

template <class T>
void prefix_scan(sycl::queue &q, sycl::buffer<T, 1> &buf, size_t bufsize, size_t blksize) {
    if (!bufsize)
        return;

    size_t partsize = (bufsize + blksize - 1) / blksize;
    sycl::buffer<T, 1> part(partsize);

    q.submit([&] (sycl::handler &cgh) {
        auto axr_buf = buf.template get_access<sycl::access::mode::discard_read_write>(cgh);
        auto axr_part = part.template get_access<sycl::access::mode::discard_write>(cgh);
        auto scanner = make_scanner<float>(cgh, blksize);

        cgh.parallel_for(sycl::nd_range<1>(partsize * blksize, blksize), [=] (sycl::nd_item<1> it) {
            size_t id = it.get_global_linear_id();
            size_t gid = it.get_group_linear_id();
            T val{};
            [[likely]] if (id < bufsize)
                val = axr_buf[id];
            scanner(it, val, axr_part[gid]);
            [[likely]] if (id < bufsize)
                axr_buf[id] = val;
        });
    });

    if (bufsize > blksize) {
        prefix_scan<T>(q, part, partsize, blksize);

        q.submit([&] (sycl::handler &cgh) {
            auto axr_buf = buf.template get_access<sycl::access::mode::discard_read_write>(cgh);
            auto axr_part = part.template get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for(sycl::nd_range<1>(partsize * blksize, blksize), [=] (sycl::nd_item<1> it) {
                size_t id = it.get_global_linear_id();
                size_t gid = it.get_group_linear_id();
                [[likely]] if (id < bufsize)
                    axr_buf[id] += axr_part[gid];
            });
        });
    }
}


}
}
ZENO_NAMESPACE_END
