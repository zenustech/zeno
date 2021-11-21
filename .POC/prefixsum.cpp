#include <CL/sycl.hpp>

template <class T>
auto make_scanner(sycl::handler &cgh, size_t blksize) {
    sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>
        lxr_tmp(sycl::range<1>(blksize), cgh);

    return [=] (sycl::nd_item<1> const &it, T &value, T &partial) {
        size_t tid = it.get_local_linear_id();

        lxr_tmp[tid] = value;

        for (size_t offset = 1, stride = blksize >> 1; stride > 0; offset <<= 1, stride >>= 1) {
            it.barrier(sycl::access::fence_space::local_space);
            if (tid < stride) {
                size_t si = offset * (2 * tid + 1) - 1;
                size_t di = si + offset;
                lxr_tmp[di] += lxr_tmp[si];
            }
        }

        if (tid == 0) {
            size_t di = blksize - 1;
            partial = lxr_tmp[di];
            lxr_tmp[di] = 0;
        }

        for (size_t offset = blksize >> 1, stride = 1; stride < blksize; offset >>= 1, stride <<= 1) {
            it.barrier(sycl::access::fence_space::local_space);
            if (tid < stride) {
                size_t si = offset * (2 * tid + 1) - 1;
                size_t di = si + offset;
                auto tmp = lxr_tmp[si];
                lxr_tmp[si] = lxr_tmp[di];
                lxr_tmp[di] += tmp;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        value = lxr_tmp[tid];
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

int main() {
    sycl::queue q;

    constexpr size_t bufsize = 126;
    constexpr size_t blksize = 16;

    sycl::buffer<float, 1> buf(bufsize);

    q.submit([&] (sycl::handler &cgh) {
        auto axr_buf = buf.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for(sycl::range<1>(bufsize), [=] (sycl::item<1> it) {
            axr_buf[it] = it[0] + 1;
        });
    });

    prefix_scan(q, buf, bufsize, blksize);

    auto axr_buf = buf.get_access<sycl::access::mode::discard_write>();
    for (size_t i = 0; i < bufsize; i++) {
        std::cout << axr_buf[i] << std::endl;
    }

    return 0;
}
