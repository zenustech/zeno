#include <CL/sycl.hpp>

int main() {
    sycl::queue q;
    sycl::buffer<float, 1> buf(64);
    sycl::buffer<float, 1> out(64);

    constexpr size_t bufsize = 64;
    constexpr size_t blksize = 64;

    q.submit([&] (sycl::handler &cgh) {
        auto axr_buf = buf.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for(sycl::range<1>(bufsize), [=] (sycl::item<1> it) {
            axr_buf[it] = it[0] + 1;
        });
    });

    q.submit([&] (sycl::handler &cgh) {
        auto axr_buf = buf.get_access<sycl::access::mode::read>(cgh);
        auto axr_out = buf.get_access<sycl::access::mode::discard_write>(cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>
            lxr_tmp(sycl::range<1>(blksize + (blksize >> 3)), cgh);

        cgh.parallel_for(sycl::nd_range<1>(bufsize, blksize), [=] (sycl::nd_item<1> it) {
            size_t tid = it.get_local_linear_id();
            size_t id = it.get_global_linear_id();

            {
                size_t di = tid;
                di += di >> 4;
                lxr_tmp[di] = axr_buf[id];
            }

            for (size_t offset = 1, stride = blksize >> 1; stride > 1; offset <<= 1, stride >>= 1) {
                it.barrier(sycl::access::fence_space::local_space);
                if (tid < stride) {
                    size_t si = offset * (2 * tid + 1) - 1;
                    size_t di = si + offset;
                    si += si >> 4;
                    di += di >> 4;
                    lxr_tmp[di] += lxr_tmp[si];
                }
            }

            if (tid == 0) {
                size_t di = blksize - 1;
                di += di >> 4;
                lxr_tmp[di] = 0;
            }

            for (size_t offset = blksize >> 1, stride = 1; stride < blksize; offset >>= 1, stride <<= 1) {
                it.barrier(sycl::access::fence_space::local_space);
                if (tid < stride) {
                    size_t si = offset * (2 * tid + 1) - 1;
                    size_t di = si + offset;
                    si += si >> 4;
                    di += di >> 4;
                    auto tmp = lxr_tmp[si];
                    lxr_tmp[si] = lxr_tmp[di];
                    lxr_tmp[di] += tmp;
                }
            }

            it.barrier(sycl::access::fence_space::local_space);

            {
                size_t si = tid;
                si += si >> 4;
                axr_out[id] = lxr_tmp[si];
            }
        });
    });

    auto axr_buf = buf.get_access<sycl::access::mode::discard_write>();
    for (size_t i = 0; i < bufsize; i++) {
        std::cout << axr_buf[i] << std::endl;
    }

    return 0;
}
