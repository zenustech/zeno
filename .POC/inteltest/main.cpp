#include <cstdio>
#include <CL/sycl.hpp>


int main() {
    sycl::queue que;

    sycl::buffer<sycl::cl_int, 1> buf(32);

    que.submit([&] (sycl::handler &cgh) {
        auto d_buf = buf.get_access<sycl::access::mode::discard_write>(cgh);
        sycl::nd_range<1> dim(32, 4);
        cgh.parallel_for(dim, [=] (sycl::nd_item<1> id) {
            auto i = id.get_global_id(0);
            d_buf[i] = (sycl::cl_int)i + 1;
        });
    });

    auto h_buf = buf.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 32; i++) {
        printf("%d\n", h_buf[i]);
    }

    return 0;
}
