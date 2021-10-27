#include <CL/sycl.hpp>
#include <cstdio>


int main() {
    sycl::queue que;

    sycl::buffer<sycl::cl_int, 1> buf(32);

    // Submitting command group(work) to queue
    que.submit([&](sycl::handler &cgh) {
        auto dbuf = buf.get_access<sycl::access::mode::write>(cgh);
        sycl::range<1> dim(32);
        cgh.parallel_for<class FillBuffer>(dim, [=](sycl::id<1> id) {
            dbuf[id] = (sycl::cl_int)id.get(0) + 1;
        });
    });

    auto hbuf = buf.get_access<sycl::access::mode::read_write>();

    for (int i = 0; i < 32; i++) {
        printf("%d\n", hbuf[i]);
    }
    return 0;
}
