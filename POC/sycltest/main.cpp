#include <CL/sycl.hpp>
#include <cstdio>

namespace sycl = cl::sycl;

int main() {
    sycl::queue q;

    constexpr int N = 1024;
    int *data = sycl::malloc_shared<int>(N, q);

    q.submit([&] (sycl::handler &cgh) {
        //auto data_axr = data.template get_access<sycl::access::mode::write>(cgh);
        class SimpleVfill;
        cgh.parallel_for<SimpleVfill>(sycl::range<1>(N), [&] (sycl::id<1> id) {
            //data_axr[id] = id;
            data[id[0]] = id[0];
        });
    });
    q.wait();

    for (int i = 0; i < N; i++) {
        printf("%d\n", data[i]);
    }
    return 0;
}
