#include <CL/sycl.hpp>
#include <cstdio>

namespace sycl = cl::sycl;

int main() {
    sycl::queue q;

    int *data = sycl::malloc_shared<int>(1024, q);
    q.parallel_for(sycl::nd_range<1>(1024, 1), [=](sycl::id<1> idx) {
        data[idx[0]] = idx[0];
    });
    q.wait();

    for (int i = 0; i < 1024; i++) {
        printf("%d\n", data[i]);
    }
}
