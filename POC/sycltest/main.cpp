#include <CL/sycl.hpp>
#include <memory>
#include <array>
#include "vec.h"


int main() {
    sycl::queue Q;

    int *result = sycl::malloc_shared<int>(32, Q);

    Q.parallel_for(32, [=] (size_t i) {
        result[i] = i;
    }).wait();

    for (int i = 0; i < 32; i++) {
        printf("%d\n", result[i]);
    }

    return 0;
}
