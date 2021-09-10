#include <CL/sycl.hpp>
#include <cstdio>

namespace sycl = cl::sycl;

static constexpr int N = 1024;

class kernel0;

int main() {
    sycl::queue q;

    std::array<int, N> arr;
    for (int i = 0; i < N; i++) {
        arr[i] = i % 4;
    }

    sycl::buffer<int> buf(arr.data(), arr.size());
    q.submit([&] (sycl::handler &cgh) {
        auto axr = buf.get_access<sycl::access::mode::read_write>(cgh);
        //cgh.parallel_for<kernel0>(sycl::range<1>(N), [=] (sycl::id<1> id) {
            //axr[id[0]] = id[0] + 1;
        //});
    });

    /*for (int i = 0; i < N; i++) {
        printf("%d\n", arr[i]);
    }*/
    return 0;
}
