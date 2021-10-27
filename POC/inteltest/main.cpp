#include <CL/sycl.hpp>
#include <cstdio>


inline namespace __zenofakesycl {
namespace sycl {

template <class T, size_t N>
struct buffer {
    std::vector<>
};

}
}


template <class T>
struct Vector {
    sycl::buffer<T, 1> buf;
};


int main() {
    sycl::queue que(sycl::default_selector{});

    sycl::buffer<sycl::cl_int, 1> buf(32);

    que.submit([&] (sycl::handler &cgh) {
        auto dbuf = buf.get_access<sycl::access::mode::write>(cgh);
        sycl::range<1> dim(32);
        cgh.parallel_for(dim, [=] (sycl::id<1> id) {
            dbuf[id] = (sycl::cl_int)id.get(0) + 1;
        });
    });

    auto hbuf = buf.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 32; i++) {
        printf("%d\n", hbuf[i]);
    }

    return 0;
}
