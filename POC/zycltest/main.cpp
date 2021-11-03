#include <CL/sycl.hpp>


auto &default_queue() {
    static sycl::queue q;
    return q;
}


template <class T>
using usm_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, alignof(T)>;


int main() {
    std::vector<int, usm_allocator<int>> v(32, {default_queue()});
    default_queue().submit([&] (sycl::handler &cgh) {
        auto v_p = v.data();
        cgh.parallel_for(sycl::range<1>(v.size()), [=] (sycl::item<1> it) {
            v_p[it.get_id(0)] = it.get_id(0);
        });
    }).wait();
    v.resize(42);
    for (auto const &x: v) {
        printf("%d\n", x);
    }
}
