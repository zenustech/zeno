#include <CL/sycl.hpp>


int main() {
    // create an array
    std::vector<float> data(65536);

    // fill the array with random numbers
    for (auto &x: data) {
        x = 1.0f;
    }

    // create a buffer for the data
    sycl::buffer<float, 1> buf(data.data(), data.size());
    sycl::buffer<float, 1> out(data.size() / 256);

    // create a queue
    sycl::queue q{sycl::gpu_selector{}};

    // create a kernel
    q.submit([&] (sycl::handler &cgh) {
        auto a_buf = buf.get_access<sycl::access::mode::read>(cgh);
        auto a_out = out.get_access<sycl::access::mode::discard_write>(cgh);

        // create a local accessor for shared memory
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> l_tmp(sycl::range<1>(256), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(data.size(), 256),
            [=](sycl::nd_item<1> it) {
                auto gid = it.get_global_id(0);
                auto lid = it.get_local_id(0);

                l_tmp[lid] = a_buf[gid];
                it.barrier(sycl::access::fence_space::local_space);
                for (int stride = 256 >> 1; stride; stride >>= 1) {
                    if (lid < stride)
                        l_tmp[lid] += l_tmp[lid + stride];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                a_out[gid] = l_tmp[0];
            }
        );
    });

    // make a host accessor for out
    auto a_out = out.get_access<sycl::access::mode::read>();

    // print the results
    float res = 0;
    for (int i = 0; i < data.size() / 256; i++) {
        std::cout << a_out[i] << std::endl;
        res += a_out[i];
    }
    std::cout << res << std::endl;

    return 0;
}
