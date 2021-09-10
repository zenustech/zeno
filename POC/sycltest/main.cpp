#include <CL/sycl.hpp>
#include <iostream>
#include <memory>
#include <array>

namespace sycl = cl::sycl;


#if 1
class Instance {
    sycl::queue m_deviceQueue;
    std::map<int, sycl::buffer<char>> m_buffers;
    int m_buffercounter;

    static std::unique_ptr<Instance> g_instance;

    struct __private_class {};

    Instance(Instance const &) = delete;
    Instance(Instance &&) = delete;
    Instance &operator=(Instance const &) = delete;
    Instance &operator=(Instance &&) = delete;

public:
    Instance(__private_class) {}

    static Instance &get() {
        if (!g_instance)
            g_instance = std::make_unique<Instance>(__private_class{});
        return *g_instance;
    }

    int new_buffer(void *data, size_t size) {
        int key = m_buffercounter++;
        m_buffers.emplace(std::piecewise_construct,
                std::make_tuple(key), std::make_tuple((char *)data, size));
        return key;
    }

    void delete_buffer(int key) {
        m_buffers.erase(key);
    }

    template <class JitKey, class RangeT, class KernelT, size_t NBuffers>
    void parallel_for(RangeT &&range, KernelT &&kernel,
            std::array<int, NBuffers> const &buffers) {

        m_deviceQueue.submit([&](sycl::handler &cgh) {
            constexpr auto sycl_read_write = sycl::access::mode::read_write;

            using AccessorT = std::decay_t<decltype(
                    m_buffers.at(0).get_access<sycl_read_write>(cgh))>;

            std::array<AccessorT, NBuffers> accessors;
            for (size_t i = 0; i < NBuffers; i++) {
                auto &buffer = m_buffers.at(buffers[i]);
                accessors[i] = buffer->get_access<sycl_read_write>(cgh);
            }

            cgh.parallel_for<JitKey>(range, [accessors, kernel](auto wiID) {
                kernel(wiID);
            });
        });
    }
};


std::unique_ptr<Instance> Instance::g_instance;
#endif


template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(std::array<T, N> const &VA, std::array<T, N> const &VB,
        std::array<T, N> &VC) {
    sycl::queue deviceQueue;
    sycl::buffer<T> bufferA(VA.data(), N);
    sycl::buffer<T> bufferB(VB.data(), N);
    sycl::buffer<T> bufferC(VC.data(), N);

    deviceQueue.submit([&](sycl::handler &cgh) {
        auto accessorA = bufferA.template get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.template get_access<sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.template get_access<sycl::access::mode::write>(cgh);

        auto kern = [=](sycl::id<1> wiID) {
            accessorC[wiID] = wiID[0][&accessorA[0]] + accessorB[wiID];
        };
        sycl::range<1> range{N};
        cgh.parallel_for<class SimpleVadd<T>>(range, kern);
    });
}

int main() {
    constexpr size_t array_size = 4;
    std::array<int, array_size> A = {1, 2, 3, 4}, B = {1, 2, 3, 4}, C;
    simple_vadd(A, B, C);
    for (unsigned int i = 0; i < array_size; i++) {
        if (C[i] != A[i] + B[i]) {
            std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
            return 1;
        }
    }
    std::cout << "The results are correct!\n";
    return 0;
}
