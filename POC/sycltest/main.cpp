#include <CL/sycl.hpp>
#include <iostream>
#include <memory>
#include <array>

namespace sycl = cl::sycl;

struct Instance {
    std::optional<sycl::queue> m_deviceQueue;

    static std::unique_ptr<Instance> g_instance;

    static Instance &get() {
        return *g_instance;
    }
};

std::unique_ptr<Instance> Instance::g_instance;

template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(std::array<T, N> const &VA, std::array<T, N> const &VB,
        std::array<T, N> &VC) {
    sycl::queue deviceQueue;
    sycl::buffer<T, 1> bufferA(VA.data(), N);
    sycl::buffer<T, 1> bufferB(VB.data(), N);
    sycl::buffer<T, 1> bufferC(VC.data(), N);

    deviceQueue.submit([&](sycl::handler &cgh) {
        auto accessorA = bufferA.template get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.template get_access<sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.template get_access<sycl::access::mode::write>(cgh);

        auto kern = [=](sycl::id<1> wiID) {
            accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
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
