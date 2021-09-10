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

    template <class Jitkey, class RangeT, class KernelT, size_t NBuffers>
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

            cgh.parallel_for<Jitkey>(range, [accessors, kernel](auto wiID) {
                std::array<void *, NBuffers> pointers;
                for (size_t i = 0; i < NBuffers; i++) {
                    pointers[i] = (void *)&accessors[i][0];
                }
                kernel(wiID, pointers);
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
    auto bufferA = Instance::get().new_buffer((void *)VA.data(), N * sizeof(T));
    auto bufferB = Instance::get().new_buffer((void *)VB.data(), N * sizeof(T));
    auto bufferC = Instance::get().new_buffer((void *)VC.data(), N * sizeof(T));

    Instance::get().parallel_for<SimpleVadd<T>>(sycl::range<1>(N), [&](auto id, auto accessors) {
        auto accessorA = accessors[0];
        auto accessorB = accessors[1];
        auto accessorC = accessors[2];
        accessorC[id[0]] = accessorA[id[0]] + accessorB[id[0]];
    }, std::array<int, 3>{bufferA, bufferB, bufferC});

    Instance::get().delete_buffer(bufferA);
    Instance::get().delete_buffer(bufferB);
    Instance::get().delete_buffer(bufferC);
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
