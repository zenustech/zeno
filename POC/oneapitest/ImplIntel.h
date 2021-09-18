#pragma once

#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <memory>
#include "Dim3.h"

#ifdef __SYCL_DEVICE_ONLY__
#define __INTEL_SYCL_CONSTANT __attribute__((opencl_constant))
#else
#define __INTEL_SYCL_CONSTANT
#endif

struct Queue {
    std::shared_ptr<sycl::queue> m_qq;

    Queue()
        : m_qq(std::make_shared<sycl::queue>())
    {}

    sycl::queue &sycl_queue() {
        return *m_qq;
    }

    Queue(Queue const &) = default;
    Queue &operator=(Queue const &) = default;
    Queue(Queue &&) = default;
    Queue &operator=(Queue &&) = default;

    void __print_device_info() {
        auto device = sycl_queue().get_device();
        auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
        std::cout << std::setw(20) << "Platform Name: " << p_name << "\n";
        auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
        std::cout << std::setw(20) << "Platform Version: " << p_version << "\n";
        auto d_name = device.get_info<sycl::info::device::name>();
        std::cout << std::setw(20) << "Device Name: " << d_name << "\n";
        auto max_work_group = device.get_info<sycl::info::device::max_work_group_size>();
        std::cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
        auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
        std::cout << std::setw(20) << "Max Compute Units: " << max_compute_units << "\n\n";
    }

    template <class Kernel>
    void parallel_for(Dim3 dim, Kernel kernel) {
        sycl_queue().parallel_for(sycl::range<3>(dim.x, dim.y, dim.z), [=] (sycl::id<3> idx) {
            kernel(Dim3(idx[0], idx[1], idx[2]));
        }).wait();
    }

    void *allocate(size_t n) {
        return (void *)sycl::malloc_shared<unsigned char>(n, sycl_queue());
    }

    void deallocate(void *p) {
        return sycl::free(p, sycl_queue());
    }

    void *reallocate(void *old_p, size_t old_n, size_t new_n) {
        void *new_p = allocate(new_n);
        if (old_n) sycl_queue().memcpy(new_p, old_p, std::min(old_n, new_n)).wait();
        deallocate(old_p);
        return new_p;
    }

    void memcpy_dtod(void *d1, void *d2, size_t size) {
        sycl_queue().memcpy(d1, d2, size).wait();
    }

    void memcpy_dtoh(void *h, void *d, size_t size) {
        sycl_queue().memcpy(h, d, size).wait();
    }

    void memcpy_htod(void *h, void *d, size_t size) {
        sycl_queue().memcpy(d, h, size).wait();
    }

    template <class T, class Parent>
    struct __AtomicRef {
        Parent parent;

        __AtomicRef(Parent &&parent) : parent(std::move(parent)) {}

        inline T load() {
            return parent.load();
        }

        inline void store(T value) {
            parent.store(value);
        }

        bool store_if_equal(T if_equal, T then_set) {
            return parent.compare_exchange_weak(if_equal, then_set);
        }

        inline T fetch_inc() {
            return parent++;
        }
    };

    template <class T>
    static auto make_atomic_ref(T &&t) {
        using SyclAtomicRef = sycl::ONEAPI::atomic_ref<std::decay_t<T>
        , sycl::ONEAPI::memory_order::acq_rel
        , sycl::ONEAPI::memory_scope::device
        , sycl::access::address_space::global_space
        >;
        return __AtomicRef<std::decay_t<T>, SyclAtomicRef>(
                SyclAtomicRef(std::forward<T>(t)));
    }

    template <class ...Args>
    static void printf(const __INTEL_SYCL_CONSTANT char *fmt, Args &&...args) {
        sycl::ONEAPI::experimental::printf(fmt, std::forward<Args>(args)...);
    }
};
