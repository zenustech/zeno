#pragma once

#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include "Dim3.h"

namespace ImplIntel {

struct Queue {
    //sycl::queue m_q{sycl::accelerator_selector{}};
    //sycl::queue m_q{sycl::gpu_selector{}};
    //sycl::queue m_q{sycl::cpu_selector{}};
    sycl::queue m_q;

    void __print_device_info() {
        auto device = m_q.get_device();
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
        m_q.parallel_for(sycl::range<3>(dim.x, dim.y, dim.z), [=] (sycl::id<3> idx) {
            kernel(Dim3(idx[0], idx[1], idx[2]));
        }).wait();
    }

    void *__malloc_device(size_t n) {
        return (void *)sycl::malloc_device<unsigned char>(n, m_q);
    }

    void *__malloc_shared(size_t n) {
        return (void *)sycl::malloc_shared<unsigned char>(n, m_q);
    }

    void memcpy_dtod(void *d1, void *d2, size_t size) {
        m_q.memcpy(d1, d2, size).wait();
    }

    void memcpy_dtoh(void *h, void *d, size_t size) {
        m_q.memcpy(h, d, size).wait();
    }

    void memcpy_htod(void *h, void *d, size_t size) {
        m_q.memcpy(d, h, size).wait();
    }

    void __free(void *p) {
        return sycl::free(p, m_q);
    }

    struct Allocator {
        Queue &m_q;

        Allocator(Queue &q) : m_q(q) {}

        void *allocate(size_t n) {
            return m_q.__malloc_shared(n);
        }

        void *reallocate(void *old_p, size_t old_n, size_t new_n) {
            void *new_p = allocate(new_n);
            if (old_n) m_q.memcpy_dtod(new_p, old_p, std::min(old_n, new_n));
            deallocate(old_p);
            return new_p;
        }

        void deallocate(void *p) {
            m_q.__free(p);
        }
    };

    Allocator allocator() {
        return {*this};
    }

    struct DeviceAllocator {
        Queue &m_q;

        DeviceAllocator(Queue &q) : m_q(q) {}

        void *allocate(size_t n) {
            return m_q.__malloc_device(n);
        }

        void *reallocate(void *old_p, size_t old_n, size_t new_n) {
            void *new_p = allocate(new_n);
            if (old_n) m_q.memcpy_dtod(new_p, old_p, std::min(old_n, new_n));
            deallocate(old_p);
            return new_p;
        }

        void deallocate(void *p) {
            m_q.__free(p);
        }
    };

    DeviceAllocator device_allocator() {
        return {*this};
    }
};

using DeviceAllocator = Queue::DeviceAllocator;
using Allocator = Queue::Allocator;

template <class T>
auto make_atomic_ref(T &&t) {
    return sycl::ONEAPI::atomic_ref<std::decay_t<T>
    , sycl::ONEAPI::memory_order::acq_rel
    , sycl::ONEAPI::memory_scope::device
    , sycl::access::address_space::global_space
    >(std::forward<T>(t));
}

}
