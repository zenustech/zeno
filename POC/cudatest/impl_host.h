#pragma once

#define FDB_IMPL_HOST 1
#define FDB_CONSTEXPR constexpr
#define FDB_HOST_DEVICE
#define FDB_DEVICE

#include <cstdlib>
#include <cstring>
#include <utility>
#include <atomic>
#include "vec.h"

namespace fdb {

static void synchronize() {
}

struct ParallelConfig {
    size_t block_size{1024};
    size_t saturation{1};
};

template <class Kernel>
static void parallel_for(size_t dim, Kernel kernel, ParallelConfig cfg = {1024, 1}) {
    #pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
        kernel(std::as_const(i));
    }
}

template <class Kernel>
static void parallel_for(vec2S dim, Kernel kernel, ParallelConfig cfg = {32, 1}) {
    parallel_for(dim[0] * dim[1], [=] (size_t i) {
        size_t y = i / dim[0];
        size_t x = i % dim[0];
        vec2S idx(x, y);
        kernel(std::as_const(idx));
    });
}

template <class Kernel>
static void parallel_for(vec3S dim, Kernel kernel, ParallelConfig cfg = {8, 1}) {
    parallel_for(dim[0] * dim[1] * dim[2], [=] (size_t i) {
        size_t z = i / dim[1];
        size_t j = i % dim[1];
        size_t y = j / dim[0];
        size_t x = j % dim[0];
        vec3S idx(x, y, z);
        kernel(std::as_const(idx));
    });
}

static void memcpy_d2d(void *dst, const void *src, size_t n) {
    std::memcpy(dst, src, n);
}

static void memcpy_d2h(void *dst, const void *src, size_t n) {
    std::memcpy(dst, src, n);
}

static void memcpy_h2d(void *dst, const void *src, size_t n) {
    std::memcpy(dst, src, n);
}

static void *allocate(size_t n) {
    return std::malloc(n);
}

static void deallocate(void *p) {
    std::free(p);
}

static void *reallocate(void *p, size_t old_n, size_t new_n) {
    return std::realloc(p, new_n);
}

#if defined(__cpp_lib_atomic_ref)
template <class T>
using atomic_ref = std::atomic_ref<T>;
#else
template <class T>
struct atomic_ref {
    std::atomic<T> *p;
    atomic_ref(T &t) : p((std::atomic<T> *)&t) {}

#define _PER_ATOMIC_OP(func) \
    template <class ...Ts> \
    auto func(Ts &&...ts) const { \
        return p->func(std::forward<Ts>(ts)...); \
    }
_PER_ATOMIC_OP(compare_exchange_weak)
_PER_ATOMIC_OP(compare_exchange_strong)
_PER_ATOMIC_OP(fetch_add)
_PER_ATOMIC_OP(fetch_sub)
_PER_ATOMIC_OP(fetch_and)
_PER_ATOMIC_OP(fetch_or)
_PER_ATOMIC_OP(fetch_xor)
_PER_ATOMIC_OP(exchage)
_PER_ATOMIC_OP(load)
_PER_ATOMIC_OP(store)
};
#endif

template <class T>
static T atomic_casw(T *dst, T cmp, T src) {
    return atomic_ref<T>(*dst).compare_exchange_weak(cmp, src);
}

template <class T>
static T atomic_cass(T *dst, T cmp, T src) {
    return atomic_ref<T>(*dst).compare_exchange_strong(cmp, src);
}

template <class T>
static T atomic_add(T *dst, T src) {
    return atomic_ref<T>(*dst).fetch_add(src);
}

template <class T>
static T atomic_sub(T *dst, T src) {
    return atomic_ref<T>(*dst).fetch_sub(src);
}

template <class T>
static T atomic_and(T *dst, T src) {
    return atomic_ref<T>(*dst).fetch_and(src);
}

template <class T>
static T atomic_or(T *dst, T src) {
    return atomic_ref<T>(*dst).fetch_or(src);
}

template <class T>
static T atomic_xor(T *dst, T src) {
    return atomic_ref<T>(*dst).fetch_xor(src);
}

template <class T>
static T atomic_swap(T *dst, T src) {
    return atomic_ref<T>(*dst).exchage(src);
}

template <class T>
static T atomic_load(T const *src) {
    return atomic_ref<T>(*(T *)src).load();
}

template <class T>
static void atomic_store(T *dst, T src) {
    atomic_ref<T>(*dst).store(src);
}

}
