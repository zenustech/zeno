#pragma once

#if defined(__GNUC__) || defined(__clang__)

#define ZENO_LIKELY(x) __builtin_expect(!!(x), 1)
#define ZENO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define ZENO_RESTRICT __restrict__
#define ZENO_FORCEINLINE __inline__ __attribute__((always_inline))
#define ZENO_ASSUME_ALIGNED(x, n) ((void *)__builtin_assume_aligned((void *)(x), (n)))
#define ZENO_GNUC_ATTRIBUTE(...) __attribute____((__VA_ARGS__))
#define ZENO_MSVC_DECLSPEC(...)
#define ZENO_ENABLE_IF_GNUC(...) __VA_ARGS__
#define ZENO_ENABLE_IF_MSVC(...)

#elif defined(_MSC_VER)

#define ZENO_LIKELY(x) (x)
#define ZENO_UNLIKELY(x) (x)
#define ZENO_RESTRICT __restrict
#define ZENO_FORCEINLINE __forceinline
#define ZENO_ASSUME_ALIGNED(x, n) ((void *)((uintptr_t)(x) & (n - 1)))
#define ZENO_GNUC_ATTRIBUTE(...)
#define ZENO_MSVC_DECLSPEC(...) __declspec(__VA_ARGS__)
#define ZENO_ENABLE_IF_GNUC(...)
#define ZENO_ENABLE_IF_MSVC(...) __VA_ARGS__

#else

#define ZENO_LIKELY(x) (x)
#define ZENO_UNLIKELY(x) (x)
#define ZENO_RESTRICT
#define ZENO_FORCEINLINE
#define ZENO_ASSUME_ALIGNED(x, n) (x)
#define ZENO_GNUC_ATTRIBUTE(...)
#define ZENO_MSVC_DECLSPEC(...)
#define ZENO_ENABLE_IF_GNUC(...)
#define ZENO_ENABLE_IF_MSVC(...)

#endif

#if defined __INTEL_COMPILER
#define ZENO_NOWARN_BEGIN \
    _Pragma("warning (push)")
#define ZENO_NOWARN_END \
    _Pragma("warning (pop)")
#elif defined __clang__
#define ZENO_NOWARN_BEGIN \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wall\"") \
    _Pragma("GCC diagnostic ignored \"-Wextra\"") \
    _Pragma("GCC diagnostic ignored \"-Wsuggest-override\"") \
    _Pragma("GCC diagnostic ignored \"-Wnon-virtual-dtor\"") \
    _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"") \
    _Pragma("GCC diagnostic ignored \"-Wmissing-declarations\"") \
    _Pragma("GCC diagnostic ignored \"-Wformat-security\"") \
    _Pragma("GCC diagnostic ignored \"-Wformat=\"")
#define ZENO_NOWARN_END \
    _Pragma("clang diagnostic pop")
#elif defined __GNUC__
#define ZENO_NOWARN_BEGIN \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wall\"") \
    _Pragma("GCC diagnostic ignored \"-Wextra\"") \
    _Pragma("GCC diagnostic ignored \"-Wnon-virtual-dtor\"")
    _Pragma("GCC diagnostic ignored \"-Wsuggest-override\"")
    _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
    _Pragma("GCC diagnostic ignored \"-Wmissing-declarations\"") \
    _Pragma("GCC diagnostic ignored \"-Wformat-security\"") \
    _Pragma("GCC diagnostic ignored \"-Wformat=\"")
#define ZENO_NOWARN_END \
    _Pragma("GCC diagnostic pop")
#elif defined _MSC_VER
#define ZENO_NOWARN_BEGIN \
    __pragma(warning(push))
#define ZENO_NOWARN_END \
    __pragma(warning(pop))
#else
#define ZENO_NOWARN_BEGIN
#define ZENO_NOWARN_END
#endif
