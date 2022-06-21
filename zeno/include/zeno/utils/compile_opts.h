#pragma once

#if defined(__GNUC__) || defined(__clang__)

#define ZENO_LIKELY(x) __builtin_expect(!!(x), 1)
#define ZENO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define ZENO_RESTRICT __restrict__
#define ZENO_FORCEINLINE __inline__ __attribute__((always_inline))
#define ZENO_ASSUME_ALIGNED(x, n) ((void *)__builtin_assume_aligned((void *)(x), (n)))

#elif defined(_MSC_VER)

#define ZENO_LIKELY(x) (x)
#define ZENO_UNLIKELY(x) (x)
#define ZENO_RESTRICT __restrict
#define ZENO_FORCEINLINE __forceinline
#define ZENO_ASSUME_ALIGNED(x, n) ((void *)((uintptr_t)(x) & (n - 1)))

#else

#define ZENO_LIKELY(x) (x)
#define ZENO_UNLIKELY(x) (x)
#define ZENO_RESTRICT
#define ZENO_FORCEINLINE
#define ZENO_ASSUME_ALIGNED(x, n) (x)

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
    _Pragma("GCC diagnostic ignored \"-Wextra\"")
#define ZENO_NOWARN_END \
    _Pragma("clang diagnostic pop")
#elif defined __GNUC__
#define ZENO_NOWARN_BEGIN \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wall\"") \
    _Pragma("GCC diagnostic ignored \"-Wextra\"")
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
