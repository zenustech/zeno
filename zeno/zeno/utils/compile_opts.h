#pragma once

#if defined(__GNUC__) || defined(__clang__)

#define ZENO_LIKELY(x) __builtin_expect(!!(x), 1)
#define ZENO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define ZENO_FORCEINLINE __inline__ __attribute__((always_inline))
#define ZENO_RESTRICT __restrict__
#define ZENO_ASSUME_ALIGNED(x, n) ((void *)__builtin_assume_aligned((void *)(x), (n)))

#elif defined(_MSC_VER)

#define ZENO_LIKELY(x) (x)
#define ZENO_UNLIKELY(x) (x)
#define ZENO_RESTRICT __restrict
#define ZENO_FORCEINLINE __forceinline
#define ZENO_ASSUME_ALIGNED(x) ((void *)((uintptr_t)(x) & (n - 1)))

#else

#define ZENO_LIKELY(x) (x)
#define ZENO_UNLIKELY(x) (x)
#define ZENO_RESTRICT
#define ZENO_FORCEINLINE
#define ZENO_ASSUME_ALIGNED(x) (x)

#endif
