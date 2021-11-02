#pragma once

#include <zeno/common.h>
#include <cstdlib>

ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _random_h {

template <class T = unsigned int>
static inline constexpr T randint(unsigned int i) {
	unsigned int value = (i ^ 61) ^ (i >> 16);
	value *= 9;
	value ^= value << 4;
	value *= 0x27d4eb2d;
	value ^= value >> 15;	
    return (T)value;
}

template <class T = float>
static inline constexpr T random(unsigned int i) {
    return (T)randint(i) / (T)4294967296;
}

template <class T = float>
static inline T random() {
#ifdef _WIN32
    return (T)rand() / (T)RAND_MAX;
#else
    return (T)drand48();
#endif
}

static inline void randseed(unsigned long i) {
    srand(i);
#ifndef _WIN32
    srand48(i);
#endif
}

}
}
ZENO_NAMESPACE_END
