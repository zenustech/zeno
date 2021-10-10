#pragma once

#include <cstdlib>

namespace zeno2::ztd {
inline namespace random {

template <class T = unsigned int>
static inline T irand(unsigned int i) {
	unsigned int value = (i ^ 61) ^ (i >> 16);
	value *= 9;
	value ^= value << 4;
	value *= 0x27d4eb2d;
	value ^= value >> 15;	
    return (T)value;
}

template <class T = float>
static inline T frand(unsigned int i) {
    return (T)irand(i) / (T)4294967296;
}

template <class T = float>
static inline T frand() {
#ifdef _WIN32
    return (T)rand() / (T)RAND_MAX;
#else
    return (T)drand48();
#endif
}

static inline void srand(unsigned long i) {
    srand(i);
#ifndef _WIN32
    srand48(i);
#endif
}

}
}
