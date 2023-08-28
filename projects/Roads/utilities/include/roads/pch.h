#pragma once

#define ROADS_API __declspec(dllexport)
#define ROADS_INLINE inline

#include "data.h"
#include <ctime>

#define ROADS_TIMING_PRE_GENERATED \
    clock_t start; clock_t stop;

#define ROADS_TIMING_BLOCK(a, MSG) \
    start = clock(); \
    a; \
    stop = clock(); \
    printf("%s Elapsed: %f seconds\n", MSG, (double)(stop - start) / CLOCKS_PER_SEC);

#define ROADS_TIMING_START start = clock();
#define ROADS_TIMING_END(MSG) stop = clock(); printf("%s Elapsed: %f seconds\n", MSG, (double)(stop - start) / CLOCKS_PER_SEC);
