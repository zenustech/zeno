#pragma once

#define ROADS_API __declspec(dllexport)
#define ROADS_INLINE inline

#include "data.h"
#include <ctime>

#ifndef NDEBUG

#define ROADS_TIMING_PRE_GENERATED \
    clock_t start; clock_t stop;

#define ROADS_TIMING_BLOCK(MSG, BLOCK) \
    start = clock(); \
    BLOCK; \
    stop = clock(); \
    printf("[Roads] %s Elapsed: %f seconds\n", MSG, (double)(stop - start) / CLOCKS_PER_SEC);

#define ROADS_TIMING_START start = clock();
#define ROADS_TIMING_END(MSG) stop = clock(); printf("[Roads] %s Elapsed: %f seconds\n", MSG, (double)(stop - start) / CLOCKS_PER_SEC);

#else

#define ROADS_TIMING_PRE_GENERATED
#define ROADS_TIMING_BLOCK(MSG, BLOCK) BLOCK;
#define ROADS_TIMING_START
#define ROADS_TIMING_END(MSG)

#endif
