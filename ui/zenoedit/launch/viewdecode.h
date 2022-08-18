#pragma once

#ifdef ZENO_MULTIPROCESS
#include <cstddef>

void viewDecodeClear();
void viewDecodeAppend(const char *buf, size_t n);
void viewDecodeSetFrameCache(const char *path, int gcmax);
void viewDecodeFinish();
#endif
