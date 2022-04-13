#pragma once

#ifdef ZENO_MULTIPROCESS
#include <cstddef>

void viewDecodeClear();
void viewDecodeAppend(const char *buf, size_t n);
void viewDecodeFinish();
#endif
