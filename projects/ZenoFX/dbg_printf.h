#pragma once

#if __has_include(<zeno/utils/logstd.h>)
#include <zeno/utils/logstd.h>
#else
#include <cstdio>
#endif


#define dbg_printf(...) zeno::__logstd::log_printf("[ZenoFX] " __VA_ARGS__)
