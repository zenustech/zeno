#pragma once

//#if __has_include(<zeno/utils/logstd.h>)
//#include <zeno/utils/logstd.h>
//#else
#include <zeno/utils/Error.h>
#include <cstdio>
//#endif


#define dbg_printf(...) printf("[ZenoFX] " __VA_ARGS__)
#define err_printf(...) do { printf("[ZenoFX] " __VA_ARGS__); throw ::zeno::makeError("ZFX error, please see error message above!"); } while (0)
