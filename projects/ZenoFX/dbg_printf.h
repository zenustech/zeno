#pragma once

#include <cstdio>

#ifdef ZENOFX_PRINT_LOGS
#define dbg_printf(...) printf("[ZenoFX] " __VA_ARGS__)
#else
#define dbg_printf(...) /* nothing */
#endif
