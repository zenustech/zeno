#include "corelaunch.h"
#if defined(Q_OS_WIN)
#include <windows.h>
#elif defined(Q_OS_LINUX)
#include <dlfcn.h>
#endif

static int load_dlls() {
#if defined(Q_OS_WIN)  // TODO
    LoadLibrary("zeno_ZenoFX.dll");
    LoadLibrary("zeno_oldzenbase.dll");
#elif defined(Q_OS_LINUX)
    if (!dlopen("libzeno_ZenoFX.so", RTLD_NOW))
        printf("fail to dlopen!\n");
    if (!dlopen("libzeno_oldzenbase.so", RTLD_NOW))
        printf("fail to dlopen!\n");
#endif
    return 1;
}

static int load_dlls_helper = load_dlls();
