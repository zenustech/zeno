#ifdef _WIN32

#include <Windows.h>
//#include <vld.h>

BOOL WINAPI DllMain(HINSTANCE hInstDll, DWORD fdwReason, PVOID fImpLoad) {
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH: {
        //VLDEnable();
        break;
    }
    case DLL_THREAD_ATTACH: {
        break;
    }
    case DLL_THREAD_DETACH: {
        break;
    }
    case DLL_PROCESS_DETACH: {
        break;
    }
    }
    return TRUE;
}

#endif