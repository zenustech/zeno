#ifdef _WIN32
#include <Windows.h>
#include <openvdb/openvdb.h>
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/EventCallbacks.h>

static int callback_id = -1;


BOOL WINAPI DllMain(HINSTANCE hInstDll, DWORD fdwReason, PVOID fImpLoad) {
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH: {
        auto& sess = zeno::getSession();
        callback_id = sess.eventCallbacks->hookEvent("init", [] {
            zeno::log_debug("Initializing OpenVDB...");
            openvdb::initialize();
            zeno::log_debug("Initialized OpenVDB successfully!");
            });
        break;
    }
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH: {
        auto& sess = zeno::getSession();
        sess.eventCallbacks->unhookEvent("init", callback_id);
        break;
    }
    }
    return TRUE;
}


#endif