#include <zen/zen.h>

static std::unique_ptr<zen::Session> sess;

#ifdef _MSC_VER
#define _DLLEXPORT __declspec(dllexport)
#else
#define _DLLEXPORT
#endif

extern "C" _DLLEXPORT zen::Session *__zensession_getSession_v1() {
    if (!sess) {
        sess = std::make_unique<zen::Session>();
    }
    return sess.get();
}
