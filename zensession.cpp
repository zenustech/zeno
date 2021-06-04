#include <zen/zen.h>

static std::unique_ptr<zen::Session> sess;

extern "C" zen::Session *__zensession_getSession_v1() {
    if (!sess) {
        sess = std::make_unique<zen::Session>();
    }
    return sess.get();
}
