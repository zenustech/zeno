#define _ZEN_INDLL
#include <zen/zen.h>

static std::unique_ptr<zen::Session> sess;

_ZEN_API zen::Session &zen::getSession() {
    if (!sess) {
        sess = std::make_unique<zen::Session>();
    }
	printf("%p\n", sess.get());
	return *sess;
}
