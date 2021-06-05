#define _ZEN_INDLL
#include <zen/zen.h>

static std::unique_ptr<zen::Session> sess;

ZENAPI zen::Session &zen::getSession() {
    if (!sess) {
        sess = std::make_unique<zen::Session>();
    }
	return *sess;
}
