#include <csignal>
#include "Exception.h"

namespace zpp {

static void handler(int signo) {
    throw SignalException(signo);
}

static int registerMyHandlers() {
    signal(SIGSEGV, handler);
    signal(SIGFPE, handler);
    signal(SIGABRT, handler);
    signal(SIGILL, handler);
    signal(SIGBUS, handler);
    signal(SIGPIPE, handler);
    return 1;
}

static int doRegisterMyHandlers = registerMyHandlers();

}
