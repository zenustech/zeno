#include <csignal>
#include "exception.h"

namespace zpp {

static void __signal_handler(int signo) {
    printf("*** recieved signal %d\n", signo);
    throw signal_exception(signo);
}

static int registerMyHandlers() {
    signal(SIGSEGV, __signal_handler);
    signal(SIGFPE, __signal_handler);
    signal(SIGABRT, __signal_handler);
    signal(SIGILL, __signal_handler);
    signal(SIGBUS, __signal_handler);
    signal(SIGPIPE, __signal_handler);
    return 1;
}

static int doRegisterMyHandlers = registerMyHandlers();

}
