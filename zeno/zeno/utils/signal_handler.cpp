#ifdef ZENO_FAULTHANDLER
#include <zeno/zeno.h>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <sstream>
#include <zeno/utils/print_traceback.h>
#ifdef __linux__
#include <string.h>
#endif

namespace zeno {

static const char *signal_to_string(int signo) {
#ifdef __linux__
    return strsignal(signo);
#else
    const char *signame = "SIG-unknown";
    if (signo == SIGSEGV) signame = "SIGSEGV";
    if (signo == SIGFPE) signame = "SIGFPE";
    if (signo == SIGILL) signame = "SIGILL";
    return signame;
#endif
}

static void signal_handler(int signo) {
    spdlog::error("recieved signal {}: {}", signo, signal_to_string(signo));
    print_traceback(1);
    exit(-signo);
}

static int register_my_handlers() {
    if (getenv("ZEN_NOSIGHOOK")) {
        return 0;
    }
    signal(SIGSEGV, signal_handler);
    signal(SIGFPE, signal_handler);
    signal(SIGILL, signal_handler);
    signal(SIGABRT, signal_handler);
#ifdef __linux__
    signal(SIGBUS, signal_handler);
#endif
    return 1;
}

static int doRegisterMyHandlers = register_my_handlers();

}
#endif
