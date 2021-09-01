#include <zeno/zeno.h>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <sstream>
#ifdef __linux__
#include <string.h>
#endif
#include <zeno/utils/FaultHandler.h>
#ifdef ZENO_FAULTHANDLER
#include <setjmp.h>
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
    return signo;
#endif
}

#ifdef ZENO_FAULTHANDLER
static jmp_buf jb;
static bool has_jb = false;

ZENO_API void signal_catcher(std::function<void()> const &callback) {
    struct dtor {
        dtor() { has_jb = true; }
        ~dtor() { has_jb = false; }
    } guard;
    if (int signo = setjmp(jb); signo) {
        spdlog::warn("recoverer from signal {}", signo);
    } else {
        callback();
    }
}

static void signal_handler(int signo) {
    spdlog::error("recieved signal {}: {}", signo, signal_to_string(signo));
    print_traceback(1);
    if (has_jb) {
        spdlog::warn("now using saved jmp_buf...");
        has_jb = false;
        longjmp(jb, signo);
    }
    spdlog::warn("no jmp_buf found, exiting...");
    exit(-signo);
}

static int register_my_handlers() {
    if (getenv("ZEN_NOSIGHOOK")) {
        return 0;
    }
    signal(SIGSEGV, signal_handler);
    signal(SIGFPE, signal_handler);
    signal(SIGILL, signal_handler);
#ifdef __linux__
    signal(SIGBUS, signal_handler);
#endif
    return 1;
}

static int doRegisterMyHandlers = register_my_handlers();
#else
ZENO_API void signal_catcher(std::function<void()> const &callback) {
    callback();
}
#endif

}
