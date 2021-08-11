#include <zeno/zeno.h>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#ifdef __linux__
#include <string.h>
#include <unistd.h>
#endif

namespace zeno {

void print_traceback();

#ifdef __linux__
void trigger_gdb() {
#ifdef ZENO_FAULTHANDLER
    if (!getenv("ZEN_AUTOGDB"))
        return;
    char cmd[1024];
    sprintf(cmd, "sudo gdb -q "
            " -ex 'set confirm off'"
            " -ex 'set pagination off'"
            " -p %d", getpid());
    system(cmd);
#endif
}
#else
void trigger_gdb() {
}
#endif

#ifdef ZENO_FAULTHANDLER
static void signal_handler(int signo) {
#ifdef __linux__
    spdlog::error("recieved signal {}: {}", signo, strsignal(signo));
#else
    const char *signame = "SIG-unknown";
    if (signo == SIGSEGV) signame = "SIGSEGV";
    if (signo == SIGFPE) signame = "SIGFPE";
    if (signo == SIGILL) signame = "SIGILL";
    spdlog::error("recieved signal {}: {}", signo, signame);
#endif
    print_traceback();
    trigger_gdb();
    exit(-signo);
}

static int registerMyHandlers() {
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

static int doRegisterMyHandlers = registerMyHandlers();
#endif

}
