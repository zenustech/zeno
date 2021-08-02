#include <zeno/zeno.h>
#include <zeno/utils/zlog.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

namespace zeno {

void print_traceback();

#ifdef __linux__
void trigger_gdb() {
    print_traceback();
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
    print_traceback();
}
#endif

#ifdef ZENO_FAULTHANDLER
static void signal_handler(int signo) {
    zlog::error("recieved signal {}: {}", signo, strsignal(signo));
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
    signal(SIGBUS, signal_handler);
    return 1;
}

static int doRegisterMyHandlers = registerMyHandlers();
#endif

}
