#ifdef __linux__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

namespace {

static void handler(int signo) {
    printf("ZENO process recieved signal %d: %s\n", signo, strsignal(signo));
    printf("Launching emergency GDB for debugging...\n");
    char cmd[1024];
    sprintf(cmd, "sudo gdb -p %d", getpid());
    system(cmd);
    exit(-signo);
}

static int registerMyHandlers() {
    if (!getenv("ZEN_AUTOGDB")) {
        return 0;
    }
    printf("ZENO is hooking signal handlers to auto-launch GDB\n");
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

#endif
