#ifdef __linux__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

static void handler(int signo) {
    printf("ZENO process recieved signal %d: %s\n", signo, strsignal(signo));
    char cmd[1024];
    sprintf(cmd, "sudo gdb -p %d", getpid());
    system(cmd);
    exit(-signo);
}

static auto _ = (
        signal(SIGSEGV, handler),
        signal(SIGFPE, handler),
        signal(SIGABRT, handler),
        signal(SIGILL, handler),
        signal(SIGBUS, handler),
        signal(SIGPIPE, handler),
        0);

#endif
