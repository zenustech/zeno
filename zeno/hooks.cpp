#include <zeno/zeno.h>
#ifdef __linux__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace zeno {

#ifdef __linux__
static void trigger_gdb(int exitcode = 0) {
    printf("*** Launching emergency GDB for debugging...\n");
    char cmd[1024];
    sprintf(cmd, "sudo gdb -q "
            " -ex 'set confirm off'"
            " -ex 'set pagination off'"
            " -p %d", getpid());
    system(cmd);
    if (exitcode != 0)
        exit(exitcode);
}

static void signal_handler(int signo) {
    printf("*** ZENO process recieved signal %d: %s\n", signo, strsignal(signo));
#ifdef ZENO_TRIGGER_GDB
    trigger_gdb(-signo);
#endif
}

static int registerMyHandlers() {
    if (getenv("ZEN_NOSIGHOOK")) {
        return 0;
    }
    signal(SIGSEGV, signal_handler);
    signal(SIGFPE, signal_handler);
    signal(SIGABRT, signal_handler);
    signal(SIGILL, signal_handler);
    signal(SIGBUS, signal_handler);
    signal(SIGPIPE, signal_handler);
    return 1;
}

static int doRegisterMyHandlers = registerMyHandlers();
#endif

ZENAPI Exception::Exception(std::string const &msg) noexcept
    : msg(msg) {
#ifdef ZENO_TRIGGER_GDB
        trigger_gdb();
#endif
}

ZENAPI Exception::~Exception() noexcept = default;

ZENAPI char const *Exception::what() const noexcept {
    return msg.c_str();
}

}
