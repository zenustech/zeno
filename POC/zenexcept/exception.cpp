#include <cstdio>
#include "exception.h"
#include <stdlib.h>
#include <unistd.h>

namespace zpp {

void __attach_debugger(int exitcode) {
    char cmd[1024];
    sprintf(cmd, "sudo gdb -p %d", getpid());
    system(cmd);
    //system("sudo killall -KILL gdb");
    if (exitcode)
        exit(exitcode);
}

}
