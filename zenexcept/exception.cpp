#include <cstdio>
#include "exception.h"
#include <stdlib.h>
#include <unistd.h>

namespace zpp {

void __attach_debugger() {
    char cmd[1024];
    sprintf(cmd, "sudo gdb -p %d", getpid());
    system(cmd);
}

}
