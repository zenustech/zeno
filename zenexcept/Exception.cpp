#include <cstdio>
#include <stdlib.h>
#include <unistd.h>

namespace zpp {

void attachDebugger() {
    char cmd[1024];
    sprintf(cmd, "sudo gdb -p %d", getpid());
    system(cmd);
}

}
