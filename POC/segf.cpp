#include <cstdio>
#include <csignal>
#include <setjmp.h>

jmp_buf jb;

void handle(int signo) {
    printf("segv handler\n");
    longjmp(jb, signo);
}

void func() {
    int *ptr = nullptr;
    *ptr = 233;
}

int main() {
    signal(SIGSEGV, handle);
    if (auto signo = setjmp(jb); signo) {
        printf("got %d\n", signo);
    } else {
        func();
    }
    return 0;
}
