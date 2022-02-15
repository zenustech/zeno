#include <cstdio>
#include <iostream>
#include <zeno/utils/log.h>

int main() {
    fprintf(stderr, "Zeno runner started...\n");
    stdout = stderr;
    std::cout = std::cerr;

    printf("Hello, C!\n");
    std::cout << "Hello, C++!" << std::endl;
    zeno::log_warn("Hello, spdlog!");

    return 0;
}
