#include <zeno/back/allocator.h>
#include <vector>
#include <cstdio>

int main() {
    std::vector<int, zeno::allocator<int, 64, true>> arr;
    arr.resize(1024 * 1024 * 1024);
    printf("%p\n", arr.data());
    return 0;
}
