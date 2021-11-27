#include <cstdio>
#include <zeno/zbb/parallel_for.h>

int main()
{
    zbb::parallel_for(0, 100, [] (int i) {
        printf("%d\n", i);
    });
    printf("Hello, world!\n");
    return 0;
}
