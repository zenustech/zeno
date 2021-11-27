#include <cstdio>
#include <zeno/zbb/parallel_for.h>
#include <zeno/zbb/parallel_reduce.h>

int main()
{
    int ret = zbb::parallel_reduce(1, 101,
    0, [] (int &lhs, int rhs) {
        lhs += rhs;
    }, [] (int &res, int i) {
        res += i;
    });
    printf("result: %d\n", ret);
    ret = zbb::parallel_reduce(1, 10,
    1, [] (int &lhs, int rhs) {
        lhs *= rhs;
    }, [] (int &res, int i) {
        res *= i;
    });
    printf("result: %d\n", ret);
    return 0;
}
