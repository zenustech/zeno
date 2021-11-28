#include <cstdio>
#include <zeno/zbb/parallel_for.h>

int main()
{
    zbb::parallel_for
    ( zbb::blocked_range<int>(0, 100), 4
    , [] (zbb::blocked_range<int> const &r) {
        printf("%d %d\n", r.begin(), r.end());
    });
    return 0;
}
