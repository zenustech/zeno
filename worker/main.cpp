#include <cstdio>
#include <zeno/zbb/parallel_for.h>

int main()
{
    zbb::parallel_for
    ( zbb::make_blocked_range(0, 1024)
    , [] (zbb::blocked_range<int> const &r) {
        printf("%d %d\n", r.begin(), r.end());
    });
    return 1;
}
