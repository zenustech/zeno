#include <cstdio>
#include <zeno/zbb/parallel_for.h>
#include <zeno/zbb/parallel_reduce.h>

int main()
{
    std::vector<int> val;
    for (int i = 0; i < 100; i++) {
        val.push_back(i + 1);
    }

    int res = zbb::parallel_reduce
    ( zbb::make_blocked_range(val.begin(), val.end())
    , 0, [] (int x, int y) { return x + y; }
    , [] (int &res, auto const &r) {
        for (auto const &i: r) {
            res += i;
        }
    });

    printf("result: %d\n", res);

    return 0;
}
