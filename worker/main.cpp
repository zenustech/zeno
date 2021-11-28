#include <zeno/zbb/auto_profiler.h>
#include <zeno/zbb/parallel_for.h>
#include <zeno/zbb/parallel_reduce.h>

int main()
{
    std::vector<int> arr;
    for (int i = 0; i < 1024*768; i++) {
        arr.push_back(i + 1);
    }

    {
        zbb::auto_profiler _("omp");
        int res = 0;
#pragma omp parallel for reduction(+: res)
        for (std::size_t i = 0; i < arr.size(); i++) {
            res += arr[i];
        }
        printf("result: %d\n", res);
    }

    {
        zbb::auto_profiler _("zbb");
        int res = zbb::parallel_reduce
        ( zbb::make_blocked_range(arr.begin(), arr.end())
        , int{0}, [] (int x, int y) { return x + y; }
        , [&] (int &res, auto const &r) {
            printf("%ld %d %d\n", r.proc_id(), *r.begin(), *r.end());
            for (auto const &i: r) {
                res += i;
            }
        });
        printf("result: %d\n", res);
    }

    return 0;
}
