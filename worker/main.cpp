#include <zeno/zbb/auto_profiler.h>
#include <zeno/zbb/parallel_for.h>
#include <zeno/zbb/parallel_reduce.h>
#include <cstdlib>

int main()
{
    std::vector<float> arr;
    for (std::size_t i = 0; i < 32*4096*768; i++) {
        arr.push_back(drand48());
    }

    for (int _ = 0; _ < 4; _++) {
        {
            zbb::auto_profiler _("omp");
            float res = 0;
#pragma omp parallel for reduction(+: res)
            for (std::size_t i = 0; i < arr.size(); i++) {
                res += arr[i];
            }
            printf("result: %f\n", res);
        }

        {
            zbb::auto_profiler _("zbb");
            float res = zbb::parallel_reduce
            ( zbb::make_blocked_range(size_t{0}, arr.size())
            , float{0}, [] (float x, float y) { return x + y; }
            , [&] (float &res, auto const &r) {
                for (std::size_t i = r.begin(); i < r.end(); i++) {
                    res += arr[i];
                }
            });
            printf("result: %f\n", res);
        }
    }

    return 0;
}
