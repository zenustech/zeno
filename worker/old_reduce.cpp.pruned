#include <zeno/zbb/blocked_range.h>
#include <zeno/zbb/auto_profiler.h>
#include <zeno/zbb/parallel_for.h>
#include <zeno/zbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <cstdlib>
#include <cmath>
#ifndef _OPENMP
#pragma message("WARNING: OpenMP not enabled")
#endif

int main()
{
    std::vector<float> arr;
    for (std::size_t i = 0; i < 32*4096*768; i++) {
        arr.push_back(drand48());
    }

    for (int _ = 0; _ < 8; _++) {
        {
            zbb::auto_profiler _("omp");
            float res = 0;
/*#pragma omp parallel for
            for (std::size_t i = 0; i < arr.size(); i++) {
                arr[i] *= arr[i];
            }*/
#pragma omp parallel for reduction(+: res)
            for (std::size_t i = 0; i < arr.size(); i++) {
                res += arr[i];
            }
            printf("result: %f\n", res);
        }

        {
            zbb::auto_profiler _("tbb");
            /*tbb::parallel_for
            ( size_t{0}, arr.size()
            , [&] (size_t i) {
                arr[i] *= arr[i];
            });*/
            float res = tbb::parallel_reduce
            ( tbb::blocked_range<size_t>(size_t{0}, arr.size())
            , float{0}
            , [&] (tbb::blocked_range<size_t> const &r, float res) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    res += arr[i];
                }
                return res;
            }, [] (float x, float y) { return x + y; }
            );
            printf("result: %f\n", res);
        }

        {
            zbb::auto_profiler _("zbb");
            /*zbb::parallel_for
            ( zbb::make_blocked_range(size_t{0}, arr.size())
            , [&] (auto const &r) {
                for (std::size_t i = r.begin(); i < r.end(); i++) {
                    arr[i] *= arr[i];
                }
            });*/
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
