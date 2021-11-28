#include <cstdio>
#include <zeno/zbb/parallel_for.h>
#include <zeno/zbb/parallel_reduce.h>
#include <iostream>
#include <chrono>

class AutoProfiler {
private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
public:
    AutoProfiler(std::string name)
        : m_name(std::move(name))
        , m_beg(std::chrono::high_resolution_clock::now())
    {}

    ~AutoProfiler() {
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
        std::cout << m_name << " : " << dur.count() << " musec\n";
    }
};

int main()
{
    std::vector<int> arr;
    for (int i = 0; i < 1024*768; i++) {
        arr.push_back(i + 1);
    }

    {
        AutoProfiler _("omp");
        int res = 0;
#pragma omp parallel for reduction(+: res)
        for (std::size_t i = 0; i < arr.size(); i++) {
            res += arr[i];
        }
        printf("result: %d\n", res);
    }

    {
        AutoProfiler _("zbb");
        int res = zbb::parallel_reduce
        ( zbb::make_blocked_range(arr.begin(), arr.end())
        , int{0}, [] (int x, int y) { return x + y; }
        , [&] (int &res, auto const &r) {
            for (auto const &i: r) {
                res += i;
            }
        });
        printf("result: %d\n", res);
    }

    return 0;
}
