#include <zeno/zbb/parallel_for.h>

int main()
{
    std::vector<int> a;

#pragma omp parallel
    {
        std::map<int, int> tls;
        for (int i = 0; i < a.size(); i++) {
        }
    }

    return 0;
}
