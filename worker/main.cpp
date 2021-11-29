#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>

int main()
{
    std::vector<int> pos = {1, 2, 3, 1};

    tbb::concurrent_unordered_multimap<int, size_t> lut;

    tbb::parallel_for
    ( size_t{0}, pos.size()
    , [&] (size_t i) {
        int p = pos[i];
        lut.emplace(p, i);
    });

    tbb::parallel_for
    ( size_t{0}, pos.size()
    , [&] (size_t i) {
        auto [b, e] = lut.equal_range(i);
        for (auto it = b; it != e; ++it) {
            size_t j = it->second;
            printf("interact %ld %ld\n", i, j);
        }
    });

    return 0;
}
