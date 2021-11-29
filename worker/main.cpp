#include <zeno/zbb/concurrent_vector.h>

int main()
{
    zbb::concurrent_vector<int> bin;
    bin.grow_twice();
    bin.at(2) = 1;
    printf("%d\n", bin.at(0));

    return 0;
}
