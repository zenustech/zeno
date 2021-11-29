#include <zeno/zbb/concurrent_vector.h>

int main()
{
    zbb::concurrent_vector<int> bin;
    bin.grow_twice();
    bin.grow_twice();
    bin.grow_twice();
    bin.grow_twice();
    bin.at(0);
    bin.at(1);
    bin.at(2);
    bin.at(3);
    bin.at(4);
    bin.at(5);
    bin.at(6);
    bin.at(7);
    bin.at(8);

    return 0;
}
