#include <zeno/zty/array/Array.h>
#include <iostream>

int main()
{
    zty::Array a = {1, 2, 3, 4};
    auto b = arraySelectOp(arrayBoolOp("cmpgt", a, zty::Array{2}), a);

    for (auto const &x: b.get<int>()) {
        printf("%d\n", x);
    }

    return 0;
}
