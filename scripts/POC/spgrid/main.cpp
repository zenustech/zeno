#include <type_traits>
#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <array>
#include <cassert>
#include <omp.h>
#include "iterator_utils.h"
#include "common_utils.h"

#define show(x) (std::cout << #x "=" << (x) << std::endl)



int main(void)
{
    auto r = range(2, 15);
    auto s = slice(r, 2, 10, 3);
    for (auto [i, j]: zip(r, s)) {
        show(i);
        show(j);
    }
}
