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
    auto r = bate::iter::range(2, 15);
    auto s = bate::iter::slice(r, 2, 10, 3);
    for (auto [i, j]: bate::iter::zip(r, s)) {
        show(i);
        show(j);
    }
}
