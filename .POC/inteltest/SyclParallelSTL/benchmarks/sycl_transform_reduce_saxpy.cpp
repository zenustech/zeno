/* Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <numeric>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include "benchmark.h"

using namespace sycl::helpers;

typedef struct elem {
  float a;
  float b;
} elem;

#define PI 3.141

benchmark<>::time_units_t benchmark_transform_reduce(
    const unsigned numReps, const unsigned num_elems,
    const cli_device_selector cds) {
  std::vector<elem> v;
  auto expected = 0.0f;

  for (unsigned int i = num_elems; i > 0; i--) {
    float a = 10.0f + static_cast<float>(10.0f * sin(i));
    float b = 10.0f + static_cast<float>(10.0f * cos(i));
    v.push_back({a, b});
    expected += a * b * PI;
  }

  cl::sycl::queue q(cds);
  sycl::sycl_execution_policy<class TransformReduceAlgorithm1> snp(q);

  auto transformReduce = [&]() {
    float pi = PI;
    float res = std::experimental::parallel::transform_reduce(
        snp, std::begin(v), std::end(v), [=](elem x) { return x.a * x.b * pi; },
        0,                                         // map multiplication
        [=](float a, float b) { return a + b; });  // reduce addition
  };

  auto time = benchmark<>::duration(numReps, transformReduce);

  return time;
}

BENCHMARK_MAIN("BENCH_SYCL_TRANSFORM_REDUCE", benchmark_transform_reduce, 2u,
               33554432u, 10);
