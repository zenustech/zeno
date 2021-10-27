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
#include <numeric>

#include "benchmark.h"

benchmark<>::time_units_t benchmark_transform(const unsigned numReps,
                                              const unsigned num_elems,
                                              const cli_device_selector cds) {
  std::vector<float> v1;
  std::vector<float> v2;
  std::vector<float> res;

  for (int i = num_elems; i > 0; i--) {
    v1.push_back(i);
    v2.push_back(i);
    res.push_back(i);
  }

  float pi = 3.14;
  auto mytransform = [&]() {
    std::transform(std::begin(v1), std::end(v1), std::begin(v2),
                   std::begin(res),
                   [=](float a, float b) { return pi * a + b; });
  };
  auto time = benchmark<>::duration(numReps, mytransform);

  return time;
}

BENCHMARK_MAIN("BENCH_STL_TRANSFORM", benchmark_transform, 2, 33554432, 10);
