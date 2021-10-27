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
#include <cstdlib>
#include <numeric>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include "benchmark.h"

using namespace sycl::helpers;

int isInsideCircleFunctor(cl::sycl::float2 p) {
  float t = cl::sycl::sqrt((p.x() * p.x()) + (p.y() * p.y()));
  return (t <= 1.0) ? 1 : 0;
}

benchmark<>::time_units_t benchmark_montecarlo(const unsigned numReps,
                                               const unsigned num_elems,
                                               const cli_device_selector cds) {
  // Container for the random points
  std::vector<cl::sycl::float2> pointset;
  std::srand((unsigned int)std::time(0));
  // scatter some random points in the unit circle
  int count = 0;
  for (size_t i = 0; i < num_elems; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    float y = ((float)std::rand()) / RAND_MAX;
    cl::sycl::float2 p(x, y);
    pointset.push_back(p);
  }

  auto myMontecarlo = [&]() {
    cl::sycl::queue q(cds);
    sycl::sycl_execution_policy<class MontecarloAlgorithm1> snp(q);
    count = std::experimental::parallel::transform_reduce(
        snp, pointset.begin(), pointset.end(),
        [=](cl::sycl::float2 p) { return isInsideCircleFunctor(p); }, 0,
        [=](int v1, int v2) { return v1 + v2; });

    float pi = count * 4.0f / pointset.size();
    std::cerr << "Aproximate value of PI: " << pi << std::endl;
  };

  auto time = benchmark<>::duration(numReps, myMontecarlo);

  return time;
}

BENCHMARK_MAIN("BENCH_MONTECARLO", benchmark_montecarlo, 2u, 16777216u, 1);
