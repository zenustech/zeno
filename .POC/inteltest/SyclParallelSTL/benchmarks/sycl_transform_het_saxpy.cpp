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
#include <sycl/heterogeneous_execution_policy.hpp>
#include "amd_cpu_selector.hpp"

#include "benchmark.h"

using namespace sycl::helpers;

/** benchmark_transform heterogeneous
 * @brief Body Function that executes the SYCL CG of the heterogeneous transform
 */
benchmark<>::time_units_t benchmark_transform_heterogeneous(
    float ratio, const unsigned num_elems, const unsigned numReps) {
  std::vector<int> v1;
  std::vector<int> v2;
  std::vector<int> res;

  for (int i = num_elems; i > 0; i--) {
    v1.push_back(i);
    v2.push_back(i);
    res.push_back(i);
  }
  cl::sycl::queue q;
  amd_cpu_selector cpu_sel;
  cl::sycl::queue q2(cpu_sel);
  sycl::sycl_heterogeneous_execution_policy<class TransformAlgorithm1> snp(
      q, q2, ratio);

  auto mytransform = [&]() {
    float pi = 3.14;
    std::experimental::parallel::transform(
        snp, std::begin(v1), std::end(v1), std::begin(v2), std::begin(res),
        [=](float a, float b) { return pi * a + b; });
  };

  auto time = benchmark<>::duration(numReps, mytransform);

  return time;
}
template <typename TimeT = std::chrono::milliseconds,
          typename ClockT = std::chrono::system_clock>
void output_data(const std::string& short_name, int num_elems, float ratio,
                 TimeT dur, output_type output = output_type::STDOUT) {
  if (output == output_type::STDOUT) {
    std::cerr << short_name << " " << num_elems << " " << ratio << " "
              << dur.count() << std::endl;
  } else if (output == output_type::CSV) {
    std::cerr << short_name << "," << num_elems << "," << ratio << ","
              << dur.count() << std::endl;
  } else {
    std::cerr << " Incorrect output " << std::endl;
  }
}

BENCHMARK_HETEROGENEOUS_MAIN("BENCH_SYCL_HET_TRANSFORM",
                             benchmark_transform_heterogeneous, 0.1f, 33554432,
                             7);
