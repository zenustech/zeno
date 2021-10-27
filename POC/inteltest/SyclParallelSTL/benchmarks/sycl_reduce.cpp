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
#include <numeric>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include "benchmark.h"

using namespace sycl::helpers;

/** benchmark_reduce
 * @brief Body Function that executes the SYCL CG of Reduce
 */
benchmark<>::time_units_t benchmark_reduce(const unsigned numReps,
                                           const unsigned N,
                                           const cli_device_selector cds) {
  std::vector<int> v;
  for (size_t i = 0; i < N; i++) {
    int x = 10 * (((float)std::rand()) / RAND_MAX);
    v.push_back(x);
  }

  cl::sycl::queue q(cds);
  auto device = q.get_device();
  const cl::sycl::id<3> maxWorkItemSizes =
    device.get_info<cl::sycl::info::device::max_work_item_sizes>();
  const auto local = std::min(
      device.get_info<cl::sycl::info::device::max_work_group_size>(),
      maxWorkItemSizes[0]);
  sycl::sycl_execution_policy<class ReduceAlgorithmBench> snp(q);
  auto bufI = sycl::helpers::make_const_buffer(v.begin(), v.end());
  size_t length = N;

  auto mainLoop = [&]() {
    do {
      auto f = [length, local, &bufI](cl::sycl::handler& h) mutable {
        cl::sycl::nd_range<1> r{cl::sycl::range<1>{std::max(length, local)},
                                cl::sycl::range<1>{local}};
        auto aI =
            bufI.template get_access<cl::sycl::access::mode::read_write>(h);
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>
            scratch(cl::sycl::range<1>(local), h);

        h.parallel_for<class ReduceAlgorithmBench>(
            r, [aI, scratch, local, length](cl::sycl::nd_item<1> id) {
              size_t globalid = id.get_global_id(0);
              size_t localid = id.get_local_id(0);

              if (globalid < length) {
                scratch[localid] = aI[globalid];
              }
              id.barrier(cl::sycl::access::fence_space::local_space);

              size_t min = (length < local) ? length : local;
              for (size_t offset = min >> 1; offset > 0; offset = offset >> 1) {
                if (localid < offset) {
                  scratch[localid] += scratch[localid + offset];
                }
                id.barrier(cl::sycl::access::fence_space::local_space);
              }

              if (localid == 0) {
                aI[id.get_group(0)] = scratch[localid];
              }
            });
      };
      q.submit(f);
      length = length / local;
    } while (length > 1);
    q.wait_and_throw();

    auto hI = bufI.template get_access<cl::sycl::access::mode::read>();
    std::cout << "SYCL Result of Reduction is: " << hI[0] << std::endl;
  };

  auto time = benchmark<>::duration(numReps, mainLoop);

  auto resstd = std::accumulate(v.begin(), v.end(), 0);
  std::cout << "STL Result of Reduction is: " << resstd << std::endl;
  return time;
}

BENCHMARK_MAIN("BENCH_REDUCE", benchmark_reduce, 2, 65536, 1);
