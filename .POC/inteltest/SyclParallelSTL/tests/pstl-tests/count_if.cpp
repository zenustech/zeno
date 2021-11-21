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
#include "gmock/gmock.h"

#include <vector>
#include <algorithm>
#include <numeric>

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

class CountIfAlgorithm : public testing::Test {
 public:
};

TEST_F(CountIfAlgorithm, TestSyclCountIf) {
  std::vector<float> v;
  int n_elems = 128;

  for (int i = 0; i < n_elems; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    v.push_back(x);
  }

  int res_std =
      std::count_if(v.begin(), v.end(), [=](float v1) { return v1 < 0.5; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class CountIfAlgorithm2> snp(q);
  int res_sycl =
      count_if(snp, v.begin(), v.end(), [=](float v1) { return v1 < 0.5; });

  EXPECT_TRUE(res_std == res_sycl);
}
