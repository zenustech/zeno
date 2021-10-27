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

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

struct ForEachAlgorithm : public testing::Test {};

TEST_F(ForEachAlgorithm, TestStdForEach) {
  std::vector<int> v = {2, 1, 3};
  std::vector<int> result = {3, 2, 4};

  std::transform(v.begin(), v.end(), v.begin(),
                 [=](int val) { return val + 1; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), result.begin()));
}

TEST_F(ForEachAlgorithm, TestSyclForEach) {
  std::vector<int> v = {2, 1, 3};
  std::vector<int> result = {3, 2, 4};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ForEachAlgorithm> snp(q);
  for_each(snp, v.begin(), v.end(), [=](int& val) { val--; });

  sycl::sycl_execution_policy<class ForEachAlgorithm2> snp2(q);
  for_each(snp2, v.begin(), v.end(), [=](int& val) { val += 2; });
#if PRINT_OUTPUT
  std::cout << " Elements " << std::endl;
  std::for_each(v.begin(), v.end(),
                [=](int elem) { std::cout << elem << std::endl; });
#endif  // PRINT_OUTPUT

  EXPECT_TRUE(std::equal(v.begin(), v.end(), result.begin()));
}
