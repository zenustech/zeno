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

struct ForEachNAlgorithm : public testing::Test {};

using namespace std::experimental::parallel;

TEST_F(ForEachNAlgorithm, TestStdForEachN) {
  std::vector<int> v = {2, 1, 3, 4, 8};
  std::vector<int> result = {3, 2, 4, 5, 9};

  for_each_n(v.begin(), v.size(), [=](int &val) { val--; });

  int adder = 2;
  for_each_n(v.begin(), v.size(), [=](int &val) { val += adder; });

#if PRINT_OUTPUT
  std::cout << " Elements " << std::endl;
  std::for_each(v.begin(), v.end(),
                [=](int elem) { std::cout << elem << std::endl; });
#endif  // PRINT_OUTPUT

  EXPECT_TRUE(std::equal(v.begin(), v.end(), result.begin()));
}

TEST_F(ForEachNAlgorithm, TestStd2ForEachN) {
  std::vector<int> v = {2, 1, 3, 4, 8};
  std::vector<int> result = {3, 2, 4, 5, 9};

  auto iterator1 = for_each_n(v.begin(), v.size(), [=](int &val) { val--; });

  for_each_n(v.begin(), v.size(), [=](int &val) { val += 2; });

  EXPECT_TRUE(v.end() == iterator1);
}

TEST_F(ForEachNAlgorithm, TestStd3ForEachN) {
  std::vector<int> v = {2, 1, 3, 4, 8};
  std::vector<int> result = {3, 2, 4, 5, 9};

  int minus_size = -v.size();
  auto iterator1 = for_each_n(v.begin(), minus_size, [=](int &val) { val++; });

  EXPECT_TRUE(v.begin() == iterator1);
}

TEST_F(ForEachNAlgorithm, TestSyclForEachN) {
  std::vector<int> v = {2, 1, 3, 4, 8};
  std::vector<int> result = {3, 2, 4, 5, 7};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ForEachNAlgorithm> snp(q);
  int threshold = 5;
  int adder = 1;
  for_each_n(snp, v.begin(), v.size(), [=](int &val) {
    if (val > threshold) {
      val -= adder;
    } else {
      val += adder;
    }
  });
#if PRINT_OUTPUT
  std::cout << " Elements " << std::endl;
  std::for_each(v.begin(), v.end(),
                [=](int elem) { std::cout << elem << std::endl; });
#endif  // PRINT_OUTPUT

  EXPECT_TRUE(std::equal(v.begin(), v.end(), result.begin()));
}

TEST_F(ForEachNAlgorithm, TestSycl2ForEachN) {
  std::array<int, 5> v = {2, 1, 3, 4, 8};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ForEachN2Algorithm> snp(q);
  int adder = 1;
  auto iterator1 =
      for_each_n(snp, v.begin(), v.size(), [=](int &val) { val = adder; });
#if PRINT_OUTPUT
  std::cout << " Elements " << std::endl;
  std::for_each(v.begin(), v.end(),
                [=](int elem) { std::cout << elem << std::endl; });
#endif  // PRINT_OUTPUT

  EXPECT_TRUE(v.end() == iterator1);
}

TEST_F(ForEachNAlgorithm, TestSycl3ForEachN) {
  std::vector<int> v = {2, 1, 3, 4, 8};
  std::vector<int> result = {3, 2, 4, 5, 9};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ForEachN3Algorithm> snp(q);
  int minus_size = -v.size();
  auto iterator1 =
      for_each_n(snp, v.begin(), minus_size, [=](int &val) { val++; });
#if PRINT_OUTPUT
  std::cout << " Elements " << std::endl;
  std::for_each(v.begin(), v.end(),
                [=](int elem) { std::cout << elem << std::endl; });
#endif  // PRINT_OUTPUT

  EXPECT_TRUE(v.begin() == iterator1);
}
