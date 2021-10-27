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

#include <iostream>
#include "gmock/gmock.h"

#include <vector>
#include <algorithm>

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

class UniquePTRAlgorithm : public testing::Test {
 public:
};

struct foo {
  int memberA;
  float memberB;
  foo() : memberA(0), memberB(0){};

  foo(int n, float f) : memberA(n), memberB(f){};
};

TEST_F(UniquePTRAlgorithm, TestSyclUniquePTR) {
  constexpr size_t N = 8;

  std::unique_ptr<foo> p(new foo(16, 3.0f));
  std::shared_ptr<foo> sP{std::move(p)};
  cl::sycl::buffer<foo, 1> a{sP, cl::sycl::range<1>(N)};

  // After buffer construction, init has been deallocated
  EXPECT_TRUE(!p);

  cl::sycl::buffer<int> b{cl::sycl::range<1>(N)};

  cl::sycl::queue{}.submit([&](cl::sycl::handler &cgh) {
    auto ka = a.get_access<cl::sycl::access::mode::read>(cgh);
    auto kb = b.get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for<class update>(
        cl::sycl::range<1>{N}, [=](cl::sycl::id<1> index) {
          kb[index] = ka[0].memberA * ka[0].memberB * 2;
        });
  });

  auto result = b.get_access<cl::sycl::access::mode::read>();
  for (int i = 0; i != N; ++i) {
    EXPECT_TRUE(result[i] == 2 * 16 * 3.0f);
  }
}
