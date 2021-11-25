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
#include <sycl/helpers/sycl_iterator.hpp>


struct SyclHostIteratorTest : public testing::Test {};

using namespace sycl::helpers;

TEST_F(SyclHostIteratorTest, TestIteratorsOnHostAccessor) {
  std::vector<float> v = {1, 3, 5, 7, 9};
  // TODO(Ruyman) : Workaround for #5327
  // cl::sycl::buffer<float> sv(cl::sycl::range<1>(v.size()));
  cl::sycl::buffer<float> sv(v.begin(), v.end());

  ASSERT_EQ(sv.get_count(), v.size());

  {
    // A host vector is just a vector that contains a host_accessor
    // The data of the vector is the host accessor
    auto hostAcc = sv.get_access<cl::sycl::access::mode::read_write>();

    auto vI = v.begin();
    int count = 0;

    for (; vI != v.end(); vI++, count++) {
      hostAcc[count] = *vI;
      ASSERT_EQ(*vI, hostAcc[count]);
    }

    vI = v.begin();
    count = 0;

    for (auto i = begin(hostAcc); i != end(hostAcc); i++, vI++) {
      // DEBUG
      // printf("[%d] %g == %g  \n", count, *i, hostAcc[count]);
      EXPECT_EQ(*vI, *i);
      ASSERT_LT(count++, sv.get_size());
    }
  }
}

TEST_F(SyclHostIteratorTest, TestUsingStlAlgorithm) {
  std::vector<float> v = {1, 3, 5, 7, 9};
  // TODO(Ruyman) : Workaround for #5327
  // cl::sycl::buffer<float> sv(cl::sycl::range<1>(v.size()));
  cl::sycl::buffer<float> sv(v.begin(), v.end());

  auto hostAcc = sv.get_access<cl::sycl::access::mode::read_write>();

  std::transform(begin(hostAcc), end(hostAcc), begin(hostAcc),
                 [=](float e) { return e * 2; });

  auto vI = v.begin();
  int count = 0;

  for (auto i = begin(hostAcc); i != end(hostAcc); i++, vI++) {
    // DEBUG
    // printf("[%d] %g == %g (%p == %p) \n",
    //          count, *i,
    //          hostAcc[count], &hostAcc[0], &(*i));
    ASSERT_EQ(*(vI)*2, *i);
    count++;
  }
}
