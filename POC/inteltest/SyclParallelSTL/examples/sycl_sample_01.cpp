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

#include <vector>
#include <iostream>
#include <algorithm>

#include <sycl/execution_policy>
#include <experimental/algorithm>
#include <sycl/helpers/sycl_buffers.hpp>

using namespace std::experimental::parallel;
using namespace sycl::helpers;

/* Simple functor that multiplies a number
 * by a factor.
 */
class multiply_by_factor {
  long m_factor;

 public:
  multiply_by_factor(long factor) : m_factor(factor){};

  multiply_by_factor(const multiply_by_factor& mp) {
    this->m_factor = mp.m_factor;
  };

  int operator()(int num) const { return num * m_factor; }
};

/* This sample shows the basic usage of the SYCL execution
 * policies on the different algorithms.
 * In this case we use a sycl buffer to perform all operations on
 * the device.
 * Note that for the moment the sycl variants of the algorithm
 * are in the sycl namespace and not in std::experimental.
 */
int main() {
  std::vector<int> v = {3, 1, 5, 6};
  sycl::sycl_execution_policy<> sycl_policy;

  {
    cl::sycl::buffer<int> b(v.data(), cl::sycl::range<1>(v.size()));
  
    sort(sycl_policy, begin(b), end(b));

    cl::sycl::default_selector h;

    {
      cl::sycl::queue q(h);
      sycl::sycl_execution_policy<class transform1> sepn1(q);
      transform(sepn1, begin(b), end(b), begin(b),
                [](int num) { return num + 1; });

      sycl::sycl_execution_policy<class transform2> sepn2(q);

      long numberone = 2;
      transform(sepn2, begin(b), end(b), begin(b),
                [=](int num) { return num * numberone; });

      transform(sycl_policy, begin(b), end(b), begin(b),
                multiply_by_factor(2));

      sycl::sycl_execution_policy<std::negate<int> > sepn4(q);
      transform(sepn4, begin(b), end(b), begin(b), std::negate<int>());
    }  // All kernels will finish at this point
  }    // The buffer destructor guarantees host syncrhonization
  std::sort(v.begin(), v.end());

  for (size_t i = 0; i < v.size(); i++) {
    std::cout << v[i] << " , " << std::endl;
  }

  return !std::is_sorted(v.begin(), v.end());
}
