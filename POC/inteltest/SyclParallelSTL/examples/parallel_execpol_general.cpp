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

#include <experimental/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

/* Basic policy example from the original proposal
 */
int main() {
  std::vector<int> v = { 3, 1, 5, 6 };
  using namespace std::experimental::parallel;

  // explicitly sequential sort
  sort(seq, v.begin(), v.end());

  // permitting parallel execution
  sort(par, v.begin(), v.end());

  // permitting vectorization as well
  sort(vec, v.begin(), v.end());

  // sort with dynamically-selected execution
  size_t threshold = 1;
  execution_policy exec = seq;
  if (v.size() > threshold) {
    exec = par;
  }

  sort(exec, v.begin(), v.end());

  return 0;
}
