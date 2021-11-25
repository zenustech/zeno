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
#include <iterator>
#include <list>
#include <algorithm>

#include <sycl/execution_policy>
#include <experimental/algorithm>

namespace parallel = std::experimental::parallel;

class FindAlgorithm : public testing::Test {
 public:
};

TEST_F(FindAlgorithm, TestSyclFind) {
  std::vector<float> v;
  int n_elems = 128;
  float search_val = 10.0f;
  int val_idx = std::rand() % n_elems;

  for (int i = 0; i < n_elems / 2; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    if (i == val_idx) {
      v.push_back(x);
    } else {
      v.push_back(
          search_val);  // make sure the searched for value is actually there
    }
  }
  v.push_back(0);  // add a value that we're not searching for

  auto res_std = std::find(begin(v), end(v), search_val);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class FindAlgorithm> snp(q);
  auto res_sycl = parallel::find(snp, begin(v), end(v), search_val);

  EXPECT_TRUE(res_std == res_sycl);
}

TEST_F(FindAlgorithm, TestSyclListFind) {
  std::list<float> v;
  int n_elems = 128;
  float search_val = 10.0f;
  int val_idx = std::rand() % n_elems;

  for (int i = 0; i < n_elems / 2; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    if (i == val_idx) {
      v.push_back(x);
    } else {
      v.push_back(
          search_val);  // make sure the searched for value is actually there
    }
  }
  v.push_back(0);  // add a value that we're not searching for

  auto res_std = std::find(begin(v), end(v), search_val);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class FindAlgorithm2> snp(q);
  auto res_sycl = parallel::find(snp, begin(v), end(v), search_val);

  EXPECT_TRUE(res_std == res_sycl);
}

TEST_F(FindAlgorithm, TestSyclFindIf) {
  std::vector<float> v;
  int n_elems = 128;
  float search_val = 10.0f;
  int val_idx = std::rand() % n_elems;

  for (int i = 0; i < n_elems / 2; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    if (i == val_idx) {
      v.push_back(x);
    } else {
      v.push_back(
          search_val);  // make sure the searched for value is actually there
    }
  }
  v.push_back(0);  // add a value that we're not searching for

  auto res_std = std::find(begin(v), end(v), search_val);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class FindIfAlgorithm> snp(q);
  auto res_sycl = parallel::find_if(snp, begin(v), end(v),
                          [=](float v) { return v == search_val; });

  EXPECT_TRUE(res_std == res_sycl);
}

TEST_F(FindAlgorithm, TestSyclListFindIf) {
  std::list<float> v;
  int n_elems = 128;
  float search_val = 10.0f;
  int val_idx = std::rand() % n_elems;

  for (int i = 0; i < n_elems / 2; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    if (i == val_idx) {
      v.push_back(x);
    } else {
      v.push_back(
          search_val);  // make sure the searched for value is actually there
    }
  }
  v.push_back(0);  // add a value that we're not searching for

  auto res_std = std::find(begin(v), end(v), search_val);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class FindIfAlgorithm2> snp(q);
  auto res_sycl = parallel::find_if(snp, begin(v), end(v),
                          [=](float v) { return v == search_val; });

  EXPECT_TRUE(res_std == res_sycl);
}

TEST_F(FindAlgorithm, TestSyclFindIfNot) {
  std::vector<float> v;
  int n_elems = 128;
  float search_val = 10.0f;
  int val_idx = std::rand() % n_elems;

  for (int i = 0; i < n_elems / 2; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    if (i == val_idx) {
      v.push_back(x);
    } else {
      v.push_back(
          search_val);  // make sure the searched for value is actually there
    }
  }
  v.push_back(0);  // add a value that we're not searching for

  auto res_std = std::find(begin(v), end(v), search_val);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class FindIfNotAlgorithm> snp(q);
  auto res_sycl = parallel::find_if_not(snp, begin(v), end(v),
                              [=](float v) { return v != search_val; });

  EXPECT_TRUE(res_std == res_sycl);
}

TEST_F(FindAlgorithm, TestSyclListFindIfNot) {
  std::list<float> v;
  int n_elems = 128;
  float search_val = 10.0f;
  int val_idx = std::rand() % n_elems;

  for (int i = 0; i < n_elems / 2; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    if (i == val_idx) {
      v.push_back(x);
    } else {
      v.push_back(
          search_val);  // make sure the searched for value is actually there
    }
  }
  v.push_back(0);  // add a value that we're not searching for

  auto res_std = std::find(begin(v), end(v), search_val);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class FindIfNotAlgorithm2> snp(q);
  auto res_sycl = parallel::find_if_not(snp, begin(v), end(v),
                              [=](float v) { return v != search_val; });

  EXPECT_TRUE(res_std == res_sycl);
}

TEST_F(FindAlgorithm, TestSyclFindNotFound) {
  int n_elems = 128;
  std::vector<float> v(n_elems, 9.0f);
  float search_val = 10.0f;

  auto res_std = std::find(begin(v), end(v), search_val);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class FindAlgorithmNotFound> snp(q);
  auto res_sycl = parallel::find(snp, begin(v), end(v), search_val);

  EXPECT_EQ(res_sycl, res_std);
}

