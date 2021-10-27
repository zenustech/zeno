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
#ifndef __SYCL_IMPL_ALGORITHM_COMPOSITE_PATTERNS__
#define __SYCL_IMPL_ALGORITHM_COMPOSITE_PATTERNS__

#include <type_traits>
#include <algorithm>
#include <iostream>

template <typename T>
using local_rw_acc =
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>;

template <typename T, cl::sycl::access::mode AM>
using global_acc =
    cl::sycl::accessor<T, 1, AM, cl::sycl::access::target::global_buffer>;

/** class RedutionStrategy.
* @brief This class represent a common pattern for reductive algorithms on GPU
*/
template <class T>
class ReductionStrategy {
 private:
  int localid_;
  int globalid_;
  int local_;
  int length_;
  cl::sycl::nd_item<1> id_;
  local_rw_acc<T> scratch_;

 public:
  ReductionStrategy(int loc, int len, cl::sycl::nd_item<1> i,
                    const local_rw_acc<T>& localmem)
      : localid_(i.get_local_id(0)),
        globalid_(i.get_global_id(0)),
        local_(loc),
        length_(len),
        id_(i),
        scratch_(localmem) {}

  template <typename BinaryOp, typename T1, cl::sycl::access::mode AM1,
            typename T2, cl::sycl::access::mode AM2>
  void workitem_get_from(BinaryOp op, const global_acc<T1, AM1>& elem1,
                         const global_acc<T2, AM2>& elem2) {
    scratch_[localid_] = op(elem1[globalid_], elem2[globalid_]);
    id_.barrier(cl::sycl::access::fence_space::local_space);
  }

  template <typename UnaryOp, typename T1, cl::sycl::access::mode AM1>
  void workitem_get_from(UnaryOp op, const global_acc<T1, AM1>& elem1) {
    scratch_[localid_] = op(elem1[globalid_]);
    id_.barrier(cl::sycl::access::fence_space::local_space);
  }

  template <typename T1, cl::sycl::access::mode AM1>
  void workitem_get_from(const global_acc<T1, AM1> elem) {
    scratch_[localid_] = elem[globalid_];
    id_.barrier(cl::sycl::access::fence_space::local_space);
  }

  template <class BinaryOperator>
  void combine_threads(BinaryOperator bop) {
    int min = (length_ < local_) ? length_ : local_;
    for (int offset = min >> 1; offset > 0; offset = offset >> 1) {
      if (localid_ < offset) {
        scratch_[localid_] =
            bop(scratch_[localid_], scratch_[localid_ + offset]);
      }
      id_.barrier(cl::sycl::access::fence_space::local_space);
    }
  }

  template <typename T1, cl::sycl::access::mode AM1>
  void workgroup_write_to(const global_acc<T1, AM1>& out) {
    if (localid_ == 0) {
      out[id_.get_group(0)] = scratch_[localid_];
    }
  }
};

#endif  // __SYCL_IMPL_ALGORITHM_COMPOSITE_PATTERNS__
