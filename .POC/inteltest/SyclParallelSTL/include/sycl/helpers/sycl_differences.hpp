/*
 * Copyright (c) 2015-2018 The Khronos Group Inc.

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

/**
 * @file
 * @brief Helper functions for finding buffer sizes from iterators before
 * passing to sycl
 */

#ifndef __EXPERIMENTAL_DETAIL_SYCL_DIFFERENCES__
#define __EXPERIMENTAL_DETAIL_SYCL_DIFFERENCES__

#include <type_traits>
#include <typeinfo>
#include <memory>
#include <algorithm>
#include <exception>

/** @defgroup sycl_helpers
 *
 * Contains definitions of classes and functions that help
 * dealing with the SYCL interface.
 *
 */

#include <sycl/helpers/sycl_iterator.hpp>

/** \addtogroup sycl
 * @{
 */
namespace sycl {
namespace helpers {

/**
 *
 * @brief A custom exception for denoting negative distance errors between
 * iterators, e.g. if a begin()/end() pair have been passed in the wrong order
 * to a function.
 *
 */
class negative_iterator_distance : public std::exception {
  virtual const char* what() const throw() {
    return "negative distance between iterator pairs";
  }
} negative_distance;

/**
 *
 * @brief Return the distance between two input iterators, as an unsigned
 * size_t value. Throws a negative_iterator_distance exception if a negative
 * value is found as the distance.
 *
 */
template <class InputIterator>
size_t distance(InputIterator first, InputIterator last) {
  // get the size as an expected difference_type value
  auto difft_val = std::distance(first, last);
  // check and handle size problems
  if (difft_val < 0) {
    // Options: exception, or give an abs value?
    // Exception catches the error early, but possibly incorrectly, and abs
    // value catches it later but possibly in a less meaningful way...
    throw negative_distance;
    // difft_val = std::abs(difft_val);
  }
  return static_cast<size_t>(difft_val);
}

} /** @} namespace helpers */
} /** @} namespace sycl */

#endif  // __EXPERIMENTAL_DETAIL_SYCL_DIFFERENCES__
