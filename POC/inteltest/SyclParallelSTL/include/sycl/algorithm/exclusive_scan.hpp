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

#ifndef __SYCL_IMPL_ALGORITHM_EXCLUSIVE_SCAN__
#define __SYCL_IMPL_ALGORITHM_EXCLUSIVE_SCAN__

#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_namegen.hpp>
#include <sycl/algorithm/buffer_algorithms.hpp>

namespace sycl {
namespace impl {

#ifdef SYCL_PSTL_USE_OLD_ALGO

/* exclusive_scan.
 * Implementation of the command group that submits a exclusive_scan kernel.
 * The kernel is implemented as a lambda.
 */
template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class ElemT, class BinaryOperation>
OutputIterator exclusive_scan(ExecutionPolicy &sep, InputIterator b,
                              InputIterator e, OutputIterator o, ElemT init,
                              BinaryOperation bop) {
  auto q = sep.get_queue();
  auto device = q.get_device();

  auto bufI = sycl::helpers::make_const_buffer(b, e);

  auto vectorSize = bufI.get_count();
  // declare a temporary "swap" buffer
  auto bufO = sycl::helpers::make_buffer(o, o + vectorSize);

  const auto ndRange = sep.calculateNdRange(vectorSize);
  // calculate iteration count, with extra if not a power of two size buffer
  int iterations = 0;
  for (size_t vs = vectorSize >> 1; vs > 0; vs >>= 1) {
    iterations++;
  }
  if ((vectorSize & (vectorSize - 1)) != 0) {
    iterations++;
  }
  // calculate the buffer to read from first, so we always finally write to bufO
  auto inBuf = &bufI;
  auto outBuf = &bufO;
  if (iterations % 2 != 0) {
    outBuf = &bufI;
    inBuf = &bufO;
  }
  // do a parallel shift right, and set the first element to the initial value.
  // this works, as an exclusive scan is equivalent to a shift
  // (with initial set at element 0) followed by an inclusive scan
  auto shr = [vectorSize, ndRange, inBuf, outBuf, init](
      cl::sycl::handler &h) {
    auto aI = inBuf->template get_access<cl::sycl::access::mode::read>(h);
    auto aO = outBuf->template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        ndRange, [aI, aO, init, vectorSize](cl::sycl::nd_item<1> id) {
          size_t m_id = id.get_global_id(0);
          if (m_id > 0) {
            aO[m_id] = aI[m_id - 1];
          } else {
            aO[m_id] = init;
          }
        });
  };
  q.submit(shr);
  // swap the buffers so we read from the buffer with the shifted contents
  std::swap(inBuf, outBuf);
  // perform an inclusive scan on the shifted array
  int i = 1;
  do {
    auto f = [vectorSize, i, ndRange, inBuf, outBuf, bop](
        cl::sycl::handler &h) {
      auto aI =
          inBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      auto aO =
          outBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<
          cl::sycl::helpers::NameGen<1, typename ExecutionPolicy::kernelName> >(
          ndRange, [aI, aO, bop, vectorSize, i](cl::sycl::nd_item<1> id) {
            size_t td = 1 << (i - 1);
            size_t m_id = id.get_global_id(0);
            if (m_id < vectorSize && m_id >= td) {
              aO[m_id] = bop(aI[m_id - td], aI[m_id]);
            } else {
              aO[m_id] = aI[m_id];
            }
          });
    };
    q.submit(f);
    // swap the buffers between iterations
    std::swap(inBuf, outBuf);
    i++;
  } while (i <= iterations);
  q.wait_and_throw();
  return o + vectorSize;
}

#else


template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename T,
          typename BinaryOperation>
OutputIterator exclusive_scan(ExecutionPolicy &snp, InputIterator b,
                              InputIterator e, OutputIterator o, T init,
                              BinaryOperation bop) {

  cl::sycl::queue q(snp.get_queue());
  auto device = q.get_device();
  auto size = sycl::helpers::distance(b, e);
  using value_type = typename std::iterator_traits<InputIterator>::value_type;
#ifdef TRISYCL_CL_LANGUAGE_VERSION
  std::vector<value_type> vect { b, e };
  *o++ = init;
#endif


  {
#ifdef TRISYCL_CL_LANGUAGE_VERSION
    cl::sycl::buffer<value_type, 1> buffer { vect.data(), size - 1 };
    buffer.set_final_data(o);
#else
    std::shared_ptr<value_type> data { new value_type[size-1],
      [&](value_type* ptr) {
        *o++ = init;
        std::copy_n(ptr, size-1, o);
      }
    };
    std::copy_n(b, size-1, data.get());
    cl::sycl::buffer<value_type, 1> buffer { data, cl::sycl::range<1>{size-1} };
#endif

    auto d = compute_mapscan_descriptor(device, size - 1, sizeof(value_type));
    buffer_mapscan(snp, q, buffer, buffer, init, d,
                   [](value_type x) { return x; },
                   bop);
  }

  return std::next(o, size - 1);
}

#endif

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_EXCLUSIVE_SCAN__
