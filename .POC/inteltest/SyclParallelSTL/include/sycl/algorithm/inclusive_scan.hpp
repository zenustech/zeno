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

#ifndef __SYCL_IMPL_ALGORITHM_INCLUSIVE_SCAN__
#define __SYCL_IMPL_ALGORITHM_INCLUSIVE_SCAN__

#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/algorithm/buffer_algorithms.hpp>

namespace sycl {
namespace impl {

#ifdef SYCL_PSTL_USE_OLD_ALGO
/* inclusive_scan.
 * Implementation of the command group that submits a inclusive_scan kernel.
 * The kernel is implemented as a lambda.
 */
 template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class T, class BinaryOperation>
OutputIterator inclusive_scan(ExecutionPolicy &sep, InputIterator b,
                              InputIterator e, OutputIterator o, T init,
                              BinaryOperation bop) {
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();
  // limits us to random access iterators :/
  *b = bop(*b, init);
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
  // calculate the buffer to read from first, based on modulo arithmetic of the
  // required iteration count so we always finally write to bufO
  // implementation based on "naive" implementation from
  // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
  auto inBuf = &bufI;
  auto outBuf = &bufO;
  if (iterations % 2 == 0) {
    outBuf = &bufI;
    inBuf = &bufO;
  }

  int i = 1;
  do {
    auto f = [vectorSize, i, ndRange, inBuf, outBuf, bop](
        cl::sycl::handler &h) {
      auto aI =
          inBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      auto aO =
          outBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<typename ExecutionPolicy::kernelName>(
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
    // swap the buffers
    std::swap(inBuf, outBuf);
    i++;
  } while (i <= iterations);
  q.wait_and_throw();
  return o + vectorSize;
}
#else

template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class T, class BinaryOperation>
OutputIterator inclusive_scan(ExecutionPolicy &snp, InputIterator b,
                              InputIterator e, OutputIterator o, T init,
                              BinaryOperation bop) {

  auto q = snp.get_queue();
  auto device = q.get_device();
  size_t size = sycl::helpers::distance(b, e);
  using value_type = typename std::iterator_traits<InputIterator>::value_type;
  {
#ifdef TRISYCL_CL_LANGUAGE_VERSION
    cl::sycl::buffer<value_type, 1> buffer { b, e };
    buffer.set_final_data(o);
#else
    std::shared_ptr<value_type> data { new value_type[size],
      [&](value_type* ptr) {
        std::copy_n(ptr, size, o);
      }
    };
    std::copy_n(b, size, data.get());
    cl::sycl::buffer<value_type, 1> buffer { data, cl::sycl::range<1>{ size } };
#endif

    auto d = compute_mapscan_descriptor(device, size, sizeof(value_type));
    buffer_mapscan(snp, q, buffer, buffer, init, d,
                   [](value_type x) { return x; },
                   bop);
  }

  return std::next(o, size);
}

#endif
}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_INCLUSIVE_SCAN__
