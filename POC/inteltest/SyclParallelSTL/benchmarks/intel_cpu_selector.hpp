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

#ifndef __INTEL_CPU_SELECTOR__
#define __INTEL_CPU_SELECTOR__

#include <CL/sycl.hpp>
#include <string>
#include <iostream>

/** class intel_cpu_selector.
* @brief Looks for an INTEL cpu among the available CPUs.
* if it finds an INTEL CPU it will return an 1, otherwise it returns a -1.
*/
class intel_cpu_selector : public cl::sycl::device_selector {
 public:
  intel_cpu_selector() : cl::sycl::device_selector() {}

  int operator()(const cl::sycl::device &device) const {
    int res = -1;
    if (device.is_host()) {
      res = -1;
    } else {
      cl::sycl::info::device_type type =
          device.get_info<cl::sycl::info::device::device_type>();
      if (type == cl::sycl::info::device_type::cpu) {
        cl::sycl::platform plat = device.get_platform();
        std::string name = plat.get_info<cl::sycl::info::platform::name>();
        if (name.find("Intel") != std::string::npos) {
          res = 1;
        }
      }
    }
    return res;
  }
};

#endif  // __INTEL_CPU_SELECTOR__
