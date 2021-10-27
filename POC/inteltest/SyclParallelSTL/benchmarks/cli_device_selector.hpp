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

/** class cli_device_selector.
* @brief Looks for an INTEL cpu among the available CPUs.
* if it finds an INTEL CPU it will return an 1, otherwise it returns a -1.
*/
class cli_device_selector : public cl::sycl::device_selector {
  std::string m_vendor_name;
  std::string m_device_type;

  static cl::sycl::info::device_type match_device_type(std::string requested) {
    if (requested.empty()) return cl::sycl::info::device_type::automatic;
    std::transform(requested.begin(), requested.end(), requested.begin(),
                   ::tolower);
    if (requested == "gpu") return cl::sycl::info::device_type::gpu;
    if (requested == "cpu") return cl::sycl::info::device_type::cpu;
    if (requested == "accel") return cl::sycl::info::device_type::accelerator;
    if (requested == "*" || requested == "any")
      return cl::sycl::info::device_type::all;

    return cl::sycl::info::device_type::automatic;
  }

 public:
  cli_device_selector(std::string vendor_name, std::string device_type)
      : cl::sycl::device_selector(),
        m_vendor_name(vendor_name),
        m_device_type(device_type) {}

  int operator()(const cl::sycl::device &device) const {
    int score = 0;

    // Score the device type...
    cl::sycl::info::device_type dtype =
        device.get_info<cl::sycl::info::device::device_type>();
    cl::sycl::info::device_type rtype = match_device_type(m_device_type);
    if (rtype == dtype || rtype == cl::sycl::info::device_type::all) {
      score += 2;
    } else if (rtype == cl::sycl::info::device_type::automatic) {
      score += 1;
    } else {
      score -= 2;
    }

    // score the vendor name
    cl::sycl::platform plat = device.get_platform();
    std::string name = plat.template get_info<cl::sycl::info::platform::name>();
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (name.find(m_vendor_name) != std::string::npos &&
        !m_vendor_name.empty()) {
      score += 2;
    } else if (m_vendor_name == "*" || m_vendor_name.empty()) {
      score += 1;
    } else {
      score -= 2;
    }
    return score;
  }
};

#endif  // __INTEL_CPU_SELECTOR__
