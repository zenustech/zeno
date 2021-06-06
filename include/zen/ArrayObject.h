#pragma once

#include <zen/zen.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace zen {

// https://www.cnblogs.com/JiangOil/p/11130670.html
// Below is the same as pybind11::buffer_info
struct ArrayObject : IObject {
  void *ptr;
  ssize_t itemsize;
  std::string format;
  ssize_t ndim;
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;
};

}
