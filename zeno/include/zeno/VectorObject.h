#pragma once

#include <zeno/zeno.h>
#include <vector>

namespace zeno {

template <class T>
struct VectorObject : IObjectClone<VectorObject<T>> {
  std::vector<T> arr;
};

}
