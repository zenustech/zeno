#pragma once

#include <zen/zen.h>
#include <vector>

namespace zen {

template <class T>
struct VectorObject : IObjectClone<VectorObject<T>> {
  std::vector<T> arr;
};

}
