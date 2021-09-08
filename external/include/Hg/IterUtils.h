#pragma once

#include <string>
#include <vector>


namespace hg {

template <class T, class S>
static inline T contains(T const &elms, S const &key) {
  return elms.find(key) != elms.end();
}

template <class T, class S>
static inline T assign_conv(S const &data) {
  T ret;
  ret.assign(data.begin(), data.end());
  return ret;
}

}
