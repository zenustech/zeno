#pragma once

#include <zeno/core/IObject.h>
#include <string>

namespace zeno {

struct VoidPtrObject : IObjectClone<VoidPtrObject> {
  void *value;

  VoidPtrObject(void *value) : value(value) {}

  void *get() const {
    return value;
  }

  void set(void *x) {
    value = x;
  }
};

}
