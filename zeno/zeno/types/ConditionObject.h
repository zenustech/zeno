#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {


struct ConditionObject : IObjectClone<ConditionObject> {
  bool value;

  ConditionObject(bool value = true) : value(value) {
  }

  bool get() const {
    return value;
  }

  void set(bool x) {
    value = x;
  }

  operator bool() const { return get(); }
};

}
