#pragma once

#include <zeno/zeno.h>
#include <zeno/vec.h>

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
