#pragma once

#include <zeno/zeno.h>
#include <zeno/vec.h>

namespace zeno {


struct ConditionObject : IObjectClone<ConditionObject> {
  bool value = true;

  bool get() {
    return value;
  }

  void set(bool x) {
    value = x;
  }
};

}
