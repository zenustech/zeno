#pragma once

#include <zen/zen.h>
#include <zen/vec.h>

namespace zen {


struct ConditionObject : IObject {
  bool value = true;

  bool get() {
    return value;
  }

  void set(bool x) {
    value = x;
  }
};

}
