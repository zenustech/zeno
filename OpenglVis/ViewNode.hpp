#pragma once

#include <zen/zen.h>
#include <set>

namespace zenvis {

struct IViewNode : zen::INode {
  static std::set<IViewNode *> instances;

  IViewNode() {
    instances.insert(this);
  }

  ~IViewNode() {
    instances.erase(this);
  }

  virtual void draw() = 0;
};

}
