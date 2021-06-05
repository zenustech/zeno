#include <zen/zen.h>
#include <cstdio>

namespace zenbase {

struct MyNode : zen::INode {
  virtual void apply() override {
    printf("MyNode::apply() called!\n");
  }
};

static int defMyNode = zen::defNodeClass<MyNode>("MyNode",
    { /* inputs: */ {
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "demo_project",
    }});

}
