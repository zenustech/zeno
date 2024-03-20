#include <zeno/core/INode.h>

extern "C" {

struct ExampleNode : zeno::INode {
  void apply() override {
    printf("==============================\n");
    printf("hahaha, example node in the demo plugin has been successfully "
           "called.\n");
    printf("==============================\n");
  }
};

ExampleNode g_ExampleNode;
}