#include <zen/zen.h>
#include <cstdio>

namespace zenbase {

struct MyNode : zen::INode {
  virtual void apply() override {
    printf("MyNode::apply() called!\n");
  }
};

int myfunc() {
	printf("demo_project: %p\n", &zen::getSession());
	return 0;
}

static int myfuncval = myfunc();

static int defMyNode = zen::defNodeClass<MyNode>("MyNode",
    { /* inputs: */ {
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "demo_project",
    }});

}
