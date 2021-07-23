#include "any.h"
#include <cstdio>

using namespace zeno;

struct IObject {
    using poly_base = IObject;

    virtual void hello() const {}
};

struct DemoObject : IObject {
    virtual void hello() const {
        printf("DemoObject\n");
    }
};

struct TestObject : IObject {
    virtual void hello() const {
        printf("TestObject\n");
    }
};

int main() {
    auto x = make_shared<DemoObject>();
    auto y = make_shared<TestObject>();
    return 0;
}

