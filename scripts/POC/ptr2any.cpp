#include "any.h"
#include <cstdio>

using namespace zeno;

struct IObject {
    using poly_base = IObject;
    virtual void hello() const {
        printf("IObject\n");
    }
};

struct DemoObject : IObject {
    virtual void hello() const {
        printf("DemoObject\n");
    }
};

int main() {
    auto x = shared_any::make<DemoObject>();
    auto p = x.cast<IObject>();
    printf("%p\n", p);
    return 0;
}

