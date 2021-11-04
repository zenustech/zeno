#include <cstdio>

namespace ns1 {
    struct type {};

    void foo(type) {
        printf("ns1::foo\n");
    }
}

namespace ns2 {
    struct type {};

    void foo(type) {
        printf("ns2::foo\n");
    }
}

namespace ns3 {
    inline namespace ns3sub {
        struct type {};
    }

    void foo(type) {
        printf("ns3::foo\n");
    }
}

namespace ns4 {
    using ns2::type;

    void foo(type) {
        printf("ns4::foo\n");
    }
}

namespace ns5 {
    struct type : ns2::type {};

    void foo(type) {
        printf("ns5::foo\n");
    }
}

namespace ns6 {
    struct type : ns2::type {};
}

template <class T>
void testadl(T t) {
    foo(t);
}

int main() {
    testadl(ns1::type());
    testadl(ns2::type());
    testadl(ns3::type());
    testadl(ns4::type());
    testadl(ns5::type());
    testadl(ns6::type());
    return 0;
}
