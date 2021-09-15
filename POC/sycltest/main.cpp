#include <memory>
#include <array>
#include "sycl_sink.h"
#include "L1PointerMap.h"


using namespace fdb;


template <class T>
struct default_minus1 {
    T m;

    default_minus1() : m((T)-1) {}
    default_minus1(T const &t)
        : m(t) {}

    operator T const &() const { return m; }
    operator T &() { return m; }

    T const &value() const { return m; }
    T &value() { return m; }

    bool has_value() const {
        return m != (T)-1;
    }
};


class kernel0;


int main() {
#if 1
    L1PointerMap<float, 1, 2, 3> arr;

    fdb::enqueue([&] (fdb::DeviceHandler dev) {
        auto arrAxr = arr.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0, 1>(arr.size(), [=] (size_t id) {
            *arrAxr(id) = id;
        });
    });

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::HOST);
        for (int i = 0; i < arr.size(); i++) {
            printf(" %.3f", *arrAxr(i));
        }
        printf("\n");
    }
#else
    NDBuffer<size_t> arr(32);
    arr.construct(FDB_BAD_OFFSET);

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::HOST);
        for (int i = 0; i < arr.shape(); i++) {
            printf(" %zd", (size_t)*arrAxr(i));
        }
        printf("\n");
    }
#endif

    return 0;
}
