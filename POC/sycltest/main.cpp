#include <memory>
#include <array>
#include "sycl_sink.h"


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


#define FDB_BAD_OFFSET ((size_t)-1)


template <class T, size_t Dim, size_t N0, size_t N1>
struct L1PointerMap {
    Vector<T> m_data;
    NDArray<size_t, 1 << N1> m_offset1;

    L1PointerMap()
        : m_data(4 << N0)
        , m_offset1(FDB_BAD_OFFSET)
    {
    }

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto dataAxr = m_data.template accessor<Mode>(hand);
        auto offset1Axr = m_offset1.template accessor<Access::read>(hand);
        return [=] (vec<Dim, size_t> indices) -> T * {
            auto offset1 = *offset1Axr(indices >> N0);
            if (offset1 == FDB_BAD_OFFSET)
                return nullptr;
            offset1 *= 1 << (Dim * N0);
            size_t offset0 = indices & ((1 << N0) - 1);
            return dataAxr(offset1 | offset0);
        };
    }

    static inline constexpr auto size() {
        return vec<Dim, size_t>(1 << (N0 + N1));
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
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < arr.size(); i++) {
            printf(" %.3f", *arrAxr(i));
        }
        printf("\n");
    }
#else
    NDBuffer<size_t> arr(32);
    arr.construct(FDB_BAD_OFFSET);

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < arr.shape(); i++) {
            printf(" %zd", (size_t)*arrAxr(i));
        }
        printf("\n");
    }
#endif

    return 0;
}
