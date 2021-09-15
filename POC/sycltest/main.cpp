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


#define FDB_BAD_VALUE ((size_t)-1)


template <class T, size_t PotBlkSize, size_t Dim>
struct L1PointerMap {
    Vector<T> m_data;
    NDArray<size_t, Dim> m_offset1;

    explicit L1PointerMap(vec<Dim, size_t> shape = {0})
        : m_offset1((shape + (1 << PotBlkSize) - 1) >> PotBlkSize)
        , m_data(1 << PotBlkSize)
    {
        m_offset1.construct(FDB_BAD_VALUE);
    }

    auto shape() const {
        return m_offset1.shape() << PotBlkSize;
    }

    void reshape(vec<Dim, size_t> shape) {
        m_data.clear();
        m_offset1.reshape((shape + (1 << PotBlkSize) - 1) >> PotBlkSize);
        m_offset1.construct(FDB_BAD_VALUE);
    }

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto dataAxr = m_data.template accessor<Mode>(hand);
        auto offset1Axr = m_offset1.template accessor<Access::read>(hand);
        return [=] (vec<Dim, size_t> indices) -> T * {
            auto offset1 = *offset1Axr(indices >> PotBlkSize);
            if (offset1 == FDB_BAD_VALUE)
                return nullptr;
            offset1 *= 1 << (Dim * PotBlkSize);
            size_t offset0 = indices & ((1 << PotBlkSize) - 1);
            return dataAxr(offset1 | offset0);
        };
    }
};


class kernel0;


int main() {
#if 1
    L1PointerMap<float, 2, 1> arr(32);

    fdb::enqueue([&] (fdb::DeviceHandler dev) {
        auto arrAxr = arr.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0, 1>(arr.shape(), [=] (size_t id) {
            *arrAxr(id) = id;
        });
    });

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < arr.shape(); i++) {
            printf(" %p", arrAxr(i));
            printf(" %.3f", *arrAxr(i));
        }
        printf("\n");
    }
#else
    NDArray<size_t> arr(32);
    arr.construct(FDB_BAD_VALUE);

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
