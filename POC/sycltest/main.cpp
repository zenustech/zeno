#include <memory>
#include <array>
#include "sycl_sink.h"


using namespace fdb;


template <class T, T Minus1 = (T)-1>
struct default_minus1 {
    T m{Minus1};

    default_minus1(T const &t)
        : m(t) {}

    operator T const &() const { return m; }
    operator T &() { return m; }

    T const &value() const { return m; }
    T &value() { return m; }

    bool has_value() {
        return m == Minus1;
    }
};


template <class T, size_t PotBlkSize, size_t Dim>
struct L1PointerMap {
    Vector<T> m_data;
    NDArray<default_minus1<size_t>, Dim> m_offset1;

    explicit L1PointerMap(vec<Dim, size_t> shape = {0})
        : m_offset1((shape + (1 << PotBlkSize) - 1) >> PotBlkSize)
        , m_data(1 << PotBlkSize)
    {}

    auto shape() const {
        return m_offset1.shape() << PotBlkSize;
    }

    void reshape(vec<Dim, size_t> shape) {
        return m_offset1.reshape((shape + (1 << PotBlkSize) - 1) >> PotBlkSize);
    }

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto dataAxr = m_data.template accessor<Mode>(hand);
        auto offset1Axr = m_offset1.template accessor<Mode>(hand);
        return [=] (vec<Dim, size_t> indices) {
            size_t offset1 = *offset1Axr(indices >> PotBlkSize);
            offset1 *= (1 << (Dim * PotBlkSize));
            size_t offset0 = indices & ((1 << PotBlkSize) - 1);
            return dataAxr(offset1 + offset0);
        };
    }
};


class kernel0;


int main() {
#if 0
    L1PointerMap<float, 2, 1> arr{32};

    fdb::enqueue([&] (fdb::DeviceHandler dev) {
        auto arrAxr = arr.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0, 1>(arr.shape(), [=] (size_t id) {
            *arrAxr(id) = id;
        });
    });

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < arr.shape(); i++) {
            printf(" %.3f", *arrAxr(i));
        }
        printf("\n");
    }
#else
    NDArray<default_minus1<int>> arr{32};

    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < arr.shape(); i++) {
            printf(" %d", (int)*arrAxr(i));
        }
        printf("\n");
    }
#endif

    return 0;
}
