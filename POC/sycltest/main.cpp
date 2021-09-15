#include <memory>
#include <array>
#include "sycl_sink.h"


using namespace fdb;


template <class T, size_t PotBlkSize, size_t Dim = 1>
struct PointerArray {
    Vector<T> m_data;
    NDArray<size_t, Dim> m_offset;

    template <auto Mode = Access::read_write, class Handler>
    auto accessor(Handler hand) {
        auto dataAxr = m_data.template accessor<Mode>(hand);
        auto offsetAxr = m_offset.template accessor<Mode>(hand);
        return [=] (vec<Dim, size_t> indices) -> typename decltype(dataAxr)::ReferenceT {
            size_t offset = offsetAxr(indices >> PotBlkSize);
            return dataAxr(offset);
        };
    }
};


class kernel0;


int main() {
    Vector<float> arr(4);
    arr.resize(16);

    fdb::enqueue([&] (fdb::DeviceHandler dev) {
        auto arrAxr = arr.accessor<fdb::Access::discard_write>(dev);
        dev.parallelFor<kernel0, 1>(arr.size(), [=] (size_t id) {
            arrAxr(id) = id;
        });
    });

    arr.resize(4);
    arr.resize(16);
    {
        auto arrAxr = arr.accessor<fdb::Access::read>(fdb::host);
        for (int i = 0; i < 16; i++) {
            printf(" %.3f", arrAxr(i));
        }
        printf("\n");
    }

    return 0;
}
