#pragma once


#include <zeno/common.h>
#include <cstdint>
#include <vector>
#include <atomic>


ZENO_NAMESPACE_BEGIN
namespace zbb {


// https://nosferalatu.com/SimpleGPUHashTable.html
template <class T>
struct concurrent_u32_map {
    struct _Node {
        std::atomic<std::uint32_t> key{};
        T value{};
    };

    static constexpr std::uint32_t _kEmpty = 0;

    std::vector<_Node> _table;
    std::size_t _tabmask{};

    static inline constexpr std::size_t _hash(std::uint32_t key) {
        return static_cast<std::size_t>(key * 1.618033988749895f);
    }

    T &put(std::uint32_t key) {
        std::size_t base = _tabmask & _hash(key), end = base + _tabmask + 1;
        for (std::size_t slot = base; slot != end; slot++) {
            std::size_t slt = _tabmask & slot;
            std::uint32_t prev = _kEmpty;
            if (_table[slt].key.compare_exchange_strong(prev, key)) {
                return _table[slt].value;
            } else if (prev == key) {
                return _table[slt].value;
            }
        }
    }
};


}
ZENO_NAMESPACE_END
