#pragma once


#include <zeno/common.h>
#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>


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

    std::vector<_Node> _table{4};
    std::size_t _tabmask{3};
    std::mutex _mtx;

    constexpr std::size_t _capacity() const noexcept {
        return _tabmask + 1;
    }

    void _recapacity(std::size_t size) noexcept {
        if (size == 0)
            size = 1;
        _table.resize(size);
        _tabmask = size - 1;
    }

    static inline constexpr std::size_t _hash(std::uint32_t key) noexcept {
        return std::size_t{static_cast<std::int32_t>(key * 1.618033988749895f)};
    }

    T &operator[](std::uint32_t key) noexcept {
        std::size_t base = _tabmask & _hash(key), end = base + _tabmask + 1;
        for (std::size_t slot = base; slot != end; slot++) {
            std::size_t slt = _tabmask & slot;
            std::uint32_t prev = _kEmpty;
            if (_table[slt].key.compare_exchange_strong(prev, key) || prev == key) {
                return _table[slt].value;
            }
        }
        std::lock_guard _(_mtx);
        _table.resize((_tabmask + 1) << 1);
        return *reinterpret_cast<T *>(0);
    }

    T &at(std::uint32_t key) noexcept {
        std::size_t base = _tabmask & _hash(key), end = base + _tabmask + 1;
        for (std::size_t slot = base; slot != end; slot++) {
            std::size_t slt = _tabmask & slot;
            if (_table[slt].key.load() == key) {
                return _table[slt].value;
            }
        }
        return *reinterpret_cast<T *>(0);
    }
};


}
ZENO_NAMESPACE_END
