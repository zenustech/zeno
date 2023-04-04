#pragma once

namespace zeno {
template <typename K, typename V>
class Pair {
public:
    Pair(const K& key, const V& value) {
        m_key = key;
        m_value = value;
    }

    K m_key;
    V m_value;
};
}
