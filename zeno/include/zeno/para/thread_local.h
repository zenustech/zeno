#pragma once

#ifdef ZENO_PARALLEL_STL

#include <zeno/para/execution.h>
#include <thread>
#include <mutex>
#include <map>

namespace zeno {

template <class Value>
struct thread_local_storage {
    using value_type = Value;
    using reference = Value &;

private:
    using MapT = std::map<std::thread::id, value_type>;
public:
    using const_iterator = typename MapT::const_iterator;
    using iterator = const_iterator;

private:
    MapT m_lut;
    std::mutex m_lut_mtx;

public:
    reference local() {
        std::lock_guard lck(m_lut_mtx);
        return m_lut[std::this_thread::get_id()];
    }

    const_iterator begin() const {
        return m_lut.begin();
    }

    const_iterator end() const {
        return m_lut.end();
    }
};

/* usage:

thread_local_storage<vector<int>> poses;

parallel_for(0, 100, [&] (int index) {
    auto &pos = poses.local();
    pos.push_back(index);
});

vector<int> zspos;
for (auto const &pos: poses) {
    zspos.insert(zspos.end(), pos.begin(), pos.end());
}

 * alternative:

static auto last_id = std::this_thread::get_id();
assert(std::this_thread::get_id() == last_id);

static thread_local_storage<vector<int>> poses;

parallel_for(0, 100, [&] (int index) {
    thread_local auto &pos = poses.local();
    pos.push_back(index);
});

vector<int> zspos;
for (auto const &pos: poses) {
    zspos.insert(zspos.end(), pos.begin(), pos.end());
}

 */

}

#else

namespace zeno {

template <class Value>
struct thread_local_storage {
    using value_type = Value;
    using reference = Value &;

private:
    value_type m_val;

public:
    reference local() {
        return m_val;
    }
};

}

#endif
