#pragma once

#include <zeno/para/execution.h>
#include <functional>
#include <algorithm>
#include <vector>

namespace zeno {

struct task_group {
    std::vector<std::function<void()>> m_tasks;

    void add(std::function<void()> task) {
        m_tasks.push_back(std::move(task));
    }

    void clear() {
        m_tasks.clear();
    }

    void run() {
        std::for_each(ZENO_PAR m_tasks.begin(), m_tasks.end(), [&] (auto &&f) {
            std::move(f)();
        });
    }
};

}
