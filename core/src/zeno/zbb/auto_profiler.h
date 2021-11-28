#pragma once


#include <zeno/common.h>
#include <iostream>
#include <chrono>
#include <optional>


ZENO_NAMESPACE_BEGIN
namespace zbb {


class auto_profiler {
private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
public:
    auto_profiler(std::string name)
        : m_name(std::move(name))
        , m_beg(std::chrono::high_resolution_clock::now())
    {}

    ~auto_profiler() {
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
        std::cout << m_name << " : " << dur.count() << " musec\n";
    }
};


}
ZENO_NAMESPACE_END
