#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cassert>

namespace zeno {

class Timer {
public:
    using ClockType = std::chrono::high_resolution_clock;

private:
    struct Record {
        std::string tag;
        int ms;

        Record(std::string &&tag_, int ms_)
            : tag(std::move(tag_)), ms(ms_) {}
    };

    static Timer *current;
    static std::vector<Record> records;

    Timer *parent = nullptr;
    ClockType::time_point beg;
    ClockType::time_point end;
    std::string tag;

    Timer(std::string_view &&tag, ClockType::time_point &&beg);
    void _destroy(ClockType::time_point &&end);

public:
    Timer(std::string_view tag_) : Timer(std::move(tag_), ClockType::now()) {}
    ~Timer() { _destroy(ClockType::now()); }

    static void print();
};

}
