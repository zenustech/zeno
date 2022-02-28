#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cassert>

namespace zeno {

class Timer {
public:
    using ClockType = std::chrono::high_resolution_clock;

    struct Record {
        std::string tag;
        int us;

        Record(std::string &&tag_, int us_)
            : tag(std::move(tag_)), us(us_) {}
    };

private:
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

    static auto const &getRecords() { return records; }
    static std::string getLog();
};

#define ZINC_FUNC_TIMER ::zeno::Timer _zeno_timer(__func__);
#define ZINC_PRETTY_TIMER ::zeno::Timer _zeno_timer(__PRETTY_FUNCTION__);

}
