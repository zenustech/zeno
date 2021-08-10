#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cassert>

namespace zinc {

class Timer {
public:
    using ClockType = std::chrono::high_resolution_clock;

private:
    struct Record {
        std::string tag;
        int us;

        Record(std::string &&tag_, int us_)
            : tag(std::move(tag_)), us(us_) {}
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

#ifdef ZINC_TIMER_IMPLEMENTATION
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <map>

namespace zinc {

Timer::Timer(std::string_view &&tag_, Timer::ClockType::time_point &&beg_)
    : parent(current)
    , beg(ClockType::now())
    , tag(current ? current->tag + '/' + (std::string)tag_ : tag_)
{
    current = this;
}

void Timer::_destroy(Timer::ClockType::time_point &&end) {
    current = parent;
    auto diff = end - beg;
    int us = std::chrono::duration_cast
        <std::chrono::microseconds>(diff).count();
    records.emplace_back(std::move(tag), us);
}

Timer *Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;

void Timer::print() {
    printf("=== Begin ZINC Timing Statistics ===\n");

    struct Statistic {
        int max_us = 0;
        int min_us = 0;
        int total_us = 0;
        int count_rec = 0;
    };
    std::map<std::string, Statistic> stats;
    for (auto const &[tag, us]: records) {
        auto &stat = stats[tag];
        stat.total_us += us;
        stat.count_rec++;
        stat.max_us = std::max(stat.max_us, us);
        stat.min_us = stat.count_rec ? stat.min_us : std::min(stat.min_us, us);
    }

    std::vector<std::pair<std::string, Statistic>> sortstats;
    for (auto const &kv: stats) {
        sortstats.push_back(kv);
    }
    std::sort(sortstats.begin(), sortstats.end(),
    [&] (auto const &lhs, auto const &rhs) {
        return lhs.second.total_us > rhs.second.total_us;
    });

    printf("   avg   |   min   |   max   |  total  | count  [tag]\n");
    for (auto const &[tag, stat]: sortstats) {
        printf("%9d|%9d|%9d|%9d|%7d [%s]\n",
                stat.total_us / stat.count_rec,
                stat.min_us, stat.max_us, stat.total_us,
                stat.count_rec, tag.c_str());
    }

    printf("==== End ZINC Timing Statistics ====\n");
}

namespace {
    static struct TimerAtexitHelper {
        ~TimerAtexitHelper() {
            if (getenv("ZEN_PRINTSTAT"))
                Timer::print();
        }
    } timerAtexitHelper;
}

}
#endif
