#include "timer.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <map>

namespace zinc {

Timer::Timer(std::string_view &&tag_, Timer::ClockType::time_point &&beg_)
    : parent(current)
    , beg(ClockType::now())
    , tag(current ? current->tag + " => " + (std::string)tag_ : tag_)
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
    if (records.size() == 0) {
        return;
    }

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
        stat.max_us = std::max(stat.max_us, us);
        stat.min_us = !stat.count_rec ? us : std::min(stat.min_us, us);
        stat.count_rec++;
    }

    std::vector<std::pair<std::string, Statistic>> sortstats;
    for (auto const &kv: stats) {
        sortstats.push_back(kv);
    }
    std::sort(sortstats.begin(), sortstats.end(),
    [&] (auto const &lhs, auto const &rhs) {
        return lhs.second.total_us > rhs.second.total_us;
    });

    printf("   avg   |   min   |   max   |  total  | cnt | tag\n");
    for (auto const &[tag, stat]: sortstats) {
        printf("%9d|%9d|%9d|%9d|%5d| %s\n",
                stat.total_us / stat.count_rec,
                stat.min_us, stat.max_us, stat.total_us,
                stat.count_rec, tag.c_str());
    }

    printf("==== End ZINC Timing Statistics ====\n");
}

namespace {
    static struct TimerAtexitHelper {
        ~TimerAtexitHelper() {
            Timer::print();
        }
    } timerAtexitHelper;
}

}
