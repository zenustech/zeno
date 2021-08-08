#ifdef ZENO_BENCHMARK
#include <zeno/utils/Timer.h>
#include <zeno/utils/zlog.h>
#include <iostream>
#include <map>

namespace zeno {

Timer::Timer(std::string_view &&tag_, Timer::ClockType::time_point &&beg_)
    : parent(current)
    , beg(ClockType::now())
    , tag(current ? current->tag + '/' + (std::string)tag_ : tag_)
{
    zlog::trace("** Enter [{}]", tag);
    current = this;
}

void Timer::_destroy(Timer::ClockType::time_point &&end) {
    current = parent;
    auto diff = end - beg;
    int us = std::chrono::duration_cast
        <std::chrono::microseconds>(diff).count();
    zlog::trace("** Leave [{}] spent {} us", tag, us);
    records.emplace_back(std::move(tag), us);
}

Timer *Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;

void Timer::print() {
    printf("=== Begin ZENO Timing Statistics ===\n");

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

    printf("   avg   |   min   |   max   |  total  | count  [tag]\n");
    for (auto const &[tag, stat]: stats) {
        printf("%9d|%9d|%9d|%9d|%7d [%s]\n",
                stat.total_us / stat.count_rec,
                stat.min_us, stat.max_us, stat.total_us,
                stat.count_rec, tag.c_str());
    }

    printf("==== End ZENO Timing Statistics ====\n");
}

namespace {
    struct AtexitHelper {
        ~AtexitHelper() {
            Timer::print();
        }
    } atexitHelper;
}

}
#endif
