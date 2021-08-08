#ifdef ZENO_BENCHMARK
#include <zeno/utils/Timer.h>
#include <zeno/utils/zlog.h>
#include <iostream>

namespace zeno {

Timer::Timer(std::string_view &&tag_, ClockType::time_point &&beg_)
    : parent(current)
    , beg(ClockType::now())
    , tag(current ? current->tag + '/' + (std::string)tag_ : tag_)
{
    current = this;
}

Timer::_destroy(ClockType::time_point &&end) {
    current = parent;
    auto end = ClockType::now();
    auto diff = end - beg;
    int ms = std::chrono::duration_cast
        <std::chrono::microseconds>(diff).count();
    zlog::info("{}: {}", tag, ms);
    records.emplace_back(std::move(tag), ms);
}

Timer *Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;

void Timer::print() {
    for (auto const &[tag, ms]: records) {
        std::cout << tag << ": " << ms << std::endl;
    }
}

}
#endif
