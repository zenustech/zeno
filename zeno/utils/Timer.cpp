#ifdef ZENO_BENCHMARK
#include <zeno/utils/Timer.h>
#include <zeno/utils/zlog.h>
#include <iostream>

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
    int ns = std::chrono::duration_cast
        <std::chrono::nanoseconds>(diff).count();
    zlog::trace("** Leave [{}] spent {} ns", tag, ns);
    records.emplace_back(std::move(tag), ns);
}

Timer *Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;

void Timer::print() {
    std::cerr << "=== Begin ZENO Timing Statistics ===" << std::endl;
    for (auto const &[tag, ns]: records) {
        std::cerr << "[" << tag << "] " << ns << " ns" << std::endl;
    }
    std::cerr << "==== End ZENO Timing Statistics ====" << std::endl;
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
