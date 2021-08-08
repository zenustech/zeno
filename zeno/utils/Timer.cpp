#ifdef ZENO_BENCHMARK
#include <zeno/utils/Timer.h>
#include <iostream>

namespace zeno {

Timer *Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;

void Timer::print() {
    for (auto const &[tag, ms]: records) {
        std::cout << tag << ": " << ms << std::endl;
    }
}

}
#endif
