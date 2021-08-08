#include <zeno/utils/Timer.h>
#include <cassert>
#include <memory>

namespace zeno {

Timer *Timer::current = nullptr;
std::vector<Timer::Record> Timer::records;

}
