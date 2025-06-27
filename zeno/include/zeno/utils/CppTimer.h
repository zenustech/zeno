//
// Created by zh on 2025/6/27.
//

#ifndef ZENO_CPPTIMER_H
#define ZENO_CPPTIMER_H
#include <ctime>
#include <string>

namespace zeno {

class CppTimer {
public:
    void tick() {
        struct timespec t;
        std::timespec_get(&t, TIME_UTC);
        last = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
    }
    void tock() {
        struct timespec t;
        std::timespec_get(&t, TIME_UTC);
        cur = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
    }
    float elapsed() const noexcept {
        return cur - last;
    }
    void tock(std::string_view tag) {
        tock();
        printf("%s: %f ms\n", tag.data(), elapsed());
    }

  private:
    double last = {};
    double cur = {};
};
}
#endif //ZENO_CPPTIMER_H
