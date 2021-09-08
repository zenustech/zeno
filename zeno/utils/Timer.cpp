#ifdef ZENO_BENCHMARKING
#include <zinc/timer.cpp>
#include <spdlog/spdlog.h>

namespace {
    static struct TimerAtexitHelper {
        ~TimerAtexitHelper() {
            auto log = zinc::Timer::getLog();
            if (auto env = getenv("ZEN_TIMER"); env) {
                FILE *fp = fopen(env, "w");
                fprintf(fp, "%s", log.c_str());
                fclose(fp);
            } else if (log.size()) {
                printf("ZENO benchmarking status:\n%s\n", log.c_str());
            }
        }
    } timerAtexitHelper;
}
#endif
