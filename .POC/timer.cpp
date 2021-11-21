#include <chrono>
#include <string>
#include <vector>
#include <cassert>
#include <unistd.h>

using ClockType = std::chrono::high_resolution_clock;

struct Timer {
    struct Record {
        std::string tag;
        int ms;

        Record(std::string &&tag_, int ms_)
            : tag(std::move(tag_)), ms(ms_) {}
    };

    inline static Timer *current = nullptr;
    std::vector<Record> records;

    Timer *parent = nullptr;
    ClockType::time_point beg;
    ClockType::time_point end;
    std::string tag;

    Timer(std::string_view tag_)
        : parent(current)
        , beg(ClockType::now())
        , tag(current ? current->tag + '/' + (std::string)tag_ : tag_)
    {
        current = this;
    }

    ~Timer() {
        current = parent;
        auto end = ClockType::now();
        auto diff = end - beg;
        int ms = std::chrono::duration_cast
            <std::chrono::microseconds>(diff).count();
        records.emplace_back(std::move(tag), ms);
    }
};

int main() {
    {
        Timer _("mainloop");
        usleep(500);
        {
            Timer _("simulation");
            usleep(8000);
        }
        usleep(500);
        {
            Timer _("rendering");
            usleep(6000);
        }
        usleep(500);
    }
    return 0;
}
