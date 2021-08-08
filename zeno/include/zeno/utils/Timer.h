#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cassert>

namespace zeno {

struct Timer {
    using ClockType = std::chrono::high_resolution_clock;

    struct Record {
        std::string tag;
        int ms;

        Record(std::string &&tag_, int ms_)
            : tag(std::move(tag_)), ms(ms_) {}
    };

    static Timer *current;
    static std::vector<Record> records;

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

    static void print();
};

}
