#include <chrono>
#include <string>
#include <vector>
#include <unistd.h>

using ClockType = std::chrono::high_resolution_clock;

struct Entry {
    Entry *parent = nullptr;
    ClockType::time_point beg;
    ClockType::time_point end;
    std::string tag;
    std::vector<Entry *> children;

    Entry(ClockType::time_point &&beg_, std::string_view &&tag_)
        : beg(beg_)
        , tag(tag_)
    {}
};

Entry *current = nullptr;

void enter(std::string_view tag) {
    auto entry = new Entry(ClockType::now(), std::move(tag));
    entry->parent = current;
    if (current) current->children.emplace_back(entry);
    current = entry;
}

void leave() {
    auto entry = current;
    current = current->parent;

    auto end = ClockType::now();
    auto diff = end - entry->beg;
    int ms = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    printf("%s: %d\n", entry->tag.c_str(), ms);
}

int main() {
    enter("mainloop");
        enter("simulation");
            usleep(10000);
        leave();
        enter("rendering");
            usleep(7000);
        leave();
    leave();
    return 0;
}
