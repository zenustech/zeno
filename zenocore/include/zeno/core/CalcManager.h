#pragma once

#include <zeno/core/Descriptor.h>
#include <zeno/core/data.h>
#include <set>

namespace zeno {

class CalcManager {
public:
    CalcManager();
    ZENO_API void run();
    ZENO_API void mark_frame_change_dirty();
    ZENO_API void collect_removing_objs(std::string key);

private:
    std::set<std::string> removing_objs;
};

}