#pragma once
#include "PathSamplingLaunch.h"
#include <memory>

namespace zeno_path_sampling {
class Resources {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    Resources();
    ~Resources();
    Resources(const Resources&)=delete;
    Resources& operator=(const Resources&)=delete;
    Launch launch(unsigned mode,unsigned sampleIndex,unsigned seed);
};
}
