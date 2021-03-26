#include <zen/zen.h>
#include <chrono>
#include <thread>

namespace zenbase {

struct SleepMilis : zen::INode {
  virtual void apply() override {
    int ms = std::get<int>(get_param("ms"));
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }
};

static int defSleepMilis = zen::defNodeClass<SleepMilis>("SleepMilis",
    { /* inputs: */ {
    }, /* outputs: */ {
    }, /* params: */ {
        {"int", "ms", "1000"},
    }, /* category: */ {
        "misc",
    }});

}
