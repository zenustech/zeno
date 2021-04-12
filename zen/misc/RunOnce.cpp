#include <zen/zen.h>

namespace zenbase {

struct RunOnce : zen::INode {
  bool is_first_time{true};

  virtual void apply() override {
    auto cond = zen::IObject::make<zen::BooleanObject>();
    cond->value = is_first_time;
    is_first_time = false;
    set_output("cond", cond);
  }
};

static int defRunOnce = zen::defNodeClass<RunOnce>("RunOnce",
    { /* inputs: */ {
    }, /* outputs: */ {
        "cond",
    }, /* params: */ {
    }, /* category: */ {
        "misc",
    }});

}
