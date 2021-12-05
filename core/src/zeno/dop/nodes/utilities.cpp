#include <zeno/dop/dop.h>


ZENO_NAMESPACE_BEGIN
namespace {


struct TwiceInt : dop::Node {
    virtual void apply() override {
        auto val = value_cast<int>(get_input(0));
        set_output(0, ztd::make_any(val * 2));
    }
};

ZENO_DOP_DEFCLASS(TwiceInt, {{
    "numeric", "times an integer by two",
}, {
    {"value", "int"},
}, {
    {"value", "int"},
}});


struct PrintInt : dop::Node {
    virtual void apply() override {
        auto val = value_cast<int>(get_input(0));
        printf("Print %d\n", val);
    }
};

ZENO_DOP_DEFCLASS(PrintInt, {{
    "numeric", "prints a integer",
}, {
    {"value", "int"},
}, {
}});


struct Route : dop::Node {
    virtual void apply() override {
        set_output(0, get_input(0));
    }
};

ZENO_DOP_DEFCLASS(Route, {{
    "misc", "always return the first input",
}, {
    {"value", "any"},
}, {
    {"value", "any"},
}});


struct ToView : dop::Node {
    virtual void apply() override {
        set_output(0, get_input(0));
    }
};

ZENO_DOP_DEFCLASS(ToView, {{
    "misc", "send object to be viewed",
}, {
    {"object", "any"},
}, {
    {"object", "any"},
}});


}
ZENO_NAMESPACE_END
