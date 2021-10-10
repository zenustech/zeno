#pragma once


#include <zeno2/UI/stdafx.h>


namespace zeno2::UI {


struct Event_Key {
    int key;
    int mode;
    bool down;
};

struct Event_Char {
    unsigned int code;
};

struct Event_Hover {
    bool enter;
};

struct Event_Mouse {
    int btn;  // lmb=0, rmb=1, mmb=2
    bool down;
};

struct Event_Motion {
    float x, y;
};

struct Event_Scroll {
    float dx, dy;
};

using Event = std::variant
    < Event_Key
    , Event_Char
    , Event_Hover
    , Event_Mouse
    , Event_Motion
    , Event_Scroll
    >;


struct SignalInst {
    using Callback = std::function<void()>;
    std::vector<Callback> callbacks;

    SignalInst() = default;
    SignalInst(SignalInst const &) = delete;
    SignalInst &operator=(SignalInst const &) = delete;
    SignalInst(SignalInst &&) = default;
    SignalInst &operator=(SignalInst &&) = default;

    ~SignalInst() {
        for (auto const &func: callbacks) {
            func();
        }
    }

    void on_destroy(Callback &&f) {
        callbacks.push_back(std::move(f));
    }
};


struct SignalSlot {
    using Callback = std::function<void()>;
    std::shared_ptr<std::list<Callback>> callbacks;

    void operator()() const {
        for (auto const &func: *callbacks) {
            func();
        }
    }

    void connect(Callback &&f, SignalInst *inst) {
        auto it = callbacks->insert(callbacks->begin(), std::move(f));
        if (inst) {
            std::weak_ptr cbwp(callbacks);
            inst->on_destroy([=] {
                if (auto cb = cbwp.lock()) {
                    cb->erase(it);
                }
            });
        }
    }
};


}  // namespace zeno2::UI
