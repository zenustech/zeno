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

    SignalInst(SignalObject const &) = delete;
    SignalInst &operator=(SignalObject const &) = delete;
    SignalInst(SignalObject &&) = default;
    SignalInst &operator=(SignalObject &&) = default;

    ~SignalInst() {
        for (auto const &func: callbacks) {
            func();
        }
    }

    void connect(Callback &&f) {
        callbacks.push_back(std::move(f));
    }
};


struct SignalSlot {
    using Callback = std::function<void()>;
    std::list<Callback> callbacks;

    void operator()() const {
        for (auto const &func: callbacks) {
            func();
        }
    }

    void connect(Callback &&f) {
        auto it = callbacks.insert(std::move(f));
    }

    void connect(Callback &&f, SignalInst &inst) {
        auto it = callbacks.insert(std::move(f));
        inst.connect([=] () {
            callbacks.erase(it);
        });
    }
};


}  // namespace zeno2::UI
