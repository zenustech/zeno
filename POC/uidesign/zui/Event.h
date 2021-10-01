#pragma once


#include "stdafx.h"


GLFWwindow *window;


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
    int btn;  // lmb=0, mmb=1, rmb=2
    bool down;
};

struct Event_Motion {
    float x, y;
};

using Event = std::variant
    < Event_Key
    , Event_Char
    , Event_Hover
    , Event_Mouse
    , Event_Motion
    >;


struct SignalSlot {
    using Callback = std::function<void()>;
    std::vector<Callback> callbacks;

    void operator()() const {
        for (auto const &func: callbacks) {
            func();
        }
    }

    void connect(Callback &&f) {
        callbacks.push_back(std::move(f));
    }
};
