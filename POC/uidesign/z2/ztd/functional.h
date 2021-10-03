#pragma once


#include <z2/ztd/stdafx.h>


namespace z2::ztd {


struct CallOnDtor : std::function<void()> {
    using std::function<void()>::function;

    CallOnDtor(CallOnDtor const &) = delete;
    CallOnDtor &operator=(CallOnDtor const &) = delete;
    CallOnDtor(CallOnDtor &&) = default;
    CallOnDtor &operator=(CallOnDtor &&) = default;

    ~CallOnDtor() {
        std::function<void()>::operator()();
    }
};


template <class ...Fs>
struct Overloaded : Fs... {
    Overloaded(Fs ...&&fs)
        : Fs(std::forward<Fs>(fs))... {}

    using Fs::operator()...;
};

template <class ...Fs>
Overloaded(Fs ...&&fs) -> Overloaded<Fs...>;


};
