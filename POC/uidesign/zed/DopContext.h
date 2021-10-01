#pragma once


#include "stdafx.h"


struct DopInputSocket {
    std::string name;
    std::string value;

    void serialize(std::ostream &ss) const {
        ss << name << "=" << value;
    }
};


struct DopOutputSocket {
    std::string name;
    std::any result;

    void serialize(std::ostream &ss) const {
        ss << name;
    }
};


struct DopContext {
    ztd::Vector<std::any> in;
    ztd::Vector<std::any> out;
};

using DopFunctor = std::function<void(DopContext *)>;
