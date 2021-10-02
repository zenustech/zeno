#pragma once


#include "DopLazy.h"


struct DopInputSocket {
    std::string name;
    std::string value;

    void serialize(std::ostream &ss) const {
        ss << name << "=" << value;
    }
};


struct DopOutputSocket {
    std::string name;
    DopLazy result;

    void serialize(std::ostream &ss) const {
        ss << name;
    }
};
