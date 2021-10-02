#pragma once


#include <zeno/dop/stdafx.h>


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
