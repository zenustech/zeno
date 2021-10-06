#pragma once


#include <z2/ztd/stdafx.h>


namespace z2::dop {


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


}  // namespace z2::dop
