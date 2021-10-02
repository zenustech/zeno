#pragma once


#include <z2/ztd/stdafx.h>


namespace z2::dop {


struct DopDescriptor {
    struct SocketInfo {
        std::string name;
    };

    std::vector<SocketInfo> inputs;
    std::vector<SocketInfo> outputs;
};


}  // namespace z2::dop
