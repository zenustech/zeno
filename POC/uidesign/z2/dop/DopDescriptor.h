#pragma once


#include <z2/dop/DopFunctor.h>


namespace z2::dop {


struct DopDescriptor {
    struct SocketInfo {
        std::string name;

        SocketInfo(std::string const &name) : name(name) {}
    };

    ztd::vector<SocketInfo> inputs;
    ztd::vector<SocketInfo> outputs;

    DopFunctor func;
};


}  // namespace z2::dop
