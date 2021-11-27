#pragma once


#include <z2/dop/DopFunctor.h>


namespace z2::dop {


struct DopDescriptor {
    struct SocketInfo {
        std::string name;
    };

    struct CategoryInfo {
        std::string category;
        std::string documentation;
    };

    CategoryInfo cate;
    ztd::vector<SocketInfo> inputs;
    ztd::vector<SocketInfo> outputs;

    DopFunctor func;
};


}  // namespace z2::dop
