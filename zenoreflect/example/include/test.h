#include "data.h"
#include "reflect/core.hpp"
#include <cstddef>
#include <iostream>
#include "reflect/reflection.generated.hpp"

namespace zeno {

    class ZRECORD() ABC {
    public:
        ABC() = delete;
    };

}

using ZABC = zeno::ABC;

struct ZRECORD() Soo {
    void wow(int* qwq) {
        std::cout << (size_t)qwq << std::endl;
    }
};
