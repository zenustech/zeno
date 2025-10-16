#pragma once

#include <memory>

namespace zeno {
    //TODO: impl SharedPtr with abi compatible
    template<class T>
    using SharedPtr = std::shared_ptr<T>;
}