#pragma once

#include <zeno/utils/Error.h>
#include <string_view>
#include <string>
#include <memory>

namespace zeno {

struct INode;

struct GlobalStatus {
    INode *node = nullptr;
    std::shared_ptr<Error> error;

    bool failed() const {
        return node != nullptr;
    }

    ZENO_API std::string toJson() const;
    ZENO_API void fromJson(std::string_view json);
};

}
