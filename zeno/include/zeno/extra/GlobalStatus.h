#pragma once

#include <zeno/utils/Error.h>
#include <string_view>
#include <string>
#include <memory>

namespace zeno {

struct INode;

struct GlobalStatus {
    std::string nodeName;
    std::shared_ptr<Error> error;

    bool failed() const {
        return !nodeName.empty();
    }

    ZENO_API void clearState();
    ZENO_API std::string toJson() const;
    ZENO_API void fromJson(std::string_view json);
};

}
