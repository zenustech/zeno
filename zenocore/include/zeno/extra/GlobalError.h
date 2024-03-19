#pragma once

#include <zeno/utils/Error.h>
#include <string_view>
#include <string>
#include <memory>
#include <zeno/core/data.h>

namespace zeno {

struct INode;

struct GlobalError {

    ZENO_API GlobalError();
    ZENO_API GlobalError(ObjPath node, std::shared_ptr<Error> error, std::string param = "");
    ZENO_API GlobalError(const GlobalError& err);

    ZENO_API bool failed() const {
        return !m_namePath.empty();
    }
    void clearState();
    ZENO_API std::shared_ptr<Error> getError() const;
    ZENO_API ObjPath getNode() const;
    ZENO_API std::string getErrorMsg() const;

private:
    ObjPath m_namePath;
    std::string m_param;
    std::shared_ptr<Error> m_error;
};

}
