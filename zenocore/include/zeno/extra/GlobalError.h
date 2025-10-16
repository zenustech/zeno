#pragma once

#include <zeno/utils/Error.h>
#include <string_view>
#include <string>
#include <memory>

namespace zeno {

struct NodeImpl;

struct GlobalError {

    ZENO_API GlobalError();
    ZENO_API GlobalError(std::string node, std::shared_ptr<Error> error, std::string param = "");
    ZENO_API GlobalError(const GlobalError& err);
    ZENO_API void set_node_info(const std::string& info) {
        m_nodeUuidPath = info;
    }
    ZENO_API void set_error(std::shared_ptr<Error> error) {
        m_error = error;
    }
    ZENO_API bool failed() const {
        return !m_nodeUuidPath.empty();
    }
    void clearState();
    ZENO_API std::shared_ptr<Error> getError() const;
    ZENO_API std::string getNode() const;
    ZENO_API std::string getErrorMsg() const;

private:
    std::string m_nodeUuidPath;
    std::string m_param;
    std::shared_ptr<Error> m_error;
};

}
