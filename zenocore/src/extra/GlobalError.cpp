#include <zeno/extra/GlobalError.h>
#include <zeno/core/INode.h>
#include <zeno/utils/log.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace zeno {

    ZENO_API GlobalError::GlobalError() {
    }

    ZENO_API GlobalError::GlobalError(ObjPath node, std::shared_ptr<Error> error, std::string param)
        : m_namePath(node)
        , m_error(error)
        , m_param(param)
    {
    }

    ZENO_API GlobalError::GlobalError(const GlobalError& err)
        : m_namePath(err.m_namePath)
        , m_error(err.m_error)
        , m_param(err.m_param)
    {
    }

    void GlobalError::clearState() {
        m_namePath.clear();
        m_error.reset();
    }

    ZENO_API std::shared_ptr<Error> GlobalError::getError() const {
        return m_error;
    }

    ZENO_API std::string GlobalError::getErrorMsg() const {
        return m_error->message;
    }

    ZENO_API ObjPath GlobalError::getNode() const {
        return m_namePath;
    }
}
