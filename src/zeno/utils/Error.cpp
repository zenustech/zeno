#include <zeno/utils/Error.h>
#include <zeno/utils/cppdemangle.h>

namespace zeno {

ZENO_API TypeError::TypeError(std::type_info const &expect, std::type_info const &got, std::string const &hint) noexcept
    : Exception((std::string)"expect `" + cppdemangle(expect) + "` got `" + cppdemangle(got) + "` (" + hint + ")")
    , expect(expect)
    , got(got)
    , hint(hint)
{
}

ZENO_API KeyError::KeyError(std::string const &key, std::string const &type, std::string const &hint) noexcept
    : Exception((std::string)"invalid " + type + " name `" + key + "` (" + hint + ")")
    , key(key)
    , type(type)
    , hint(hint)
{
}

}
