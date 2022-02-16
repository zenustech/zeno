#include <zeno/utils/Error.h>
#include <zeno/utils/cppdemangle.h>

namespace zeno {

ZENO_API ZenoException(std::unique_ptr<Error> &&err) noexcept
    : err(std::move(err)) {
}

ZENO_API ~ZenoException() = default;

ZENO_API char const *ZenoException::what() const noexcept
{
    return this->err->what().c_str();
}

ZENO_API Error(std::string_view message) noexcept
    : message(message) {
}

ZENO_API ~Error() = default;

ZENO_API std::string const &Error::what() const {
    return message;
}

ZENO_API StdError::StdError(const char *what) noexcept
    : Error((std::string)"e.what(): `" + what "`")
{
}

ZENO_API ~StdError() = default;

ZENO_API TypeError::TypeError(std::type_info const &expect, std::type_info const &got, std::string const &hint) noexcept
    : Error((std::string)"expect `" + cppdemangle(expect) + "` got `" + cppdemangle(got) + "` (" + hint + ")")
    , expect(expect)
    , got(got)
    , hint(hint)
{
}

ZENO_API ~TypeError() = default;

ZENO_API KeyError::KeyError(std::string const &key, std::string const &type, std::string const &hint) noexcept
    : Error((std::string)"invalid " + type + " name `" + key + "` (" + hint + ")")
    , key(key)
    , type(type)
    , hint(hint)
{
}

ZENO_API ~KeyError() = default;

}
