#include <zeno/utils/Error.h>
#include <zeno/utils/cppdemangle.h>

namespace zeno {

ZENO_API ErrorException::ErrorException(std::shared_ptr<Error> &&err) noexcept
    : err(std::move(err)) {
}

ZENO_API ErrorException::~ErrorException() = default;

ZENO_API char const *ErrorException::what() const noexcept {
    return this->err->what().c_str();
}

ZENO_API std::shared_ptr<Error> ErrorException::get() const noexcept {
    return err;
}

ZENO_API Error::Error(std::string_view message) noexcept
    : message(message) {
}

ZENO_API Error::~Error() = default;

ZENO_API std::string const &Error::what() const {
    return message;
}

static const char *get_eptr_what(std::exception_ptr const &eptr) {
    try {
        if (eptr) {
            std::rethrow_exception(eptr);
        }
    } catch (std::exception const &e) {
        return e.what();
    }
    return "(no error)";
}

ZENO_API StdError::StdError(std::exception_ptr &&eptr) noexcept
    : Error((std::string)"std::exception::what(): `" + get_eptr_what(eptr) + "`"), eptr(std::move(eptr))
{
}

ZENO_API StdError::~StdError() = default;

ZENO_API TypeError::TypeError(std::type_info const &expect, std::type_info const &got, std::string const &hint) noexcept
    : Error((std::string)"expect `" + cppdemangle(expect) + "` got `" + cppdemangle(got) + "` (" + hint + ")")
    , expect(expect)
    , got(got)
    , hint(hint)
{
}

ZENO_API TypeError::~TypeError() = default;

ZENO_API KeyError::KeyError(std::string const &key, std::string const &type, std::string const &hint) noexcept
    : Error((std::string)"invalid " + type + " name `" + key + "` (" + hint + ")")
    , key(key)
    , type(type)
    , hint(hint)
{
}

ZENO_API KeyError::~KeyError() = default;

}
