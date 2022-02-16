#pragma once

#include <zeno/utils/api.h>
#include <string_view>
#include <typeinfo>
#include <string>
#include <memory>

namespace zeno {

class Error {
private:
    std::string message;

public:
    ZENO_API explicit Error(std::string_view message) noexcept;
    ZENO_API virtual ~Error();
    ZENO_API std::string const &what() const;

    Error(Error const &) = delete;
    Error &operator=(Error const &) = delete;
    Error(Error &&) = delete;
    Error &operator=(Error &&) = delete;
};

class StdError : public Error {
public:
    ZENO_API explicit StdError(const char *what) noexcept;
    ZENO_API ~StdError() override;
};

class TypeError : public Error {
private:
    std::type_info const &expect;
    std::type_info const &got;
    std::string hint;
public:
    ZENO_API explicit TypeError(std::type_info const &expect, std::type_info const &got, std::string const &hint = "nohint") noexcept;
    ZENO_API ~TypeError() override;
};


class KeyError : public Error {
private:
    std::string key;
    std::string type;
    std::string hint;
public:
    ZENO_API explicit KeyError(std::string const &key, std::string const &type = "key", std::string const &hint = "nohint") noexcept;
    ZENO_API ~KeyError() override;
};

class ZenoException : public std::exception {
private:
    std::unique_ptr<Error> const err;

public:
    ZENO_API explicit ZenoException(std::unique_ptr<Error> &&err) noexcept;
    ZENO_API ~ZenoException() noexcept;
    ZENO_API char const *what() const noexcept;

    ZenoException(ZenoException const &) = delete;
    ZenoException &operator=(ZenoException const &) = delete;
    ZenoException(ZenoException &&) = default;
    ZenoException &operator=(ZenoException &&) = default;
};

template <class T, class ...Ts>
static ZenoException makeError(Ts &&...ts) {
    return ZenoException(std::make_unique<T>(std::forward<Ts>(ts)...));
}

}
