#include <cstdio>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <memory>
#include <tuple>
#include <set>

namespace bate::common {

using std::cout;
using std::endl;

template <int First, int Last, typename Lambda>
inline constexpr bool static_for(Lambda const &f) {
    if constexpr (First < Last) {
        if (f(std::integral_constant<int, First>{})) {
            return true;
        } else {
            return static_for<First + 1, Last>(f);
        }
    }
    return false;
}

template <class T>
struct copiable_unique_ptr : std::unique_ptr<T> {
    using std::unique_ptr<T>::unique_ptr;
    using std::unique_ptr<T>::operator=;

    copiable_unique_ptr &operator=(copiable_unique_ptr const &o) {
        std::unique_ptr<T>::operator=(std::unique_ptr<T>(
            std::make_unique<T>(static_cast<T const &>(*o))));
        return *this;
    }

    copiable_unique_ptr(std::unique_ptr<T> &&o)
        : std::unique_ptr<T>(std::move(o)) {
    }

    copiable_unique_ptr(copiable_unique_ptr const &o)
        : std::unique_ptr<T>(std::make_unique<T>(
            static_cast<T const &>(*o))) {
    }

    operator std::unique_ptr<T> &() { return *this; }
    operator std::unique_ptr<T> const &() const { return *this; }
};

template <class T>
copiable_unique_ptr(std::unique_ptr<T> &&o) -> copiable_unique_ptr<T>;

template <class T>
bool contains(std::set<T> const &list, T const &value) {
    return list.find(value) != list.end();
}

template <size_t BufSize = 4096, class ...Ts>
std::string format(const char *fmt, Ts &&...ts) {
    char buf[BufSize];
    sprintf(buf, fmt, std::forward<Ts>(ts)...);
    return buf;
}

template <size_t BufSize = 4096, class T = void>
std::string format_join(const char *sep,
    const char *fmt, std::vector<T> const &ts) {
    std::string res;
    bool any = false;
    for (auto t: ts) {
        if (any) res += sep; else any = true;
        res += format<BufSize, T>(fmt, std::move(t));
    }
    return res;
}

template <class ...Ts>
[[noreturn]] void error(const char *fmt, Ts &&...ts) {
    printf("ERROR: ");
    printf(fmt, std::forward<Ts>(ts)...);
    putchar('\n');
    exit(-1);
}

template <class T>
T from_string(std::string const &s) {
    std::stringstream ss(s);
    T t;
    ss >> t;
    return t;
}

template <class T, class S>
static std::string join_str(std::vector<T> const &elms, S const &delim) {
    std::stringstream ss;
    auto p = elms.begin(), end = elms.end();
    if (p != end)
        ss << *p++;
    for (; p != end; ++p) {
        ss << delim << *p;
    }
    return ss.str();
}

static std::vector<std::string> split_str(std::string const &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

template <class T>
static inline std::string to_string(T const &value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

class strbuild {
  std::stringstream ss;

public:
  strbuild() = default;
  strbuild(std::string const &str) : ss(str) {}

  operator std::string() const {
    return ss.str();
  }

  template <class T>
  strbuild &operator<<(T const &value) {
    ss << value;
    return *this;
  }

  template <class T>
  strbuild &operator>>(T &value) {
    ss >> value;
    return *this;
  }
};

}
