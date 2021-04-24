#pragma once


#include <string>
#include <sstream>
#include <vector>


namespace hg {

template <class S, class T>
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

template <class T, class S>
static inline T assign_conv(S const &data)
{
  T ret;
  ret.assign(data.begin(), data.end());
  return ret;
}

template <class T, class S>
static inline T assign_conv(S const &begin, S const &end)
{
  T ret;
  ret.assign(begin, end);
  return ret;
}

}
