#pragma once

#include <string>
#include <sstream>
#include <vector>


namespace hg {

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
static inline std::string to_string(T const &value)
{
  std::stringstream ss;
  ss << value;
  return ss.str();
}

template <class T>
static inline T from_string(std::string const &str)
{
  std::stringstream ss(str);
  T value;
  ss >> value;
  return value;
}

class StringBuilder {
  std::stringstream ss;

public:
  StringBuilder() = default;
  StringBuilder(std::string const &str) : ss(str) {}

  operator std::string() {
    return ss.str();
  }

  template <class T>
  StringBuilder &operator<<(T const &value) {
    ss << value;
    return *this;
  }

  template <class T>
  StringBuilder &operator>>(T &value) {
    ss >> value;
    return *this;
  }
};

}
