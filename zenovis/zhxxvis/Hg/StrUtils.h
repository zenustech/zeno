#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <cctype>


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

static bool starts_with(std::string line, std::string pattern) {
	return line.find(pattern) == 0;
}

static std::string trim(std::string str) {
	while (str.size() != 0 && std::isspace(str[0])) {
		str.erase(0, 1);
	}
	while (str.size() != 0) {
		auto len = str.size();
		if (std::isspace(str[len - 1])) {
			str.pop_back();
		} else {
			break;
		}
	}
	return str;
}
}
