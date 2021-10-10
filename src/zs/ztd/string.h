#pragma once

#include <cstdio>
#include <string>
#include <sstream>
#include <zs/zeno/ztd/error.h>

namespace zeno2::ztd {

static inline auto split_str(std::string const &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

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

static bool starts_with(std::string line, std::string pattern) {
	return line.find(pattern) == 0;
}

static std::string trim_string(std::string str) {
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
