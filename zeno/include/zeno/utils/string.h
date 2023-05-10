#pragma once


#include <map>
#include <string>
#include <memory>
#include <sstream>


namespace zeno {


static std::vector<std::string> split_str(std::string const &s, char delimiter = ' ') {
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

static bool starts_with(std::string const &line, std::string const &pattern) {
	return line.find(pattern) == 0;
}

static bool ends_with(std::string const &line, std::string const &pattern, bool isCaseSensitive = true) {
    if (line.size() < pattern.size()) {
        return false;
    }
    for (auto i = 0; i < pattern.size(); i++) {
        if (isCaseSensitive) {
            auto a = pattern[i];
            auto b = line[line.size() - pattern.size() + i];
            if (a != b) {
                return false;
            }
        }
        else {
            auto a = std::tolower(pattern[i]);
            auto b = std::tolower(line[line.size() - pattern.size() + i]);
            if (a != b) {
                return false;
            }
        }
    }
    return true;
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

static std::string replace_all(const std::string& inout_, std::string_view what, std::string_view with) {
    std::string inout = inout_;
    for (std::string::size_type pos{};
         std::string::npos != (pos = inout.find(what.data(), pos, what.length()));
         pos += with.length()
     ) {
        inout.replace(pos, what.length(), with.data(), with.length());
    }
    return inout;
}

static std::string remove_all(const std::string& inout, std::string_view what) {
    return replace_all(inout, what, "");
}

}
