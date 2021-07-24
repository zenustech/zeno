#pragma once


#include <map>
#include <string>
#include <memory>
#include <zeno/zeno.h>


namespace zeno {


template <class T>
T *safe_at(std::map<std::string, std::unique_ptr<T>> const &m,
           std::string const &key, std::string const &msg,
           std::string const &extra = {}) {
  auto it = m.find(key);
  if (it == m.end()) {
    auto extra_ = extra;
    if (extra.size()) extra_ = " for `" + extra + "`";
    throw Exception("invalid " + msg + " name `" + key + "`" + extra_);
  }
  return it->second.get();
}

template <class T>
T const &safe_at(std::map<std::string, T> const &m, std::string const &key,
          std::string const &msg, std::string const &extra = {}) {
  auto it = m.find(key);
  if (it == m.end()) {
    auto extra_ = extra;
    if (extra.size()) extra_ = " for `" + extra + "`";
    throw Exception("invalid " + msg + " name `" + key + "`" + extra_);
  }
  return it->second;
}

template <class T, class S>
T const &safe_at(std::map<S, T> const &m, S const &key, std::string const &msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw Exception("invalid " + msg + " as index");
  }
  return it->second;
}

static inline auto split_str(std::string const &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}


}
