#pragma once


#include <map>
#include <string>
#include <memory>
#include <zeno/utils/Error.h>
#include <zeno/utils/to_string.h>


namespace zeno {


template <class T>
T const &safe_at(std::map<std::string, T> const &m, std::string const &key, std::string_view msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw makeError<KeyError>(key, msg);
  }
  return it->second;
}

template <class T>
T &safe_at(std::map<std::string, T> &m, std::string const &key, std::string_view msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw makeError<KeyError>(key, msg);
  }
  return const_cast<T &>(it->second);
}


template <class T, class S>
T const &safe_at(std::map<S, T> const &m, S const &key, std::string_view msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw makeError<KeyError>(to_string(key), msg);
  }
  return it->second;
}


template <class T, class S>
T &safe_at(std::map<S, T> &m, S const &key, std::string_view msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw makeError<KeyError>(to_string(key), msg);
  }
  return const_cast<T &>(it->second);
}


}
