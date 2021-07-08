#pragma once

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <zeno/vec.h>
#include <zeno/zeno.h>

namespace zeno {

using AttributeArray =
    std::variant<std::vector<zeno::vec3f>, std::vector<float>>;

struct PrimitiveObject : zeno::IObjectClone<PrimitiveObject> {

  std::map<std::string, AttributeArray> m_attrs;
  size_t m_size{0};

  std::vector<int> points;
  std::vector<zeno::vec2i> lines;
  std::vector<zeno::vec3i> tris;
  std::vector<zeno::vec4i> quads;

#ifndef ZEN_NOREFDLL
  ZENAPI virtual void dumpfile(std::string const &path) override;
#else
  virtual void dumpfile(std::string const &path) override {}
#endif

  template <class T> std::vector<T> &add_attr(std::string const &name) {
    if (!has_attr(name))
      m_attrs[name] = std::vector<T>(m_size);
    return attr<T>(name);
  }
  template <class T> std::vector<T> &add_attr(std::string const &name, T value) {
    if (!has_attr(name))
      m_attrs[name] = std::vector<T>(m_size, value);
    return attr<T>(name);
  }
  template <class T> std::vector<T> &attr(std::string const &name) {
    return std::get<std::vector<T>>(m_attrs.at(name));
  }

  AttributeArray &attr(std::string const &name) { return m_attrs.at(name); }

  template <class T> std::vector<T> const &attr(std::string const &name) const {
    return std::get<std::vector<T>>(m_attrs.at(name));
  }

  AttributeArray const &attr(std::string const &name) const {
    return m_attrs.at(name);
  }

  bool has_attr(std::string const &name) const {
    return m_attrs.find(name) != m_attrs.end();
  }

  template <class T> bool attr_is(std::string const &name) const {
    return std::holds_alternative<std::vector<T>>(m_attrs.at(name));
  }

  size_t size() const { return m_size; }

  void resize(size_t size) {
    m_size = size;
    for (auto &[key, val] : m_attrs) {
      std::visit([&](auto &val) { val.resize(m_size); }, val);
    }
  }
};

} // namespace zeno
