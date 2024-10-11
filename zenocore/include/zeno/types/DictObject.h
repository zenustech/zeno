#pragma once

#include <zeno/core/IObject.h>
#include <zeno/funcs/LiterialConverter.h>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct DictObject : IObjectClone<DictObject> {
  std::map<std::string, zany> lut;

  template <class T = IObject>
  std::map<std::string, std::shared_ptr<T>> get() const {
      std::map<std::string, std::shared_ptr<T>> res;
      for (auto const &[key, val]: lut) {
          res.emplace(key, safe_dynamic_cast<T>(val));
      }
      return res;
  }

  template <class T>
  std::map<std::string, T> getLiterial() const {
      std::map<std::string, T> res;
      for (auto const &[key, val]: lut) {
          res.emplace(key, objectToLiterial<T>(val));
      }
      return res;
  }

  bool update_key(const std::string& key) override {
      m_key = key;
      for (auto& [key, spObject] : lut) {
          if (spObject->key().empty())
          {
              std::string itemKey = m_key + "/" + key;
              spObject->update_key(itemKey);
          } else {
              size_t pos = spObject->key().find_last_of("/");
              std::string itemKey = m_key + "/" + (pos == std::string::npos ? spObject->key() : spObject->key().substr(pos));
              spObject->update_key(itemKey);
          }
      }
      return true;
  }
};

}
