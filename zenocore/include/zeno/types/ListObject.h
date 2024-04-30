#pragma once

#include <zeno/core/IObject.h>
#include <zeno/funcs/LiterialConverter.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {

  ListObject() = default;

  explicit ListObject(std::vector<zany> arrin) : m_objects(std::move(arrin)) {
  }

  template <class T = IObject>
  std::vector<std::shared_ptr<T>> get() const {
      std::vector<std::shared_ptr<T>> res;
      for (auto const &val: m_objects) {
          res.push_back(safe_dynamic_cast<T>(val));
      }
      return res;
  }

  template <class T = IObject>
  std::vector<T *> getRaw() const {
      std::vector<T *> res;
      for (auto const &val: m_objects) {
          res.push_back(safe_dynamic_cast<T>(val.get()));
      }
      return res;
  }

  void resize(const size_t sz) {
      m_objects.resize(sz);
  }

  void append(zany spObj) {
      m_objects.push_back(spObj);
      spObj->set_parent(this);
      //m_ptr2Index.insert(std::make_pair((uint16_t)spObj.get(), m_objects.size()));
  }

  void append(zany&& spObj) {
      m_objects.push_back(spObj);
      spObj->set_parent(this);
      //m_ptr2Index.insert(std::make_pair((uint16_t)spObj.get(), m_objects.size()));
  }

  zany get(int index) const {
      if (0 > index || index >= m_objects.size())
          return nullptr;
      return m_objects[index];
  }

  void set(const std::vector<zany>& arr) {
      m_objects = arr;
  }

  void set(size_t index, zany&& obj) {
      if (0 > index || index >= m_objects.size())
          return;
      m_objects[index] = obj;
  }

  void mark_dirty(int index) {
      dirtyIndice.insert(index);
  }

  bool has_dirty(int index) const {
      return dirtyIndice.count(index);
  }

  size_t size() const {
      return m_objects.size();
  }

  void emplace_back(zany&& obj) {
      append(obj);
  }

  void push_back(zany&& obj) {
      append(obj);
  }

  void push_back(const zany& obj) {
      append(obj);
  }

  bool update_key(const std::string& key) override {
      m_key = key;
      for (int i = 0; i < m_objects.size(); i++) {
          if (m_objects[i]->key().empty())
          {
              std::string itemKey = m_key + "/" + std::to_string(i);
              m_objects[i]->update_key(itemKey);
          }
      }
      return true;
  }

  void clear() {
      m_objects.clear();
  }

  template <class T>
  std::vector<T> get2() const {
      std::vector<T> res;
      for (auto const &val: m_objects) {
          res.push_back(objectToLiterial<T>(val));
      }
      return res;
  }

  template <class T>
  [[deprecated("use get2<T>() instead")]]
  std::vector<T> getLiterial() const {
      return get2<T>();
  }

private:
    std::vector<zany> m_objects;
    std::set<int> dirtyIndice;                        //该list下dirty的obj的index
    //std::map<std::string, int> nodeNameArrItemMap;    //obj所在的节点名到obj在m_objects中索引的map
    //std::map<uint16_t, int> m_ptr2Index;
};

}
