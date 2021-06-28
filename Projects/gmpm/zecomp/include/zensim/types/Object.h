#pragma once

/* <editor-fold desc="MIT License">

Copyright(c) 2018 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include "Property.h"
#include "zensim/Reflection.h"
#include "zensim/execution/Concurrency.h"
#include "zensim/meta/Sequence.h"
#include "zensim/tpls/fmt/format.h"

namespace zs {

  template <typename T> struct wrapt;
  struct Object;
  ZS_type_name(Object);
  using ObjOwner = std::unique_ptr<Object>;
  using ObjObserver = RefPtr<Object>;
  using ConstObjObserver = ConstRefPtr<Object>;

  struct Object {
  private:
    /// used as trackpads, not owners
    inline static concurrent_map<std::string, ObjObserver> _objectMap;
    inline static concurrent_map<ConstRefPtr<Object>, std::string> _objectNameMap;

    inline static concurrent_map<std::string, ObjOwner> _globalObjects;

  public:
    // static ref_ptr<Object> create(Allocator* allocator=nullptr);
    virtual ~Object() = default;
    Object() noexcept = default;
    Object(Object &&) noexcept = default;
    Object &operator=(Object &&) noexcept = default;
    Object(const Object &) = default;
    Object &operator=(const Object &) = default;

    /// object interface
    constexpr std::size_t sizeofObject() const noexcept { return sizeof(Object); }
    constexpr const char *className() const noexcept { return type_name<Object>(); }
    /// return the std::type_info of this Object
    constexpr const std::type_info &typeInfo() const noexcept { return typeid(Object); }
    bool isCompatible(const std::type_info &type) const noexcept { return typeid(Object) == type; }

    template <class T> constexpr T *cast() {
      return isCompatible(typeid(T)) ? static_cast<T *>(this) : nullptr;
    }

    template <class T> constexpr const T *cast() const {
      return isCompatible(typeid(T)) ? static_cast<const T *>(this) : nullptr;
    }

    static auto &globalObjects() noexcept { return _globalObjects; }
    void trackBy(const std::string &key) {
      /// should throw when found an existing name
      _objectMap.set(key, this);
      _objectNameMap.set(this, key);
    }
    void unregister() {
      /// should throw when found an existing name
      if (auto namePtr = tryGet(this); namePtr) {
        _objectMap.erase(*namePtr);
        _objectNameMap.erase(this);
      }
    }
    std::string objectName() const { return _objectNameMap.get(this); }
    static ObjObserver track(const std::string &key) { return _objectMap.get(key); }
    static std::string track(const ObjObserver obj) { return _objectNameMap.get(obj); }
    static ObjObserver *tryGet(const std::string &key) { return _objectMap.find(key); }
    static std::string *tryGet(ConstObjObserver obj) { return _objectNameMap.find(obj); }
    template <typename T> static bool object_exist(ConstRefPtr<T> obj) {
      if (auto ptr = dynamic_cast<ConstObjObserver>(obj); ptr) return tryGet(ptr) != nullptr;
      return false;
    }

    /// assign an Object associated with key
    void setObject(const std::string &key, Object *object);

    /// get Object associated with key, return nullptr if no object associated
    /// with key has been assigned
    Object *getObject(const std::string &key);

    /// get const Object associated with key, return nullptr if no object
    /// associated with key has been assigned
    const Object *getObject(const std::string &key) const;

    /// get object of specified type associated with key, return nullptr if no
    /// object associated with key has been assigned
    template <class T> T *getObject(const std::string &key) {
      return dynamic_cast<T *>(getObject(key));
    }

    /// get const object of specified type associated with key, return nullptr if
    /// no object associated with key has been assigned
    template <class T> const T *getObject(const std::string &key) const {
      return dynamic_cast<const T *>(getObject(key));
    }

    /// remove meta object or value associated with key
    void removeObject(const std::string &key);
#if 0
  // ref counting methods
  inline void ref() const noexcept {
    _referenceCount.fetch_add(1, std::memory_order_relaxed);
  }
  inline void unref() const noexcept {
    if (_referenceCount.fetch_sub(1, std::memory_order_seq_cst) <= 1)
      _attemptDelete();
  }
  inline void unref_nodelete() const noexcept {
    _referenceCount.fetch_sub(1, std::memory_order_seq_cst);
  }
  inline unsigned int referenceCount() const noexcept {
    return _referenceCount.load();
  }

  /// meta data access methods
  /// wraps the value with a vsg::Value<T> object and then assigns via
  /// setObject(key, vsg::Value<T>)
  template <typename T> void setValue(const std::string &key, const T &value);

  /// specialization of setValue to handle passing c strings
  void setValue(const std::string &key, const char *value) {
    setValue(key, value ? std::string(value) : std::string());
  }

  /// get specified value type, return false if value associated with key is not
  /// assigned or is not the correct type
  template <typename T> bool getValue(const std::string &key, T &value) const;

#endif

    // Auxiliary object access methods, the optional Auxiliary is used to store
    // meta data and links to Allocator
    // Auxiliary *getOrCreateUniqueAuxiliary();
    // Auxiliary *getAuxiliary() { return _auxiliary; }
    // const Auxiliary *getAuxiliary() const { return _auxiliary; }

    // convenience method for getting the optional Allocator, if present this
    // Allocator would have been used to create this Objects memory
    // Allocator *getAllocator() const;

  protected:
    // virtual void _attemptDelete() const;
    // void setAuxiliary(Auxiliary *auxiliary);

  private:
    // friend class Auxiliary;

    // mutable std::atomic_uint _referenceCount;

    // Auxiliary *_auxiliary;
  };

  template <typename T> using is_object = std::is_base_of<Object, T>;

  /// generate named objects that can be indexed by strings
  /// non-intrusive way
  struct Instancer;
  ZS_type_name(Instancer);
  struct Instancer {
  private:
    struct Concept {
      virtual ~Concept() = default;
      virtual ObjOwner new_object(const std::string &) const = 0;
    };
    template <typename T> struct Builder : Concept {
      static inline std::once_flag registerFlag{};
      //
      ObjOwner new_object(const std::string &objName) const override {
        auto objPtr = std::make_unique<T>();
        objPtr->trackBy(objName);
        return objPtr;  ///< the caller is the owner
      }
    };
    using ClassPtr = std::unique_ptr<Concept>;
    inline static concurrent_map<std::string, ClassPtr> _classMap;

  public:
    template <typename T> Instancer(std::string name, wrapt<T>) {
      using InstanceBuilder = Builder<T>;
      std::call_once(InstanceBuilder::registerFlag, [className = std::move(name)]() {
        // only register once (when empty)
        if (_classMap.find(className) == nullptr) {
          // ZS_TRACE("registering class [{}]\n", className);
          _classMap.emplace(std::move(className), std::move(std::make_unique<InstanceBuilder>()));
        } else
          throw std::runtime_error(fmt::format(
              "Another class has already been registered with the same name [{}]", className));
      });
    }
    static ObjOwner create(std::string className, std::string objName) {
      if (const auto creator = _classMap.find(className); creator) {
        // ZS_TRACE("creating class instance [{}]: [{}]\n", className, objName);
        return (*creator)->new_object(objName);
      }
      throw std::runtime_error(
          fmt::format("Classname[{}] used for creating a instance does not exist!", className));
    }
    template <typename... Ts> static ObjOwner create(std::string className, std::string objName,
                                                     std::tuple<Ts &&...> &&args) {
      if (const auto creator = _classMap.find(className); creator != nullptr) {
        // ZS_TRACE("creating class instance [{}]: [{}]\n", className, objName);
        auto handle = (*creator)->new_object(objName);
        *handle = std::move(std::make_from_tuple(args));
        return handle;
      }
      throw std::runtime_error(
          fmt::format("Classname[{}] used for creating a instance does not exist!", className));
    }
  };

#define REGISTER_CUSTOM_CLASS(CLASS_NAME) ::zs::Instancer(#CLASS_NAME, ::zs::wrapt<CLASS_NAME>{});
#define REGISTER_CLASS(CLASS_NAME) ::zs::Instancer(#CLASS_NAME, ::zs::wrapt<::zs::CLASS_NAME>{});

#define LET(CLASS_NAME, OBJECT_NAME, ...) \
  auto OBJECT_NAME                        \
      = ::zs::Instancer::create(#CLASS_NAME, #OBJECT_NAME, std::forward_as_tuple(__VA_ARGS__));
#define VAR(CLASS_NAME, OBJECT_NAME) \
  auto OBJECT_NAME = ::zs::Instancer::create(#CLASS_NAME, #OBJECT_NAME);
#define MEMBER_VAR(OBJECT_NAME, CLASS_NAME, MEMBER_NAME, ...)                                \
  OBJECT_NAME->MEMBER_NAME                                                                   \
      = ::zs::Instancer::create(#CLASS_NAME, std::string{#OBJECT_NAME} + "." + #MEMBER_NAME, \
                                std::forward_as_tuple(__VA_ARGS__));

  struct GraphNode;
  ZS_type_name(GraphNode);

}  // namespace zs
