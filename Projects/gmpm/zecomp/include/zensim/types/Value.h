#pragma once
#include <any>
#include <type_traits>

#include "Object.h"
#include "Polymorphism.h"
#include "zensim/math/Vec.h"
#include "zensim/meta/Meta.h"

namespace zs {

  enum class value_category_e : char { unknown = 0, arithmetic, vec, string, custom };
  template <typename T> value_category_e value_category() {
    if constexpr (std::is_arithmetic_v<T>) {
      return value_category_e::arithmetic;
    } else if constexpr (is_vec<T>::value) {
      return value_category_e::vec;
    } else if constexpr (is_same_v<T, std::string>) {
      return value_category_e::string;
    } else if constexpr (is_object<T>::value) {
      return value_category_e::string;
    } else {
      return value_category_e::unknown;
    }
  }

  template <typename T, typename = void> struct Value;
  /// this special case corresponds to any registered class
  template <> struct Value<Object> {
  protected:
    struct ObjectCopyInterface {
      virtual ~ObjectCopyInterface() = default;
      virtual ObjOwner copy(ObjObserver obj) const = 0;
      virtual Holder<ObjectCopyInterface> copyCloner() const = 0;
    };
    template <typename T, typename = void> struct ObjectValueCloner;
    template <typename T> struct ObjectValueCloner<
        T, void_t<enable_if_all<is_object<T>::value, std::is_copy_constructible_v<T>>>>
        : ObjectCopyInterface {
      ObjOwner copy(ObjObserver obj) const override {
        if (obj == nullptr) return {};
        return std::make_unique<T>(*static_cast<T *>(obj));
      }
      Holder<ObjectCopyInterface> copyCloner() const override {
        return std::make_unique<ObjectValueCloner<T>>();
      }
    };

  public:
    using value_t = Object;
    using reference_t = ObjOwner &;
    using result_t = ObjObserver;

    template <typename V, enable_if_all<std::is_copy_constructible_v<std::decay_t<V>>,
                                        is_object<std::decay_t<V>>::value> = 0>
    constexpr Value(V &&v)
        : _val{std::make_unique<std::decay_t<V>>(FWD(v))},
          _cloner{std::make_unique<ObjectValueCloner<std::decay_t<V>>>()} {}

    ObjOwner clone() const { return _cloner->copy(_val.get()); }

    Value(const Value &o) : _val{o.clone()}, _cloner{o._cloner->copyCloner()} {}
    Value &operator=(const Value &o) {
      Value tmp{o};
      *this = std::move(tmp);
      return *this;
    }
    Value(Value &&o) noexcept = default;
    Value &operator=(Value &&o) noexcept = default;

    constexpr reference_t get() noexcept { return _val; }
    result_t get() const noexcept { return _val.get(); }

    template <typename T> constexpr T &get() {
      /// have not tested this type check yet
      auto &obj = *_cloner;
      if (typeid(ObjectValueCloner<T>) != typeid(obj))
        throw std::runtime_error("value type being extracted from this ObjectValue mismatch");
      return *static_cast<T *>(_val.get());
    }
    template <typename T> constexpr const T &get() const {
      /// have not tested this type check yet
      const auto &obj = *_cloner;
      if (typeid(ObjectValueCloner<T>) != typeid(obj))
        throw std::runtime_error("value type being extracted from this ObjectValue mismatch");
      return *static_cast<const T *>(_val.get());
    }

    template <typename V, enable_if_t<is_object<std::decay_t<V>>::value> = 0>
    constexpr void set(V &&v) {
      using Val = std::decay_t<V>;
      // if (_val.get() == nullptr)
      //  throw std::runtime_error("");
      Val *self = dynamic_cast<Val *>(_val.get());
      if (self == nullptr)
        throw std::runtime_error("value type of the holder does not match the assigned value");
      if constexpr (std::is_move_assignable_v<Val>) {
        *self = std::move(v);
        _cloner.reset(new ObjectValueCloner<Val>());
      } else if constexpr (std::is_copy_assignable_v<Val>) {
        *self = v;
        _cloner.reset(new ObjectValueCloner<Val>());
      } else if constexpr (std::is_swappable_v<Val>) {
        std::swap(*self, FWD(v));
        _cloner.reset(new ObjectValueCloner<Val>());
      } else
        throw std::runtime_error("value type mismatch!");
    }
    template <typename V, enable_if_t<is_object<std::decay_t<V>>::value> = 0>
    constexpr Value &operator=(V &&v) {
      set(FWD(v));
      return *this;
    }

  protected:
    ObjOwner _val;
    Holder<ObjectCopyInterface> _cloner;
  };

  /// regular values
  template <typename T> struct Value<T, void_t<enable_if_t<!is_object<T>::value>>> {
    static_assert(std::is_arithmetic_v<T> || is_vec<T>::value || is_same_v<std::string, T>,
                  "value type should be arithmetic, vec or string!");
    using value_t = T;
    using reference_t = std::decay_t<T> &;
    using result_t = const std::decay_t<T> &;
    // conditional_t<std::is_fundamental_v<value_t>, value_t, const value_t &>;

    template <typename V = T, enable_if_t<std::is_arithmetic_v<V>> = 0> constexpr Value(V v = 0)
        : _val{static_cast<value_t>(v)} {}
    template <typename V = T, enable_if_t<is_vec<V>::value> = 0>
    constexpr Value(V v = V::uniform(0)) : _val{static_cast<value_t>(std::move(v))} {}
    template <typename V = T, enable_if_t<is_same_v<std::string, V>> = 0> constexpr Value(V v = "")
        : _val{static_cast<value_t>(std::move(v))} {}

    Value(const Value &o) = default;
    Value &operator=(const Value &o) = default;
    Value(Value &&o) noexcept = default;
    Value &operator=(Value &&o) noexcept = default;

    constexpr operator reference_t() { return std::any_cast<reference_t>(_val); }
    constexpr operator result_t() const { return std::any_cast<result_t>(_val); }
    constexpr reference_t get() { return std::any_cast<reference_t>(_val); }
    constexpr result_t get() const { return std::any_cast<result_t>(_val); }

    constexpr void set(T const &v) noexcept { _val = v; }
    constexpr Value &operator=(T const &v) noexcept {
      _val = v;
      return *this;
    }

  protected:
    // T _val;
    std::any _val;
  };

  template <typename T, enable_if_t<!is_object<std::decay_t<T>>::value> = 0> Value(T &&)
      -> Value<std::decay_t<T>>;
  template <typename T, enable_if_t<is_object<std::decay_t<T>>::value> = 0> Value(T &&)
      -> Value<Object>;

  template <typename T> struct is_value : std::false_type {};
  template <typename T> struct is_value<Value<T>> : std::true_type {};

  using Bool = Value<bool>;
  using Char = Value<char>;
  using Int = Value<i32>;
  using Float = Value<f32>;
  using Double = Value<f64>;

  using vec2i = vec<i32, 2>;
  using vec2f = vec<f32, 2>;
  using vec2l = vec<i64, 2>;
  using vec2d = vec<f64, 2>;

  using vec3i = vec<i32, 3>;
  using vec3f = vec<f32, 3>;
  using vec3l = vec<i64, 3>;
  using vec3d = vec<f64, 3>;

  using vec4i = vec<i32, 4>;
  using vec4f = vec<f32, 4>;
  using vec4l = vec<i64, 4>;
  using vec4d = vec<f64, 4>;

  using mat2f = vec<f32, 2, 2>;
  using mat2d = vec<f64, 2, 2>;
  using mat3f = vec<f32, 3, 3>;
  using mat3d = vec<f64, 3, 3>;
  using mat4f = vec<f32, 4, 4>;
  using mat4d = vec<f64, 4, 4>;

  using PresetValue
      = variant<bool, i32, i64, f32, f64, vec2d, vec3d, vec4d, mat2d, mat3d, mat4d, std::string>;

  using Attribute = variant<Value<bool>, Value<i32>, Value<i64>, Value<f32>, Value<f64>,
                            Value<vec2d>, Value<vec3d>, Value<vec4d>, Value<mat2d>, Value<mat3d>,
                            Value<mat4d>, Value<std::string>>;

#if 0
decltype(auto) getValue(Attribute& val) { 
  return match([](auto& val) {
    return val.get();
  })(val);
}
#endif

  enum class prop_access_e { rw = 0, ro, wo };
  enum class prop_value_e {
    i64 = 0,
    f64,
    vec2d,
    vec3d,
    vec4d,
    mat2d,
    mat3d,
    mat4d,
    string,
    custom
  };

  template <typename T> using ValueHolder = Holder<Value<T>>;  ///< non-owning reference

  using PropertyValue = variant<Value<bool>, Value<i32>, Value<i64>, Value<f32>, Value<f64>,
                                Value<vec2d>, Value<vec3d>, Value<vec4d>, Value<mat2d>,
                                Value<mat3d>, Value<mat4d>, Value<std::string>, Value<Object>>;
  static constexpr const char *g_value_names[]
      = {"bool",  "i32",   "i64",   "f32",   "f64",    "vec2d", "vec3d",
         "vec4d", "mat2d", "mat3d", "mat4d", "string", "custom"};

  template <typename T> inline void setValue(PropertyValue &val, T &&v) {
    match([val = FWD(v)](auto &value) { value = val; })(val);
  }
  template <typename T> inline void setValue(Value<T> &val, T &&v) { val.set(FWD(v)); }
  template <typename T> inline decltype(auto) getValue(Value<T> &val) {
    if constexpr (is_object<T>::value)
      return val.template get<T>();
    else
      return val.get();
  }
  template <typename T> inline decltype(auto) getValue(const Value<T> &val) {
    if constexpr (is_object<T>::value)
      return val.template get<T>();
    else
      return val.get();
  }
  template <typename T> inline decltype(auto) getValue(PropertyValue &val) {
    if constexpr (is_object<T>::value)
      return std::get<Value<Object>>(val).template get<T>();
    else
      return std::get<Value<T>>(val).get();
    // return match([](auto &valPtr) { return valPtr->get(); })(val);
  }
  template <typename T> inline decltype(auto) getValue(const PropertyValue &val) {
    if constexpr (is_object<T>::value)
      return std::get<Value<Object>>(val).template get<T>();
    else
      return std::get<Value<T>>(val).get();
  }
  struct Property {
    std::size_t index() const noexcept { return _value.index(); }
    std::string name() const noexcept {
      auto id = index();
      if (id != std::variant_npos) return g_value_names[id];
      return "empty";
    }
    template <typename T>
    // typename V = decltype(Value{std::declval<std::decay_t<T>>()})
    Property(T &&v, prop_access_e tag = prop_access_e::rw)
        : _value{Value{FWD(v)}}, _accessTag{tag} {}
    Property(const Property &) = default;
    Property &operator=(const Property &) = default;
    Property(Property &&) noexcept = default;
    Property &operator=(Property &&) noexcept = default;

    constexpr PropertyValue &value() noexcept { return _value; }
    constexpr const PropertyValue &value() const noexcept { return _value; }

  protected:
    PropertyValue _value{};
    prop_access_e _accessTag{prop_access_e::rw};
  };

  inline std::pair<std::string, Property &&> make_tagged_property(std::string name,
                                                                  Property &&prop) {
    return std::make_pair<std::string, Property &&>(std::move(name), std::move(prop));
  }

  /// this is an aggregator of attribute data (aka options)
  struct Properties {
    ConstRefPtr<Property> find(const std::string &key) const {
      if (auto it = _properties.find(key); it != _properties.end()) return &(it->second);
      return nullptr;
    }
    RefPtr<Property> find(const std::string &key) {
      if (auto it = _properties.find(key); it != _properties.end()) return &(it->second);
      return nullptr;
    }
    Properties &add(std::pair<std::string, Property> &&entry) {
      if (find(entry.first) == nullptr)
        _properties.emplace(std::move(entry.first), std::move(entry.second));
      return *this;
    }
    Properties &add(const std::string &name, const Property &prop) {
      if (find(name) == nullptr) _properties.emplace(name, prop);
      return *this;
    }
    template <typename T, enable_if_t<!is_value<T>::value> = 0>
    Properties &add(const std::string &name, T &&v, prop_access_e tag = prop_access_e::rw) {
      if (find(name) == nullptr) _properties.emplace(name, Property{FWD(v), tag});
      return *this;
    }
#if 0
  template <typename T, enable_if_t<is_value<T>::value> = 0>
  Properties &add(const std::string &name, T &&v,
                  prop_access_e tag = prop_access_e::rw) {
    if (find(name) == nullptr)
      _properties.emplace(name, Property{v.get(), tag});
    return *this;
  }
#endif
    template <typename T, enable_if_t<!is_value<T>::value> = 0>
    Properties &set(std::string name, T &&v, prop_access_e tag = prop_access_e::rw) {
      _properties.insert_or_assign(std::move(name), Property{v, tag});
      return *this;
    }
    template <typename T, enable_if_t<is_value<T>::value> = 0>
    Properties &set(std::string name, T &&v, prop_access_e tag = prop_access_e::rw) {
      _properties.insert_or_assign(std::move(name), Property{v.get(), tag});
      return *this;
    }
    auto &refProperties() noexcept { return _properties; }
    const auto &refProperties() const noexcept { return _properties; }

  protected:
    std::map<std::string, Property> _properties;
  };

}  // namespace zs