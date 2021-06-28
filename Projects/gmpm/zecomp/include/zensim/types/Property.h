#pragma once
#include <iterator>
#include <type_traits>

namespace zs {

  /// mutable
  /// (preferably no) dependent preconditions
  /// invariants

  /// behavior: default constructible, movable
  /// movable: default constructible, swappable/movable, comparable, assignable
  /// immutable: default constructible, copyable, comparable
  /// semi-regular: default constructible, copyable/movable, swappable,
  /// assignable
  /// regular: + comparable

  /// assignable

  /// copyable, movable (all)
  struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(const NonCopyable &) = delete;
    NonCopyable &operator=(const NonCopyable &) = delete;
  };
  struct MovableNonCopyable {
    MovableNonCopyable() = default;
    MovableNonCopyable(const MovableNonCopyable &other) = delete;
    MovableNonCopyable &operator=(const MovableNonCopyable &) = delete;

    MovableNonCopyable(MovableNonCopyable &&other) noexcept = default;
    MovableNonCopyable &operator=(MovableNonCopyable &&) noexcept = default;
  };
  /// swappable

  /// comparable
  template <typename T> struct is_equality_comparable {
  private:
    static void *conv(bool);
    template <typename U> static std::true_type test(
        decltype(conv(std::declval<U const &>() == std::declval<U const &>())),
        decltype(conv(!std::declval<U const &>() == std::declval<U const &>())));
    template <typename U> static std::false_type test(...);

  public:
    static constexpr bool value = decltype(test<T>(nullptr, nullptr))::value;
  };

  template <typename Base, typename... Ts> struct is_base_of;
  template <typename Base, typename T> struct is_base_of<Base, T> {
    static_assert(std::is_base_of<Base, T>::value, "T is not a subclass of Base!");
  };
  template <typename Base, typename T, typename... Ts> struct is_base_of<Base, T, Ts...>
      : public is_base_of<Base, Ts...> {
    static_assert(std::is_base_of<Base, T>::value, "T is not a subclass of Base!");
  };

  template <template <class T> class Feature, typename... Ts> struct satisfy;
  template <template <class T> class Feature, typename T> struct satisfy<Feature, T> {
    static_assert(Feature<T>::value, "T does not satisfy the feature!");
  };
  template <template <class T> class Feature, typename T, typename... Ts>
  struct satisfy<Feature, T, Ts...> : public satisfy<Feature, Ts...> {
    static_assert(Feature<T>::value, "T does not satisfy the feature!");
  };

  template <typename> struct is_atomic : std::false_type {};
  template <typename T> struct is_atomic<std::atomic<T>> : std::true_type {};

#if 0
/// reference: C++ templates - 21.2.3
template<typename Derived, typename Value, typename Category, typename Reference = Value&, typename Distance = std::ptrdiff_t> struct IteratorInterface {
public:
  using value_type = typename std::remove_const<Value>::type;
  using reference = Reference;
  using pointer = Value*;
  using difference_type = Distance;
  using iterator_category = Category; /// with type decorations

  // input iterator interface:
  reference operator *() const { ... } 
  pointer operator ->() const { ... } 
  Derived& operator ++() { ... } 
  Derived operator ++(int) { ... } 
  friend bool operator== (IteratorInterface const& lhs, IteratorInterface const& rhs) { ... }
  // bidirectional iterator interface: 
  Derived& operator --() { ... } 
  Derived operator --(int) { ... }
  // random access iterator interface:
  reference operator [](difference_type n) const { ... }
  Derived& operator +=(difference_type n) { ... }
  // ...
  friend difference_type operator -(IteratorInterface const& lhs, IteratorInterface const& rhs) { ... }
  friend bool operator <(IteratorInterface const& lhs, IteratorInterface const& rhs) { ... }
protected:
  auto &self() noexcept { return static_cast<Derived &>(*this); }
  const auto &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }
};

/// input_iterator_tag
// dereference()
// equals()

/// forward_iterator_tag
// increment()

/// bidirectional_iterator_tag
// decrement()

/// random_access_iterator_tag
// random access

/// contiguous_iterator_tag
// contiguous storage
template <typename Derived>
struct ViewInterface {
protected:
  auto &self() noexcept { return static_cast<Derived &>(*this); }
  const auto &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }
};
#endif

}  // namespace zs
