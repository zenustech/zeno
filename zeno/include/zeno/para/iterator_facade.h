#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

namespace zeno {

namespace iterator_facade_details {

template <class T>
struct arrow_proxy {
    T t;

    T *operator->() {
        return &t;
    }

    std::add_const_t<T> *operator->() const {
        return &t;
    }
};

template <class T>
arrow_proxy(T) -> arrow_proxy<T>;

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
class iterator_facade_base {
    using self_type = Derived;

protected:
    self_type &self() {
        return static_cast<self_type &>(*this);
    }

    self_type const &self() const {
        return static_cast<self_type const &>(*this);
    }

public:
    using value_type = Value;
    using reference = Reference;
    using pointer = Pointer;
    using difference_type = Difference;
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
class iterator_facade_forward : public iterator_facade_base
< Derived
, Value
, Reference
, Pointer
, Difference
> {
    using self_type = Derived;

public:
    self_type &operator++() {
        this->self().increment();
        return this->self();
    }

    self_type operator++(int) {
        self_type tmp = this->self();
        ++*this;
        return tmp;
    }

    friend bool operator==(self_type const &lhs, self_type const &rhs) {
        return lhs.equal_to(rhs);
    }

    friend bool operator!=(self_type const &lhs, self_type const &rhs) {
        return !(lhs == rhs);
    }

    decltype(auto) operator*() const {
        return this->self().dereference();
    }

    auto *operator->() const {
        decltype(auto) ref = **this;
        if constexpr (std::is_reference_v<decltype(ref)>) {
            return std::addressof(ref);
        } else {
            return arrow_proxy{std::move(ref)};
        }
    }
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
class iterator_facade_bidirectional : public iterator_facade_forward
< Derived
, Value
, Reference
, Pointer
, Difference
> {

    using self_type = Derived;

public:
    self_type &operator--() {
        this->self().decrement();
        return this->self();
    }

    self_type operator--(int) {
        self_type tmp = this->self();
        --*this;
        return tmp;
    }
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
class iterator_facade_random_access : public iterator_facade_bidirectional
< Derived
, Value
, Reference
, Pointer
, Difference
> {
    using self_type = Derived;

public:
    self_type &operator+=(Difference n) {
        this->self().advance(n);
        return this->self();
    }

    self_type &operator-=(Difference n) {
        this->self().advance(-n);
        return this->self();
    }

    self_type operator+(Difference n) const {
        auto tmp = this->self();
        tmp.advance(+n);
        return tmp;
    }

    self_type operator-(Difference n) const {
        auto tmp = this->self();
        tmp.advance(-n);
        return tmp;
    }

    friend Difference operator-(self_type const &lhs, self_type const &rhs) {
        return rhs.distance_to(lhs);
    }

    friend bool operator<(self_type const &lhs, self_type const &rhs) {
        return (lhs - rhs) < 0;
    }

    friend bool operator>(self_type const &lhs, self_type const &rhs) {
        return (lhs - rhs) > 0;
    }

    friend bool operator<=(self_type const &lhs, self_type const &rhs) {
        return (lhs - rhs) <= 0;
    }

    friend bool operator>=(self_type const &lhs, self_type const &rhs) {
        return (lhs - rhs) >= 0;
    }

    decltype(auto) operator[](Difference n) const {
        auto tmp = this->self();
        tmp.advance(n);
        return tmp.dereference();
    }
};

template <class Derived, class Value, class = void>
struct try_get_reference_type {
    using type = std::add_lvalue_reference_t<Value>;
};

template <class Derived, class Value>
struct try_get_reference_type<Derived, Value, std::void_t<typename Derived::reference>> {
    using type = typename Derived::reference;
};

template <class Derived, class Value, class = void>
struct try_get_pointer_type {
    using type = std::add_pointer_t<Value>;
};

template <class Derived, class Value>
struct try_get_pointer_type<Derived, Value, std::void_t<typename Derived::pointer>> {
    using type = typename Derived::pointer;
};

template <class Derived, class = void>
struct try_get_difference_type {
    using type = std::ptrdiff_t;
};

template <class Derived>
struct try_get_difference_type<Derived, std::void_t<typename Derived::difference_type>> {
    using type = typename Derived::difference_type;
};

}

template
< class Derived
, class Value = typename Derived::value_type
, class Category = typename Derived::iterator_category
, class Reference = typename iterator_facade_details::try_get_reference_type<Derived, Value>::type
, class Pointer = typename iterator_facade_details::try_get_pointer_type<Derived, Value>::type
, class Difference = typename iterator_facade_details::try_get_difference_type<Derived>::type
>
struct iterator_facade {
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::input_iterator_tag
, Reference
, Pointer
, Difference
> : iterator_facade_details::iterator_facade_forward
< Derived
, Value
, Reference
, Pointer
, Difference
> {
    using iterator_category = std::input_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::output_iterator_tag
, Reference
, Pointer
, Difference
> : iterator_facade_details::iterator_facade_forward
< Derived
, Value
, Reference
, Pointer
, Difference
> {
    using iterator_category = std::output_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::forward_iterator_tag
, Reference
, Pointer
, Difference
> : iterator_facade_details::iterator_facade_forward
< Derived
, Value
, Reference
, Pointer
, Difference
> {
    using iterator_category = std::forward_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::bidirectional_iterator_tag
, Reference
, Pointer
, Difference
> : iterator_facade_details::iterator_facade_bidirectional
< Derived
, Value
, Reference
, Pointer
, Difference
> {
    using iterator_category = std::bidirectional_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Pointer
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::random_access_iterator_tag
, Reference
, Pointer
, Difference
> : iterator_facade_details::iterator_facade_random_access
< Derived
, Value
, Reference
, Pointer
, Difference
> {
    using iterator_category = std::random_access_iterator_tag;
};

}

/*template
< class Derived
, class Value
, class Category
, class Reference
, class Pointer
, class Difference
>
struct allocator_traits<::zeno::iterator_facade
< Derived
, Value
, Category
, Reference
, Pointer
, Difference
>> {
    using value_type = Value;
    using iterator_category = Category;
    using reference = Reference;
    using pointer = Pointer;
    using difference_type = Difference;
};*/
