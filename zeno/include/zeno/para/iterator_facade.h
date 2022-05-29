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
    using difference_type = Difference;
};

template
< class Derived
, class Value
, class Reference
, class Difference
>
class iterator_facade_forward : public iterator_facade_base
< Derived
, Value
, Reference
, Difference
> {
    using self_type = Derived;

public:
    void operator++() {
        this->self().increment();
    }

    self_type operator++(int) {
        self_type old = this->self();
        ++*this;
        return old;
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
, class Difference
>
class iterator_facade_bidirectional : public iterator_facade_forward
< Derived
, Value
, Reference
, Difference
> {

    using self_type = Derived;

public:
    self_type &operator--() {
        this->self().decrement();
    }

    self_type operator--(int) {
        self_type old = this->self();
        --*this;
        return old;
    }
};

template
< class Derived
, class Value
, class Reference
, class Difference
>
class iterator_facade_random_access : public iterator_facade_bidirectional
< Derived
, Value
, Reference
, Difference
> {
    using self_type = Derived;

public:
    self_type &operator+=(Difference n) {
        this->self().advance(n);
    }

    self_type &operator-=(Difference n) {
        this->self().advance(-n);
    }

    self_type operator-(Difference n) {
        this->self();
    }
};

/* template <class Iterator, class = void>
struct has_increment : std::false_type {
};

template <class Iterator>
struct has_increment<Iterator, std::void_t<decltype(
    std::declval<Iterator>().increment(),
    0)>> : std::true_type {
};

template <class Iterator, class = void>
struct has_dereference : std::false_type {
};

template <class Iterator>
struct has_dereference<Iterator, std::void_t<decltype(
    std::declval<Iterator>().dereference(),
    0)>> : std::true_type {
};

template <class Iterator, class = void>
struct detect_category {
};

template <class Iterator>
struct detect_category<Iterator, std::enable_if_t<
    has_increment<Iterator>
    >> {
    using type = std::forward_iterator_tag;
}; */

}

template
< class Derived
, class Value //= decltype(std::declval<Derived>().dereference())
, class Category //= typename iterator_facade_details::detect_category<Derived>::type
, class Reference = std::add_lvalue_reference_t<Value>
, class Difference = std::ptrdiff_t
>
struct iterator_facade {
};

template
< class Derived
, class Value
, class Reference
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::input_iterator_tag
, Reference
, Difference
> : iterator_facade_details::iterator_facade_forward
< Derived
, Value
, Reference
, Difference
> {
    using iterator_category = std::input_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::output_iterator_tag
, Reference
, Difference
> : iterator_facade_details::iterator_facade_forward
< Derived
, Value
, Reference
, Difference
> {
    using iterator_category = std::output_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::forward_iterator_tag
, Reference
, Difference
> : iterator_facade_details::iterator_facade_forward
< Derived
, Value
, Reference
, Difference
> {
    using iterator_category = std::forward_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::bidirectional_iterator_tag
, Reference
, Difference
> : iterator_facade_details::iterator_facade_bidirectional
< Derived
, Value
, Reference
, Difference
> {
    using iterator_category = std::bidirectional_iterator_tag;
};

template
< class Derived
, class Value
, class Reference
, class Difference
>
struct iterator_facade
< Derived
, Value
, std::random_access_iterator_tag
, Reference
, Difference
> : iterator_facade_details::iterator_facade_random_access
< Derived
, Value
, Reference
, Difference
> {
    using iterator_category = std::random_access_iterator_tag;
};

}
