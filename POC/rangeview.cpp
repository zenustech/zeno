#include <vector>
#include <iterator>
#include <concepts>
#include <iostream>
#include <memory>


template <class T>
concept is_iterator =
requires (T t, T tt)
{
    *t;
    t != tt;
};

template <class T>
concept is_forward_iterator = is_iterator<T> &&
requires (T t)
{
    ++t;
};

template <class T>
concept is_bidirectional_iterator = is_iterator<T> &&
requires (T t)
{
    ++t;
    --t;
};

template <class T>
concept is_random_iterator = is_iterator<T> &&
requires (T t, T tt, std::size_t i)
{
    t - tt;
    t += i;
    t -= i;
};


template <class T>
concept is_ranged = requires (T t)
{
    t.begin();
    t.end();
};


template <is_iterator T>
struct range
{
    T m_begin;
    T m_end;

    using iterator = T;

    constexpr range(T begin, T end)
        : m_begin(std::move(begin)), m_end(std::move(end))
    {}

    template <is_ranged R>
    constexpr range(R &&r) : range(r.begin(), r.end())
    {
    }

    constexpr iterator begin() const
    {
        return m_begin;
    }

    constexpr iterator end() const
    {
        return m_end;
    }
};

template <is_ranged R>
range(R &&r) -> range<decltype(std::declval<R>().begin())>;


template <class F>
struct transformer
{
    F m_f;

    constexpr transformer(F f)
        : m_f(std::move(f))
    {
    }

    constexpr decltype(auto) operator()(is_ranged auto &...rs) const
    {
        return m_f(range(rs)...);
    }

    friend constexpr decltype(auto) operator|(is_ranged auto &r, transformer const &self)
    {
        return self(r);
    }
};


template <class R, class F>
struct map_range
{
    R m_r;
    F m_f;

    struct iterator
    {
        typename R::iterator m_it;
        F m_f;

        constexpr decltype(auto) operator*() const
        {
            return m_f(*m_it);
        }

        constexpr iterator &operator++()
        {
            m_it++;
            return *this;
        }

        constexpr iterator &operator--()
        {
            m_it--;
            return *this;
        }

        constexpr iterator &operator+=(std::size_t i)
        {
            m_it += i;
            return *this;
        }

        constexpr iterator &operator-=(std::size_t i)
        {
            m_it -= i;
            return *this;
        }

        constexpr std::ptrdiff_t operator-(iterator const &o) const
        {
            return m_it - o.m_it;
        }

        constexpr std::ptrdiff_t operator!=(iterator const &o) const
        {
            return m_it != o.m_it;
        }
    };

    constexpr iterator begin() const
    {
        return {m_r.begin(), m_f};
    }

    constexpr iterator end() const
    {
        return {m_r.end(), m_f};
    }
};

inline constexpr auto map(auto f)
{
    return transformer([=] (auto r) {
        return map_range<decltype(r), decltype(f)>{r, f};
    });
}

int main()
{
    std::vector<std::unique_ptr<int>> arr = {new int, new int};
    for (auto *x: arr | map([] (auto &&t) { return t.get(); })) {
        std::cout << x << std::endl;
    }
}
