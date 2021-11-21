#include <type_traits>
#include <tuple>

namespace bate::iter {

struct null_iterator {
    template <class T>
    bool operator!=(T const &other) const {
        return (bool)other;
    }

    template <class T>
    bool operator==(T const &other) const {
        return !(bool)other;
    }
};

template <class T>
inline bool operator!=(T const &other, null_iterator) {
    return (bool)other;
}

template <class T>
inline bool operator==(T const &other, null_iterator) {
    return !(bool)other;
}

inline constexpr null_iterator npos{};

template <class T>
struct iterator_base {
    //static_assert(std::is_base_of_v<iterator_base, T>);

    inline T const &begin() const {
        return *reinterpret_cast<T const *>(this);
    }

    inline auto end() const {
        return npos;
    }

    template <class _ = void>
    void next(int skip) {
        static_assert(sizeof(_), "next(int) not implemented");
    }

    template <class _ = void>
    void eof() const {
        static_assert(sizeof(_), "eof() not implemented");
    }

    template <class _ = void>
    void get() const {
        static_assert(sizeof(_), "get() not implemented");
    }

    inline auto &get() {
        return const_cast<T const *>(reinterpret_cast<T *>(this))->get();
    }

    inline T &operator++() {
        that()->next(1);
        return *that();
    }

    inline T &operator--() {
        that()->next(-1);
        return *that();
    }

    inline T operator++(int) {
        auto old = *that();
        that()->next(1);
        return old;
    }

    inline T operator--(int) {
        auto old = *that();
        that()->next(-1);
        return old;
    }

    inline T &operator+=(int skip) {
        that()->next(skip);
        return *that();
    }

    inline T &operator-=(int skip) {
        that()->next(-skip);
        return *that();
    }

    inline operator bool() const {
        return that()->alive();
    }

    inline decltype(auto) operator*() const {
        return that()->get();
    }

    inline decltype(auto) operator*() {
        return that()->get();
    }

private:
    T *that() {
        return reinterpret_cast<T *>(this);
    }

    T const *that() const {
        return reinterpret_cast<T const *>(this);
    }
};

template <class T = int>
struct range : iterator_base<range<T>> {
    using value_type = T;

    T m_now;
    T m_end;
    T m_skip;

    range
        ( T const &now_
        , T const &end_
        , T const &skip_ = T(1)
        )
    : m_now(now_)
    , m_end(end_)
    , m_skip(skip_)
    {}

    void next(int skip) {
        m_now += skip * m_skip;
    }

    bool alive() const {
        return m_now < m_end;
    }

    value_type const &get() const {
        return m_now;
    }
};

template <class T>
struct slice : iterator_base<slice<T>> {
    using value_type = typename T::value_type;

    T m_iter;
    int m_now;
    int m_end;
    int m_skip;

    slice
        ( T const &iter
        , int begin_
        , int end_
        , int skip_ = 1
        )
        : m_iter(iter)
        , m_now(begin_)
        , m_end(end_)
        , m_skip(skip_)
        {}

    void next(int skip) {
        m_now += skip * m_skip;
        m_iter += skip * m_skip;
    }

    bool alive() const {
        return m_iter && m_now < m_end;
    }

    auto get() const {
        return *m_iter;
    }
};


template <class ...Ts>
struct zip : iterator_base<zip<Ts...>> {
    using value_type = std::tuple<typename Ts::value_type...>;

    std::tuple<Ts...> m_iters;

    explicit zip
        ( Ts const &...iters
        )
        : m_iters(iters...)
        {}

    template <size_t ...Inds>
    void _next(std::index_sequence<Inds...>, int skip) {
        ((std::get<Inds>(m_iters) += skip), ...);
    }

    void next(int skip) {
        return _next(std::make_index_sequence<sizeof...(Ts)>(), skip);
    }

    template <size_t ...Inds>
    bool _alive(std::index_sequence<Inds...>) const {
        return (bool(std::get<Inds>(m_iters)) && ... && (sizeof...(Ts) != 0));
    }

    bool alive() const {
        return _alive(std::make_index_sequence<sizeof...(Ts)>());
    }

    template <size_t ...Inds>
    decltype(auto) _get(std::index_sequence<Inds...>) const {
        return value_type(*std::get<Inds>(m_iters)...);
    }

    decltype(auto) get() const {
        return _get(std::make_index_sequence<sizeof...(Ts)>());
    }
};

}
