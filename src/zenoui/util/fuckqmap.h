#pragma once

#include <QMap>  // cihou the fucking qt company

template <class K, class T>
class FuckQMap
{
    struct FuckQPair : QPair<K, T> {
        using QPair<K, T>::QPair;
        using QPair<K, T>::operator=;

        operator T const &() const noexcept {
            return this->second;
        }

        operator T &() noexcept {
            return this->second;
        }

        T const &value() const noexcept {
            return this->second;
        }

        T &value() noexcept {
            return this->second;
        }

        K const &key() const noexcept {
            return this->first;
        }

        K &key() noexcept {
            return this->first;
        }
    };

    QList<FuckQPair> mother;

public:
    inline FuckQMap() noexcept = default;
    inline FuckQMap(std::initializer_list<std::pair<K,T> > list)
    {
        for (typename std::initializer_list<std::pair<K,T> >::const_iterator it = list.begin(); it != list.end(); ++it)
            insert(it->first, it->second);
    }
    FuckQMap(const FuckQMap<K, T> &other)=default;

    inline ~FuckQMap() = default;

    FuckQMap<K, T> &operator=(const FuckQMap<K, T> &other) = default;
    inline FuckQMap(FuckQMap<K, T> &&other) noexcept = default;

    inline FuckQMap<K, T> &operator=(FuckQMap<K, T> &&other) noexcept=default;
    //explicit FuckQMap(const typename std::map<K, T> &other);
    //std::map<K, T> toStdMap() const;

    bool operator==(const FuckQMap<K, T> &other) const { return mother == other.mother; }
    inline bool operator!=(const FuckQMap<K, T> &other) const { return !(*this == other); }

    inline int size() const { return mother->size(); }

    inline bool isEmpty() const { return mother->size() == 0; }

    void clear() { mother.clear(); }

    int remove(const K &key) {
        auto it = this->find(key);
        if (it == this->end())
            return 0;
        mother.erase(it);
        return 1;
    }

    bool contains(const K &key) const { return std::find_if(mother.begin(), mother.end(), [&] (auto &&a) { return a.first == key; }) != mother.end(); }
    const K key(const T &value, const K &defaultKey = K()) const;
    const T value(const K &key, const T &defaultValue = T()) const;
    T &operator[](const K &key) { if (auto it = this->find(key); it != mother.end()) { return it->second; } else { return this->insert(key, T{})->second; }}
    const T operator[](const K &key) const{ if (auto it = this->find(key); it != mother.end()) { return it->second; } else { return T{}; }}

    QList<K> keys() const
    { QList<K> ret; for (auto &&p: mother) { ret.push_back(p.first); }; return ret; }
    template <int mama = 1>
    QList<K> keys(const T &value) const
    { QList<K> ret; for (auto &&p: mother) { if (p.second == value && mama) ret.push_back(p.first); }; return ret; }
    QList<T> values() const
    { QList<T> ret; for (auto &&p: mother) { ret.push_back(p.second); }; return ret; }
    template <int mama = 1>
    int count(const K &key) const
    { int ret=0; for (auto &&p: mother) { if (p.first == key) { ret++; }; }; return ret; }

    auto begin() { return mother.begin(); }
    auto begin() const { return mother.begin(); }
    auto constBegin() const { return mother.cbegin(); }
    auto end() { return mother.end(); }
    auto end() const { return mother.end(); }
    auto constEnd() const { return mother.cend(); }

    typedef typename QList<FuckQPair>::iterator iterator;
    typedef typename QList<FuckQPair>::const_iterator const_iterator;


    inline const K &firstKey() const { Q_ASSERT(!isEmpty()); return constBegin().key(); }
    inline const K &lastKey() const { Q_ASSERT(!isEmpty()); return (constEnd() - 1).key(); }

    inline T &first() { Q_ASSERT(!isEmpty()); return *begin(); }
    inline const T &first() const { Q_ASSERT(!isEmpty()); return *constBegin(); }
    inline T &last() { Q_ASSERT(!isEmpty()); return *(end() - 1); }
    inline const T &last() const { Q_ASSERT(!isEmpty()); return *(constEnd() - 1); }

    /*class const_iterator;

    class iterator
    {
        friend class const_iterator;
        Node *i;

    public:
        typedef std::bidirectional_iterator_tag iterator_category;
        typedef qptrdiff difference_type;
        typedef T value_type;
        typedef T *pointer;
        typedef T &reference;

        inline iterator() : i(nullptr) { }
        inline iterator(Node *node) : i(node) { }

        inline const K &key() const { return i->key; }
        inline T &value() const { return i->value; }
        inline T &operator*() const { return i->value; }
        inline T *operator->() const { return &i->value; }
        inline bool operator==(const iterator &o) const { return i == o.i; }
        inline bool operator!=(const iterator &o) const { return i != o.i; }

        inline iterator &operator++() {
            i = i->nextNode();
            return *this;
        }
        inline iterator operator++(int) {
            iterator r = *this;
            i = i->nextNode();
            return r;
        }
        inline iterator &operator--() {
            i = i->previousNode();
            return *this;
        }
        inline iterator operator--(int) {
            iterator r = *this;
            i = i->previousNode();
            return r;
        }
        inline iterator operator+(int j) const
        { iterator r = *this; if (j > 0) while (j--) ++r; else while (j++) --r; return r; }
        inline iterator operator-(int j) const { return operator+(-j); }
        inline iterator &operator+=(int j) { return *this = *this + j; }
        inline iterator &operator-=(int j) { return *this = *this - j; }
        friend inline iterator operator+(int j, iterator k) { return k + j; }

#ifndef QT_STRICT_ITERATORS
    public:
        inline bool operator==(const const_iterator &o) const
            { return i == o.i; }
        inline bool operator!=(const const_iterator &o) const
            { return i != o.i; }
#endif
        friend class FuckQMap<K, T>;
        friend class QMultiMap<K, T>;
    };
    friend class iterator;

    class const_iterator
    {
        friend class iterator;
        const Node *i;

    public:
        typedef std::bidirectional_iterator_tag iterator_category;
        typedef qptrdiff difference_type;
        typedef T value_type;
        typedef const T *pointer;
        typedef const T &reference;

        Q_DECL_CONSTEXPR inline const_iterator() : i(nullptr) { }
        inline const_iterator(const Node *node) : i(node) { }
#ifdef QT_STRICT_ITERATORS
        explicit inline const_iterator(const iterator &o)
#else
        inline const_iterator(const iterator &o)
#endif
        { i = o.i; }

        inline const K &key() const { return i->key; }
        inline const T &value() const { return i->value; }
        inline const T &operator*() const { return i->value; }
        inline const T *operator->() const { return &i->value; }
        Q_DECL_CONSTEXPR inline bool operator==(const const_iterator &o) const { return i == o.i; }
        Q_DECL_CONSTEXPR inline bool operator!=(const const_iterator &o) const { return i != o.i; }

        inline const_iterator &operator++() {
            i = i->nextNode();
            return *this;
        }
        inline const_iterator operator++(int) {
            const_iterator r = *this;
            i = i->nextNode();
            return r;
        }
        inline const_iterator &operator--() {
            i = i->previousNode();
            return *this;
        }
        inline const_iterator operator--(int) {
            const_iterator r = *this;
            i = i->previousNode();
            return r;
        }
        inline const_iterator operator+(int j) const
        { const_iterator r = *this; if (j > 0) while (j--) ++r; else while (j++) --r; return r; }
        inline const_iterator operator-(int j) const { return operator+(-j); }
        inline const_iterator &operator+=(int j) { return *this = *this + j; }
        inline const_iterator &operator-=(int j) { return *this = *this - j; }
        friend inline const_iterator operator+(int j, const_iterator k) { return k + j; }

#ifdef QT_STRICT_ITERATORS
    private:
        inline bool operator==(const iterator &o) const { return operator==(const_iterator(o)); }
        inline bool operator!=(const iterator &o) const { return operator!=(const_iterator(o)); }
#endif
        friend class FuckQMap<K, T>;
        friend class QMultiMap<K, T>;
    };
    friend class const_iterator;*/

    /*class key_iterator
    {
        const_iterator i;

    public:
        typedef typename const_iterator::iterator_category iterator_category;
        typedef typename const_iterator::difference_type difference_type;
        typedef K value_type;
        typedef const K *pointer;
        typedef const K &reference;

        key_iterator() = default;
        explicit key_iterator(const_iterator o) : i(o) { }

        const K &operator*() const { return i.key(); }
        const K *operator->() const { return &i.key(); }
        bool operator==(key_iterator o) const { return i == o.i; }
        bool operator!=(key_iterator o) const { return i != o.i; }

        inline key_iterator &operator++() { ++i; return *this; }
        inline key_iterator operator++(int) { return key_iterator(i++);}
        inline key_iterator &operator--() { --i; return *this; }
        inline key_iterator operator--(int) { return key_iterator(i--); }
        const_iterator base() const { return i; }
    };

    typedef QKeyValueIterator<const K&, const T&, const_iterator> const_key_value_iterator;
    typedef QKeyValueIterator<const K&, T&, iterator> key_value_iterator;

    // STL style
    inline iterator begin() { detach(); return iterator(d->begin()); }
    inline const_iterator begin() const { return const_iterator(d->begin()); }
    inline const_iterator constBegin() const { return const_iterator(d->begin()); }
    inline const_iterator cbegin() const { return const_iterator(d->begin()); }
    inline iterator end() { detach(); return iterator(d->end()); }
    inline const_iterator end() const { return const_iterator(d->end()); }
    inline const_iterator constEnd() const { return const_iterator(d->end()); }
    inline const_iterator cend() const { return const_iterator(d->end()); }
    inline key_iterator keyBegin() const { return key_iterator(begin()); }
    inline key_iterator keyEnd() const { return key_iterator(end()); }
    inline key_value_iterator keyValueBegin() { return key_value_iterator(begin()); }
    inline key_value_iterator keyValueEnd() { return key_value_iterator(end()); }
    inline const_key_value_iterator keyValueBegin() const { return const_key_value_iterator(begin()); }
    inline const_key_value_iterator constKeyValueBegin() const { return const_key_value_iterator(begin()); }
    inline const_key_value_iterator keyValueEnd() const { return const_key_value_iterator(end()); }
    inline const_key_value_iterator constKeyValueEnd() const { return const_key_value_iterator(end()); }*/
    iterator erase(iterator it) { return mother.erase(it); }

    // more Qt
    typedef iterator Iterator;
    typedef const_iterator ConstIterator;
    inline int count() const { return mother->size(); }
    auto find(const K &key) { return std::find_if(mother.begin(), mother.end(), [&] (auto &&a) { return a.first == key; }); }
    auto find(const K &key) const { return std::find_if(mother.begin(), mother.end(), [&] (auto &&a) { return a.first == key; }); }
    iterator insert(const K &key, const T &value) { mother.push_back(FuckQPair{key, value}); return mother.end()-1; }
    //iterator insert(const K &key, const T &value) const { mother.push_back(FuckQPair{key, value}); return mother.end()-1; }
    void insert(const FuckQMap<K, T> &map) {
        for (auto const &[k, v]: map) {
            insert(k, v);
        }
    }
    // STL compatibility
    typedef K key_type;
    typedef T mapped_type;
    typedef qptrdiff difference_type;
    typedef int size_type;
    inline bool empty() const { return isEmpty(); }
};
