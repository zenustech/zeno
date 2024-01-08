#ifndef __QKEYLIST_H__
#define __QKEYLIST_H__

#include <QMap>

//change the name from FuckQMap, which was implemented by pyb, to QKeyVector.

template <class K, class T>
class QKeyList
{
    struct _QPair : QPair<K, T> {
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

    QList<_QPair> mother;

public:
    inline QKeyList() noexcept = default;
    inline QKeyList(std::initializer_list<std::pair<K,T> > list)
    {
        for (typename std::initializer_list<std::pair<K,T> >::const_iterator it = list.begin(); it != list.end(); ++it)
            insert(it->first, it->second);
    }
    QKeyList(const QKeyList<K, T> &other)=default;

    inline ~QKeyList() = default;

    QKeyList<K, T> &operator=(const QKeyList<K, T> &other) = default;
    inline QKeyList(QKeyList<K, T> &&other) noexcept = default;

    inline QKeyList<K, T> &operator=(QKeyList<K, T> &&other) noexcept=default;
    //explicit QKeyList(const typename std::map<K, T> &other);
    //std::map<K, T> toStdMap() const;

    bool operator==(const QKeyList<K, T> &other) const { return mother == other.mother; }
    inline bool operator!=(const QKeyList<K, T> &other) const { return !(*this == other); }

    inline int size() const { return mother.size(); }

    inline bool isEmpty() const { return mother.size() == 0; }

    void clear() { mother.clear(); }

    int remove(const K &key) {
        auto it = this->find(key);
        if (it == this->end())
            return 0;
        mother.erase(it);
        return 1;
    }

    void swap(int left, int right) {
        _QPair lItem = mother[left];
        mother[left] = mother[right];
        mother[right] = lItem;
    }

    void move(int from, int to) {
        mother.move(from, to);
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

    typedef typename QList<_QPair>::iterator iterator;
    typedef typename QList<_QPair>::const_iterator const_iterator;


    inline const K &firstKey() const { Q_ASSERT(!isEmpty()); return constBegin()->key(); }
    inline const K &lastKey() const { Q_ASSERT(!isEmpty()); return (constEnd() - 1)->key(); }

    inline T &first() { Q_ASSERT(!isEmpty()); return *begin(); }
    inline const T &first() const { Q_ASSERT(!isEmpty()); return *constBegin(); }
    inline T &last() { Q_ASSERT(!isEmpty()); return *(end() - 1); }
    inline const T &last() const { Q_ASSERT(!isEmpty()); return *(constEnd() - 1); }

    iterator erase(iterator it) { return mother.erase(it); }

    // more Qt
    typedef iterator Iterator;
    typedef const_iterator ConstIterator;
    inline int count() const { return mother->size(); }
    auto find(const K &key) { return std::find_if(mother.begin(), mother.end(), [&] (auto &&a) { return a.first == key; }); }
    auto find(const K &key) const { return std::find_if(mother.begin(), mother.end(), [&] (auto &&a) { return a.first == key; }); }
    iterator insert(const K &key, const T &value) { mother.push_back(_QPair{key, value}); return mother.end()-1; }
    iterator push_front(const K &key, const T &value) { mother.push_front(_QPair{key, value}); return mother.begin(); }
    //iterator insert(const K &key, const T &value) const { mother.push_back(_QPair{key, value}); return mother.end()-1; }
    void insert(const QKeyList<K, T> &map) {
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

#endif