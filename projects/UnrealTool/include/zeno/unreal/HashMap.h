#pragma once
#include <memory>
#include <string>
#include "SimpleList.h"

namespace zeno {
// Hash node class template
template <typename K, typename V>
class HashNode {
  public:
    HashNode(const K &key, const V &value) :
          key(key), value(value), next(NULL) {
    }

    K getKey() const {
        return key;
    }

    V getValue() const {
        return value;
    }

    void setValue(V value) {
        HashNode::value = value;
    }

    HashNode *getNext() const {
        return next;
    }

    void setNext(HashNode *next) {
        HashNode::next = next;
    }

  private:
    // key-value pair
    K key;
    V value;
    // next bucket with the same key
    HashNode *next;
};

// Default hash function class
template <typename K>
struct KeyHash {
    unsigned long operator()(const K& key) const
    {
        return reinterpret_cast<unsigned long>(key) % 16;
    }
};

template <>
struct KeyHash<std::string> {
    unsigned long operator()(const std::string& key) const
    {
        std::hash<std::string> hash_func;
        return hash_func(key);
    }
};

template <typename K, typename V, typename F = KeyHash<K>>
class HashMap {
  public:
    explicit HashMap(size_t size = 16) : mapSize(size >= 16 ? size : 16) {
        // construct zero initialized hash table of size
        table = new HashNode<K, V> *[mapSize]();
    }

    ~HashMap() {
        // destroy all buckets one by one
        for (int i = 0; i < mapSize; ++i) {
            HashNode<K, V> *entry = table[i];
            while (entry != NULL) {
                HashNode<K, V> *prev = entry;
                entry = entry->getNext();
                delete prev;
            }
            table[i] = NULL;
        }
        // destroy the hash table
        delete [] table;
    }

    bool get(const K &key, V &value) const {
        unsigned long hashValue = hashFunc(key);
        HashNode<K, V> *entry = table[hashValue];

        while (entry != NULL) {
            if (entry->getKey() == key) {
                value = entry->getValue();
                return true;
            }
            entry = entry->getNext();
        }
        return false;
    }

    void put(const K &key, const V &value) {
        unsigned long hashValue = hashFunc(key);
        HashNode<K, V> *prev = NULL;
        HashNode<K, V> *entry = table[hashValue];

        while (entry != NULL && entry->getKey() != key) {
            prev = entry;
            entry = entry->getNext();
        }

        if (entry == NULL) {
            entry = new HashNode<K, V>(key, value);
            if (prev == NULL) {
                // insert as first bucket
                table[hashValue] = entry;
            } else {
                prev->setNext(entry);
            }
        } else {
            // just update the value
            entry->setValue(value);
        }
    }

    void remove(const K &key) {
        unsigned long hashValue = hashFunc(key);
        HashNode<K, V> *prev = NULL;
        HashNode<K, V> *entry = table[hashValue];

        while (entry != NULL && entry->getKey() != key) {
            prev = entry;
            entry = entry->getNext();
        }

        if (entry == NULL) {
            // key not found
            return;
        }
        else {
            if (prev == NULL) {
                // remove first bucket of the list
                table[hashValue] = entry->getNext();
            } else {
                prev->setNext(entry->getNext());
            }
            delete entry;
        }
    }

  private:
    // hash table
    HashNode<K, V> **table;
    F hashFunc;
    size_t mapSize;
};
}
