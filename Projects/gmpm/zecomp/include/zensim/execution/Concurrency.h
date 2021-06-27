#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>

#include "zensim/TypeAlias.hpp"
#include "zensim/types/Optional.h"

namespace zs {

  template <typename F, typename... Ts> inline auto reallyAsync(F &&f, Ts &&...params) {
    return std::async(std::launch::async, std::forward<F>(f), std::forward<Ts>(params)...);
  }

  /// <<C++ concurrency in action>>
  template <typename T> class threadsafe_queue {
  private:
    mutable std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;

  public:
    threadsafe_queue() {}
    threadsafe_queue(threadsafe_queue const &other) {
      std::lock_guard<std::mutex> lk(other.mut);
      data_queue = other.data_queue;
    }
    void push(const T &new_value) {
      std::lock_guard<std::mutex> lk(mut);
      data_queue.push(new_value);
      data_cond.notify_one();
    }
    void push(T &&new_value) {
      std::lock_guard<std::mutex> lk(mut);
      data_queue.push(std::move(new_value));
      data_cond.notify_one();
    }
    void wait_and_pop(T &value) {
      /// spinlock should be better
      std::unique_lock<std::mutex> lk(mut);
      data_cond.wait(lk, [this] { return !data_queue.empty(); });
      value = data_queue.front();
      data_queue.pop();
    }
    decltype(auto) wait_and_pop() {
      /// spinlock should be better
      std::unique_lock<std::mutex> lk(mut);
      data_cond.wait(lk, [this] { return !data_queue.empty(); });
      T value = data_queue.front();
      data_queue.pop();
      return value;
    }
    std::shared_ptr<T> wait_and_pop_ptr() {
      std::unique_lock<std::mutex> lk(mut);
      data_cond.wait(lk, [this] { return !data_queue.empty(); });
      std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
      data_queue.pop();
      return res;
    }
    bool try_pop(T &value) {
      std::lock_guard<std::mutex> lk(mut);
      if (data_queue.empty()) return false;
      value = data_queue.front();
      data_queue.pop();
      return true;
    }
    decltype(auto) try_pop() {
      std::lock_guard<std::mutex> lk(mut);
      if (data_queue.empty()) return optional<T>{};
      optional<T> value{data_queue.front()};
      data_queue.pop();
      return value;
    }
    std::shared_ptr<T> try_pop_ptr() {
      std::lock_guard<std::mutex> lk(mut);
      if (data_queue.empty()) return std::shared_ptr<T>();
      std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
      data_queue.pop();
      return res;
    }
    bool empty() const {
      std::lock_guard<std::mutex> lk(mut);
      return data_queue.empty();
    }
  };

  template <typename KeyT, typename ValueT> struct concurrent_map {
    using key_t = KeyT;
    using value_t = ValueT;

    std::map<key_t, value_t> _map{};
    mutable std::shared_mutex _rw{};

    auto begin() { return std::begin(_map); }
    auto end() { return std::end(_map); }
    auto begin() const { return std::begin(_map); }
    auto end() const { return std::end(_map); }

    void set(const key_t &key, const value_t &value) {
      std::unique_lock<std::shared_mutex> lk(_rw);
      _map.insert_or_assign(key, value);
    }
    void erase(const key_t &key) {
      std::unique_lock<std::shared_mutex> lk(_rw);
      _map.erase(key);
    }
    template <typename... Args> decltype(auto) emplace(Args &&...args) {
      std::unique_lock<std::shared_mutex> lk(_rw);
      return _map.emplace(std::forward<Args>(args)...);
    }
    const value_t &get(const key_t &key) const {
      std::shared_lock<std::shared_mutex> lk(_rw);
      return _map.at(key);
    }
    value_t &get(const key_t &key) {
      std::shared_lock<std::shared_mutex> lk(_rw);
      return _map.at(key);
    }
    /// pointer-semantic in case value_t is unique_ptr
    ConstRefPtr<value_t> find(const key_t &key) const {
      std::shared_lock<std::shared_mutex> lk(_rw);
      if (auto it = _map.find(key); it != _map.end()) return &(it->second);
      return nullptr;
    }
    RefPtr<value_t> find(const key_t &key) {
      std::shared_lock<std::shared_mutex> lk(_rw);
      if (auto it = _map.find(key); it != _map.end()) return &(it->second);
      return nullptr;
    }
  };

}  // namespace zs
