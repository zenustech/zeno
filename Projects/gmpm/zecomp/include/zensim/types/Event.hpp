// reference: https://github.com/TheLartians/Observe.git

#pragma once
#include <mutex>
#include <shared_mutex>

#include "Object.h"
#include "zensim/types/Function.h"

namespace zs {

  using ListenerCallback = Object;
  template <typename... Args> struct Event;

  struct Listener;
  ZS_type_name(Listener);
  struct Listener : Object {
    Listener() = default;
    /// allow move
    Listener(Listener &&o) noexcept = default;
    Listener &operator=(Listener &&o) noexcept = default;
    /// prevent copy
    Listener(const Listener &o) = delete;
    Listener &operator=(const Listener &o) = delete;
    /// assign real callback
    template <typename L> constexpr Listener(L &&o) : _listener{new L(std::move(o))} {}
    template <typename L> constexpr Listener &operator=(L &&o) noexcept {
      _listener.reset(new L(std::move(o)));
      return *this;
    }
    template <typename Callback, typename... Args>
    constexpr void listen(Event<Args...> &event, Callback &&callback) {
      _listener.reset(new typename Event<Args...>::Listener(
          event.createListener(std::forward<Callback>(callback))));
    }
    explicit operator bool() const noexcept { return static_cast<bool>(_listener); }
    void reset() { _listener.reset(); }

  protected:
    std::unique_ptr<ListenerCallback> _listener;
  };

  template <typename... Args> struct Event : Object {
    using Callback = std::function<void(const Args &...)>;
    using CallbackID = std::size_t;

    struct StoredCallback {
      CallbackID id{};
      std::shared_ptr<Callback> callback{};
    };

    struct EventMeta {
      CallbackID counter{0};
      std::vector<StoredCallback> listeners{};
      mutable std::shared_mutex listenerMutex{};
    };

    /// allow move
    constexpr Event() : _meta{std::make_shared<EventMeta>()} {}
    constexpr Event(Event &&o) noexcept : _meta{std::move(o._meta)} {}
    constexpr Event &operator=(Event &&o) noexcept {
      std::swap(_meta, o._meta);
      return *this;
    }
    /// prevent copy
    constexpr Event(const Event &o) = delete;
    constexpr Event &operator=(const Event &o) = delete;

    struct Listener : Object {
    private:
      std::weak_ptr<EventMeta> _meta;
      CallbackID _id;

    public:
      Listener() = default;
      Listener(std::weak_ptr<EventMeta> meta, CallbackID id) : _meta(std::move(meta)), _id(id) {}

      Listener(Listener &&other) noexcept = default;
      Listener &operator=(Listener &&other) noexcept = default;

      Listener(const Listener &other) = delete;
      Listener &operator=(const Listener &other) = delete;

      void listen(const Event &event, const Callback &callback) {
        reset();
        *this = event.createObserver(callback);
      }

      void reset() {
        if (auto m = _meta.lock()) {
          std::unique_lock<std::shared_mutex> lk{m->listenerMutex};
          auto it = std::find_if(m->listeners.begin(), m->listeners.end(),
                                 [&](auto &o) { return o.id == _id; });
          if (it != m->listeners.end()) {
            m->listeners.erase(it);
          }
        }
        _meta.reset();
      }

      ~Listener() { reset(); }
    };

    void emit(Args... args) const {
      std::vector<std::weak_ptr<Callback>> callbacks;
      {
        std::unique_lock<std::shared_mutex> lk{_meta->listenerMutex};
        callbacks.resize(_meta->listeners.size());
        std::transform(_meta->listeners.begin(), _meta->listeners.end(), callbacks.begin(),
                       [](auto &c) { return c.callback; });
      };
      for (auto &weakCallback : callbacks)
        if (auto callback = weakCallback.lock()) std::invoke(*callback, args...);
    }

    Listener createListener(const Callback &cb) const { return Listener(_meta, addListener(cb)); }

    CallbackID connect(const Callback &cb) const { return addListener(cb); }
    void disconnect(CallbackID id) const { Listener(_meta, id).reset(); }

    void reset() const {
      std::unique_lock<std::shared_mutex> lk(_meta->listenerMutex);
      _meta->listeners.clear();
      _meta->listeners.counter = 0;
    }
    std::size_t listenerCount() const noexcept {
      std::shared_lock<std::shared_mutex> lk(_meta->listenerMutex);
      return _meta->listeners.size();
    }

  protected:
    CallbackID addListener(const Callback &cb) const {
      std::unique_lock<std::shared_mutex> lk(_meta->listenerMutex);
      _meta->listeners.emplace_back(StoredCallback{_meta->counter, std::make_shared<Callback>(cb)});
      return _meta->counter++;
    }

    std::shared_ptr<EventMeta> _meta;
  };

}  // namespace zs