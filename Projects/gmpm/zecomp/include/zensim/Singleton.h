#pragma once

#include <assert.h>
// http ://www.boost.org/doc/libs/1_39_0/boost/pool/detail/singleton.hpp

namespace zs {

  /*
   *	@note	Singleton
   */
  // T must be: no-throw default constructible and no-throw destructible
  template <typename T> struct Singleton {
  public:
    static T &instance() {
      static T _instance{};
      return _instance;
    }
  };

  /*
   *	\class	ManagedSingleton
   *	\note	Used in global systems
   */
  template <typename T> struct ManagedSingleton {
    static void startup() {
      assert(_pInstance == nullptr);
      _pInstance = new T();
    }
    static void shutdown() {
      assert(_pInstance != nullptr);
      delete _pInstance;
    }

  protected:
    static T *_pInstance;

  public:
    ///
    T *operator->() { return _pInstance; }

    static T &instance() {
      assert(_pInstance != nullptr);
      return *_pInstance;
    }
    static T *getInstance() {
      assert(_pInstance != nullptr);
      return _pInstance;
    }
    static const T *getConstInstance() {
      assert(_pInstance != nullptr);
      return _pInstance;
    }
  };
  template <typename T> T *ManagedSingleton<T>::_pInstance = nullptr;
}  // namespace zs
