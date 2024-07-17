#include "polyfill.hpp"

#define ZENO_NONCOPYABLE(TypeName) \
	TypeName(TypeName&&) = delete; \
	TypeName(const TypeName&) = delete; \
	TypeName& operator=(const TypeName&) = delete; \
	TypeName& operator=(TypeName&&) = delete;

#define ZENO_DECLARE_FIELD_API(TClass, TSuperClass) \
public:\
	typedef TSuperClass Super;\
	typedef TClass ThisClass;\
	template<typename T> static constexpr bool is_same() { return std::is_same<T, TClass>::value; }

// Library exports controling
#if defined(_MSC_VER)
  #if defined(LIBREFLECT_EXPORTS)
    #define LIBREFLECT_API __declspec(dllexport)
  #else
    #define LIBREFLECT_API __declspec(dllimport)
  #endif
  #define LIBREFLECT_LOCAL 
#elif defined(__GNUC__) || defined(__clang__)
  #define LIBREFLECT_API __attribute__((visibility("default")))
  #define LIBREFLECT_LOCAL __attribute__((visibility("hidden")))
#else
  #define LIBREFLECT_API
  #define LIBREFLECT_LOCAL
#endif

// #define REFLECT_CHECK(expr, MSG) if (!(expr)) { std::cout << "[Reflection Assertion] Failure:\n" << MSG << std::endl; exit(100); }
