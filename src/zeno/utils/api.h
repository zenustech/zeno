#pragma once

#if defined(_MSC_VER)
# if defined(ZENO_DLLEXPORT)
#  define ZENO_API __declspec(dllexport)
# elif defined(ZENO_DLLIMPORT)
#  define ZENO_API __declspec(dllimport)
# else
#  define ZENO_API
# endif
#else
# define ZENO_API
#endif