#pragma once

#ifdef _MSC_VER
# ifdef DLL_ZENO
#  define ZENO_API __declspec(dllexport)
# else
#  define ZENO_API __declspec(dllimport)
# endif
#else
# define ZENO_API
#endif
