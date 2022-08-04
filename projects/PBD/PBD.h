#pragma once

#if defined (_MSC_VER)
#undef max
#undef min
#define NOMINMAX
#endif

// disable some warnings on Windows
#if defined (_MSC_VER)
    __pragma(warning (push))
    __pragma(warning (disable : 4244))
    __pragma(warning (disable : 4457))
    __pragma(warning (disable : 4458))
    __pragma(warning (disable : 4389))
    __pragma(warning (disable : 4996))
#elif defined (__GNUC__)
    _Pragma("GCC diagnostic push")
    _Pragma("GCC diagnostic ignored \"-Wconversion\"")
    _Pragma("GCC diagnostic ignored \"-Wsign-compare\"")
    _Pragma("GCC diagnostic ignored \"-Wshadow\"")
#endif


//=============================================================
class PBD
{
public:
    PBD();
};

#if defined (_MSC_VER)
    __pragma(warning (pop))
#elif defined (__GNUC__)
    _Pragma("GCC diagnostic pop")
#endif