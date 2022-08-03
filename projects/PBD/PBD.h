#ifndef __PBD_h__
#define __PBD_h__

#if defined (_MSC_VER)
#undef max
#undef min
#define NOMINMAX
#endif

#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <iterator>
#include <algorithm>
#include <limits>

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
    //=============================================================
    /** Constructor */
    PBD();

private:
    
};


//=============================================================
/* IMPLEMENTATION */
//=============================================================



#if defined (_MSC_VER)
    __pragma(warning (pop))
#elif defined (__GNUC__)
    _Pragma("GCC diagnostic pop")
#endif

#endif /* __PBD_h__ */
