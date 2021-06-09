#ifndef FENVINCLUDE_H
#define FENVINCLUDE_H

#ifndef _MSC_VER

//if not Windows, include the standard fenv.v.h
#include <fenv.h>

#else 

//on Windows, cook up the functions we need
#include <float.h>
#include <stdio.h>
#include <errno.h>

#pragma fenv_access(on)


//define the functions we need. I hate Microsoft! Just implement C99 already, c'mon.
#define FE_DOWNWARD   _RC_DOWN
#define FE_UPWARD     _RC_UP
#define FE_TONEAREST  _RC_NEAR
#define FE_TOWARDZERO _RC_CHOP

#include <iostream>

inline int fegetround() {
   unsigned int result;
   errno_t err = _controlfp_s(&result, 0, 0);
   result = result & _MCW_RC;
   return (int)result;
}

inline void fesetround(unsigned int choice) {
   unsigned int result;
   int err = _controlfp_s(&result, choice, _MCW_RC);
}

#endif

#endif
