#pragma once


#if defined ( _WIN32 )
  #define __FUNC__ __FUNCTION__
#else
  #define __FUNC__ __func__
#endif

