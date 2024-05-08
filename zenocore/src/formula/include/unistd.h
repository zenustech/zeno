#ifdef __MINGW32__

#  include_next <unistd.h>

#else

#  include <io.h>
#  include <process.h>

#endif
