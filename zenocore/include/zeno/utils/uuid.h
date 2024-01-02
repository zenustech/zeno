#ifndef __uuid_h__
#define __uuid_h__


#include <stdio.h>
#include <string>
#include <zeno/utils/api.h>

namespace zeno
{
    ZENO_API std::string generateUUID();
}

#endif