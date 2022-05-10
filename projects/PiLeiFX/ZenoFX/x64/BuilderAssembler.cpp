//
// Created by admin on 2022/5/10.
//
#include <memory>

namespace zfx {
    #define ERROR_IF(x) do { \
    if(x) {                  \
        error("%s", #x);                     \
        }                         \
    } while(0)

   struct ImplAssembler {

   };
}

