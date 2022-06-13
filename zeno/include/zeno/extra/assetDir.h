#pragma once

#include <zeno/utils/api.h>
#include <string>

namespace zeno {

ZENO_API void setAssetRoot(std::string root);  // called in zenoedit
ZENO_API std::string getAssetDir(std::string dir);

}
