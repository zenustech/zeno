#pragma once

#include <zeno/utils/api.h>
#include <string>

namespace zeno {

// called in zenoedit/main.cpp:
ZENO_API void setExecutableDir(std::string dir);
ZENO_API void setConfigVariable(std::string key, std::string val);

// used from zeno node apply():
ZENO_API std::string getAssetDir(std::string dir);
ZENO_API std::string getAssetDir(std::string dir, std::string extra);
ZENO_API std::string getConfigVariable(std::string key);
ZENO_API void cihouWinPath(std::string &s);

}
