#include <zeno/extra/assetDir.h>
#include <zeno/utils/filesystem.h>
#include <zeno/utils/Error.h>
#include <map>

namespace zeno {

static std::string g_assetRoot;
static std::map<std::string, std::string> g_cfgvars;

ZENO_API void setConfigVariable(std::string key, std::string val) {
    g_cfgvars[key] = val;
}

ZENO_API std::string getConfigVariable(std::string key) {
    auto it = g_cfgvars.find(key);
    if (it != g_cfgvars.end())
        return it->second;
    return {};
}

ZENO_API void setExecutableDir(std::string dir) {
#ifdef _WIN32
    g_assetRoot = dir + "/assets/";
#else
    g_assetRoot = dir + "/../share/Zeno/assets/";
#endif
}

ZENO_API std::string getAssetDir(std::string dir) {
    //dir = fs::absolute(dir).string();
    if (fs::exists(dir))
        return dir;
#ifdef _WIN32
    if (auto i = dir.find(':'); i != std::string::npos)
        dir.replace(i, 1, "_pan");
#endif
    if (auto edir = g_assetRoot + dir; fs::exists(edir))
        return edir;
    throw makeError("cannot find asset directory: " + dir);
}

}
