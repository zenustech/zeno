#include <zeno/extra/assetDir.h>
#include <zeno/utils/filesystem.h>
#include <zeno/utils/Error.h>

namespace zeno {

static std::string g_assetRoot;

ZENO_API void setAssetRoot(std::string root) {
    g_assetRoot = std::move(root);
}

ZENO_API std::string getAssetDir(std::string dir) {
    if (dir.empty() || dir.front() != '/')
        throw makeError("asset path must start with slash: " + dir);
    if (fs::exists(dir))
        return dir;
    if (auto edir = g_assetRoot + dir; fs::exists(edir))
        return edir;
    throw makeError("cannot find asset directory: " + dir);
}

}
