#ifndef __CORE_ASSETS_H__
#define __CORE_ASSETS_H__

#include <zeno/core/Graph.h>

namespace zeno {

struct Asset {
    std::shared_ptr<Graph> sharedGraph;
    std::vector<ParamInfo> inputs;
    std::vector<ParamInfo> outputs;
};

struct AssetsMgr : std::enable_shared_from_this<AssetsMgr> {
    Session *session = nullptr;

    std::map<std::string, Asset> m_assets;

    ZENO_API AssetsMgr();
    ZENO_API ~AssetsMgr();

    AssetsMgr(AssetsMgr const&) = delete;
    AssetsMgr& operator=(AssetsMgr const&) = delete;
    AssetsMgr(AssetsMgr&&) = delete;
    AssetsMgr& operator=(AssetsMgr&&) = delete;

    ZENO_API void createAsset(const std::string& name);
    CALLBACK_REGIST(createAsset, void, const std::string&)

    ZENO_API void removeAsset(const std::string& name);
    CALLBACK_REGIST(removeAsset, void, const std::string&)

    ZENO_API void renameAsset(const std::string& old_name, const std::string& new_name);
    CALLBACK_REGIST(renameAsset, void, const std::string&, const std::string&)

    ZENO_API Asset getAsset(const std::string& name) const;
    ZENO_API void updateAssets(const std::string name, ParamsUpdateInfo info);
    ZENO_API std::shared_ptr<INode> newInstance(const std::string& assetsName);
};

}

#endif