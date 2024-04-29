#ifndef __CORE_ASSETS_H__
#define __CORE_ASSETS_H__

#include <zeno/core/Graph.h>

namespace zeno {

struct Asset {
    AssetInfo m_info;
    std::shared_ptr<Graph> sharedGraph;
    std::vector<ParamInfo> inputs;
    std::vector<ParamInfo> outputs;
    CustomUI m_customui;
};

struct ZenoAsset {
    std::vector<ParamInfo> inputs;
    std::vector<ParamInfo> outputs;
    AssetInfo info;
    CustomUI m_customui;
    std::optional<GraphData> optGraph;
};

using AssetsData = std::map<std::string, ZenoAsset>;

struct AssetsMgr : std::enable_shared_from_this<AssetsMgr> {
    Session *session = nullptr;

    std::map<std::string, Asset> m_assets;

    ZENO_API AssetsMgr();
    ZENO_API ~AssetsMgr();

    AssetsMgr(AssetsMgr const&) = delete;
    AssetsMgr& operator=(AssetsMgr const&) = delete;
    AssetsMgr(AssetsMgr&&) = delete;
    AssetsMgr& operator=(AssetsMgr&&) = delete;

    ZENO_API void createAsset(const zeno::ZenoAsset asset);
    CALLBACK_REGIST(createAsset, void, zeno::AssetInfo)

    ZENO_API void removeAsset(const std::string& name);
    CALLBACK_REGIST(removeAsset, void, const std::string&)

    ZENO_API void renameAsset(const std::string& old_name, const std::string& new_name);
    CALLBACK_REGIST(renameAsset, void, const std::string&, const std::string&)

    ZENO_API Asset getAsset(const std::string& name) const;
    ZENO_API std::shared_ptr<Graph> getAssetGraph(const std::string& name, bool bLoadIfNotExist);
    ZENO_API std::vector<Asset> getAssets() const;
    ZENO_API void updateAssets(const std::string name, ParamsUpdateInfo info, const zeno::CustomUI& customui);
    ZENO_API std::shared_ptr<INode> newInstance(Graph* pGraph, const std::string& assetsName, const std::string& nodeName, bool createInAsset);
    ZENO_API void updateAssetInstance(const std::string& assetsName, std::shared_ptr<SubnetNode> &spNode);


    ZENO_API bool isAssetGraph(std::shared_ptr<Graph> spGraph) const;
    ZENO_API bool generateAssetName(std::string& name);

private:
    void initAssetsInfo();
    std::shared_ptr<Graph> forkAssetGraph(std::shared_ptr<Graph> assetGraph, std::shared_ptr<SubnetNode> subNode);
    bool m_bInitAssetInfo = false;
};

}

#endif