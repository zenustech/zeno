#ifndef __CORE_ASSETS_H__
#define __CORE_ASSETS_H__

#include <zeno/core/Graph.h>

namespace zeno {

struct Asset {
    AssetInfo m_info;
    std::shared_ptr<Graph> sharedGraph;
    std::vector<ParamPrimitive> primitive_inputs;
    std::vector<ParamPrimitive> primitive_outputs;
    std::vector<ParamObject> object_inputs;
    std::vector<ParamObject> object_outputs;
    CustomUI m_customui;
};

struct AssetsMgr {
    Session *session = nullptr;

    std::map<std::string, Asset> m_assets;

    ZENO_API AssetsMgr();
    ZENO_API ~AssetsMgr();

    AssetsMgr(AssetsMgr const&) = delete;
    AssetsMgr& operator=(AssetsMgr const&) = delete;
    AssetsMgr(AssetsMgr&&) = delete;
    AssetsMgr& operator=(AssetsMgr&&) = delete;

    ZENO_API void createAsset(const zeno::ZenoAsset asset, bool isFirstCreate = false);
    CALLBACK_REGIST(createAsset, void, zeno::AssetInfo)

    ZENO_API void clear();
    CALLBACK_REGIST(clear, void)

    ZENO_API void removeAsset(const std::string& name);
    CALLBACK_REGIST(removeAsset, void, const std::string&)

    ZENO_API void renameAsset(const std::string& old_name, const std::string& new_name);
    CALLBACK_REGIST(renameAsset, void, const std::string&, const std::string&)

    ZENO_API Asset getAsset(const std::string& name) const;
    ZENO_API bool hasAsset(const std::string& name) const;
    ZENO_API std::shared_ptr<Graph> getAssetGraph(const std::string& name, bool bLoadIfNotExist);
    ZENO_API std::vector<Asset> getAssets() const;
    ZENO_API void updateAssets(const std::string& name, const ParamsUpdateInfo& info, const zeno::CustomUI& customui);
    ZENO_API std::unique_ptr<NodeImpl> newInstance(Graph* pGraph, const std::string& assetsName, const std::string& nodeName, bool createInAsset, bool bAssetLock);
    ZENO_API void updateAssetInstance(const std::string& assetsName, SubnetNode* spNode);
    ZENO_API void updateAssetInfo(const std::string& assetsName, std::shared_ptr<Graph> new_shared_gra, CustomUI customui);

    ZENO_API bool isAssetGraph(Graph* spGraph) const;
    ZENO_API bool generateAssetName(std::string& name);

    ZENO_API std::shared_ptr<Graph> forkAssetGraph(std::shared_ptr<Graph> assetGraph, NodeImpl* subNode);
    ZENO_API std::shared_ptr<Graph> syncInstToAssets(NodeImpl* assetInstNode);

private:
    void initAssetsInfo();
    void initAssetSubInputOutput(Asset& asset);
    bool m_bInitAssetInfo = false;
};

}

#endif