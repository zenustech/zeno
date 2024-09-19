#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>

namespace zeno {

struct SubnetNode : INode {
    std::shared_ptr<Graph> subgraph;

    CustomUI m_customUi;

    ZENO_API SubnetNode();
    ZENO_API ~SubnetNode();

    ZENO_API void initParams(const NodeData& dat) override;
    ZENO_API params_change_info update_editparams(const ParamsUpdateInfo& params) override;
    ZENO_API std::shared_ptr<Graph> get_graph() const;
    ZENO_API bool isAssetsNode() const;
    ZENO_API void apply() override;
    ZENO_API NodeData exportInfo() const override;

    ZENO_API CustomUI get_customui() const override;
    ZENO_API void setCustomUi(const CustomUI& ui);
    void mark_subnetdirty(bool bOn);
};

struct DopNetwork : zeno::SubnetNode {

    DopNetwork();
    ZENO_API void apply() override;

    ZENO_API void setEnableCache(bool enable);
    ZENO_API void setAllowCacheToDisk(bool enable);
    ZENO_API void setMaxCacheMemoryMB(int size);
    ZENO_API void setCurrCacheMemoryMB(int size);
    static size_t getObjSize(std::shared_ptr<IObject> obj);
    void resetFrameState();

    CALLBACK_REGIST(dopnetworkFrameRemoved, void, int)
    CALLBACK_REGIST(dopnetworkFrameCached, void, int)

    bool m_bEnableCache;
    bool m_bAllowCacheToDisk;
    int m_maxCacheMemoryMB;
    int m_currCacheMemoryMB;

    std::map<int, std::map<std::string, std::shared_ptr<zeno::IObject>>> m_frameCaches;
    std::map<int, size_t> m_frameCacheSizes;
    size_t m_totalCacheSizeByte;
};

}
