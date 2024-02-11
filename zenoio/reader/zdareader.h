#ifndef __ZENO_DIGTAL_ASSET_READER_H__
#define __ZENO_DIGTAL_ASSET_READER_H__

#include "zenreader.h"
#include <zeno/core/data.h>

namespace zenoio
{
    class ZdaReader : public ZenReader
    {
    public:
        ZdaReader();
        zeno::ZenoAsset getParsedAsset() const;

    protected:
        bool _parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret) override;

    private:
        void _parseParams(
            const rapidjson::Value& paramsObj,
            std::vector<zeno::ParamInfo>& inputs,
            std::vector<zeno::ParamInfo>& outputs
        );

        zeno::ZenoAsset m_asset;
    };
}

#endif