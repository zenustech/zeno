#ifndef __ZSG2_READER_H__
#define __ZSG2_READER_H__

#include "zsgreader.h"

namespace zenoio
{
    class Zsg2Reader : public ZsgReader 
    {
    public:
        Zsg2Reader();

    protected:
        bool _parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret) override;

        zeno::NodeData _parseNode(
            const std::string& subgPath,    //也许无用了，因为边信息不再以path的方式储存（解析麻烦），先保留着
            const std::string& nodeid,
            const rapidjson::Value& nodeObj,
            const std::map<std::string, zeno::GraphData>& subgraphDatas,
            zeno::LinksData& links) override;    //在parse节点的时候顺带把节点上的边信息也逐个记录到这里

        zeno::ParamInfo _parseSocket(
            const std::string& id,
            const std::string& nodeCls,
            const std::string& inSock,
            bool bInput,
            const rapidjson::Value& sockObj,
            zeno::LinksData& links) override;

    private:
        bool _parseSubGraph(
                const std::string& graphPath,   //例如 "/main"  "/main/aaa"
                const rapidjson::Value &subgraph,
                const std::map<std::string, zeno::GraphData>& subgraphDatas,
                zeno::GraphData& subgData);

        bool _parseParams(
                const std::string& id,
                const std::string& nodeCls,
                const rapidjson::Value& jsonParams,
                zeno::NodeData& ret);

        void _parseDictPanel(
                bool bInput,
                const rapidjson::Value& dictPanelObj,
                const std::string& id,
                const std::string& inSock,
                const std::string& nodeName,
                zeno::LinksData& links);
    };
}

#endif
