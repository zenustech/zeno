#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <rapidjson/document.h>
#include <zenoio/include/iocommon.h>
#include <common/data.h>


namespace zenoio
{
    class Zsg2Reader
    {
    public:
        static Zsg2Reader& getInstance();
        bool openFile(const std::string& fn, zeno::ZSG_PARSE_RESULT& ret);

    private:
        Zsg2Reader();
        bool _parseSubGraph(
                const std::string& graphPath,   //例如 "/main"  "/main/aaa"
                const rapidjson::Value &subgraph,
                const zeno::AssetsData& subgraphDatas,
                zeno::GraphData& subgData);

        zeno::NodeData _parseNode(
                const std::string& subgPath,    //也许无用了，因为边信息不再以path的方式储存（解析麻烦），先保留着
                const std::string& nodeid,
                const rapidjson::Value& nodeObj,
                const zeno::AssetsData& subgraphDatas,
                zeno::LinksData& links);    //在parse节点的时候顺带把节点上的边信息也逐个记录到这里

        zeno::ParamInfo _parseSocket(
                const std::string& subgPath,
                const std::string& id,
                const std::string& nodeCls,
                const std::string& inSock,
                bool bInput,
                const rapidjson::Value& sockObj,
                zeno::LinksData& links);

        void _parseInputs(
                const std::string& subgPath,
                const std::string& id,
                const std::string& nodeName,
                const rapidjson::Value& inputs,
                zeno::NodeData& ret,
                zeno::LinksData& links);

        bool _parseParams(
                const std::string& id,
                const std::string& nodeCls,
                const rapidjson::Value& jsonParams,
                zeno::NodeData& ret);

        void _parseOutputs(
                const std::string& id,
                const std::string& nodeName,
                const rapidjson::Value& jsonParams,
                zeno::NodeData& ret,
                zeno::LinksData& links);

        void _parseCustomPanel(
                const std::string& id,
                const std::string& nodeName,
                const rapidjson::Value& jsonCutomUI,
                zeno::NodeData& ret);

        void _parseDictPanel(
                const std::string& subgPath,
                bool bInput,
                const rapidjson::Value& dictPanelObj,
                const std::string& id,
                const std::string& inSock,
                const std::string& nodeName,
                zeno::LinksData& links);

        void _parseViews(
                const rapidjson::Value& jsonViews,
                zeno::ZSG_PARSE_RESULT& res);

        void _parseTimeline(
                const rapidjson::Value& jsonTimeline,
                zeno::ZSG_PARSE_RESULT& res);

        void _parseChildNodes(
                const std::string& id,
                const rapidjson::Value& jsonNodes,
                const zeno::NodeDescs& descriptors,
                zeno::NodeData& ret);

        zeno::NodeDescs _parseDescs(const rapidjson::Value& descs);

        zeno::NodesData _parseChildren(const rapidjson::Value& jsonNodes);

        zeno::ZSG_VERSION m_ioVer;
        bool m_bDiskReading;        //disk io read.
    };
}

#endif
