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
                const zeno::NodeDescs& descriptors,
                const zeno::AssetsData& subgraphDatas,
                zeno::GraphData& subgData);

        bool _parseNode(
                const std::string& subgPath,    //也许无用了，因为边信息不再以path的方式储存（解析麻烦），先保留着
                const std::string& nodeid,
                const rapidjson::Value& nodeObj,
                const zeno::NodeDescs& descriptors,
                const zeno::AssetsData& subgraphDatas,
                zeno::NodeData& retNode,
                zeno::LinksData& links);    //在parse节点的时候顺带把节点上的边信息也逐个记录到这里

        void _parseSocket(
                const std::string& subgPath,
                const std::string& id,
                const std::string& nodeCls,
                const std::string& inSock,
                bool bInput,
                const rapidjson::Value& sockObj,
                const zeno::NodeDescs& descriptors,
                zeno::NodeData& ret,
                zeno::LinksData& links);

        void _parseInputs(
                const std::string& subgPath,
                const std::string& id,
                const std::string& nodeName,
                const zeno::NodeDescs& descriptors,
                const rapidjson::Value& inputs,
                zeno::NodeData& ret,
                zeno::LinksData& links);

        void _parseParams(
                const std::string& id,
                const std::string& nodeName,
                const rapidjson::Value &jsonParams,
                const zeno::NodeDescs& legacyDescs,
                zeno::NodeData& ret);

        bool _parseParams2(
                const std::string& id,
                const std::string& nodeCls,
                const rapidjson::Value& jsonParams,
                zeno::NodeData& ret);

        void _parseOutputs(
                const std::string& id,
                const std::string& nodeName,
                const rapidjson::Value& jsonParams,
                zeno::NodeData& ret);

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
                zeno::NodeData& ret,
                zeno::LinksData& links);

        void _parseColorRamps(
                const std::string& id,
                const rapidjson::Value& jsonColorRamps,
                zeno::NodeData& ret);

        void _parseLegacyCurves(
                const std::string &id,
                const rapidjson::Value &jsonPoints,
                const rapidjson::Value &jsonHandlers,
                zeno::NodeData& ret);

        void _parseViews(
                const rapidjson::Value& jsonViews,
                zeno::ZSG_PARSE_RESULT& res);

        void _parseTimeline(
                const rapidjson::Value& jsonTimeline,
                zeno::ZSG_PARSE_RESULT& res);

        void _parseBySocketKeys(
                const std::string& id,
                const rapidjson::Value& objValue,
                zeno::NodeData& ret);

        void _parseDictKeys(
                const std::string& id,
                const rapidjson::Value& objValue,
                zeno::NodeData& ret);

        void _parseChildNodes(
                const std::string& id,
                const rapidjson::Value& jsonNodes,
                const zeno::NodeDescs& descriptors,
                zeno::NodeData& ret);

        zeno::NodeDescs _parseDescs(const rapidjson::Value& descs);

        zeno::NodesData _parseChildren(const rapidjson::Value& jsonNodes);

        void initSockets(
                const std::string& name,
                const zeno::NodeDescs& legacyDescs,
                zeno::NodeData& ret);

        zeno::zvariant _parseDeflValue(
                        const std::string &nodeCls,
                        const zeno::NodeDescs& legacyDescs,
                        const std::string& sockName,
                        bool bInput,
                        const rapidjson::Value &defaultValue);

        zeno::ZSG_VERSION m_ioVer;
        bool m_bDiskReading;        //disk io read.
    };
}

#endif
