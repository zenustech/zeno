#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <rapidjson/document.h>
#include <zenoio/include/iocommon.h>
#include <zeno/core/data.h>

namespace zenoio
{
    class ZsgReader
    {
    public:
        ZsgReader();
        ZSG_PARSE_RESULT openFile(const std::string& fn);

    protected:
        virtual bool _parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret);

        virtual zeno::NodeData _parseNode(
            const std::string& subgPath,    //也许无用了，因为边信息不再以path的方式储存（解析麻烦），先保留着
            const std::string& nodeid,
            const rapidjson::Value& nodeObj,
            const std::map<std::string, zeno::GraphData>& subgraphDatas,
            zeno::LinksData& links);    //在parse节点的时候顺带把节点上的边信息也逐个记录到这里

        virtual zeno::ParamInfo _parseSocket(
            const std::string& id,
            const std::string& nodeCls,
            const std::string& inSock,
            bool bInput,
            const rapidjson::Value& sockObj,
            zeno::LinksData& links);

        void _parseInputs(
            const std::string& id,
            const std::string& nodeName,
            const rapidjson::Value& inputs,
            zeno::NodeData& ret,
            zeno::LinksData& links);

        void _parseOutputs(
            const std::string& id,
            const std::string& nodeName,
            const rapidjson::Value& jsonParams,
            zeno::NodeData& ret,
            zeno::LinksData& links);

        void _parseViews(
            const rapidjson::Value& jsonViews,
            zenoio::ZSG_PARSE_RESULT& res);

    protected:
        zeno::ZSG_VERSION m_ioVer;
        bool m_bDiskReading;        //disk io read.

    private:
        zeno::TimelineInfo _parseTimeline(const rapidjson::Value& jsonTimeline);

        zeno::NodeDescs _parseDescs(const rapidjson::Value& descs);
    };
}


#endif