#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <rapidjson/document.h>
#include <zeno/io/iocommon.h>
#include <zeno/core/data.h>

namespace zenoio
{
    class ZsgReader
    {
    public:
        ZsgReader();
        virtual ~ZsgReader() = default;
        ZENO_API ZSG_PARSE_RESULT openFile(const std::wstring& fn);
        ZENO_API int numOfNodes() const;

    protected:
        virtual bool _parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret);

        virtual void _parseSocket(
            const bool bInput,
            const bool bSubnetNode,
            const std::string& id,
            const std::string& nodeCls,
            const std::string& inSock,
            const rapidjson::Value& sockObj,
            zeno::NodeData& ret,
            zeno::LinksData& links);

        virtual void _parseInputs(
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

        float total_time;
        float used_time;
        int m_num_of_nodes;

    private:
        zeno::TimelineInfo _parseTimeline(const rapidjson::Value& jsonTimeline);
        zeno::NodeDescs _parseDescs(const rapidjson::Value& descs);
    };
}


#endif