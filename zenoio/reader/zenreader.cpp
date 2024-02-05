#include "zenreader.h"


namespace zenoio
{
    ZenReader::ZenReader()
    {
    }

    bool ZenReader::_parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret)
    {
        return false;
    }

    zeno::NodeData ZenReader::_parseNode(
        const std::string& subgPath,    //也许无用了，因为边信息不再以path的方式储存（解析麻烦），先保留着
        const std::string& nodeid,
        const rapidjson::Value& nodeObj,
        const std::map<std::string, zeno::GraphData>& subgraphDatas,
        zeno::LinksData& links)
    {
        zeno::NodeData node;
        return node;
    }

    zeno::ParamInfo ZenReader::_parseSocket(
        const bool bInput,
        const std::string& id,
        const std::string& nodeCls,
        const std::string& inSock,
        const rapidjson::Value& sockObj,
        zeno::LinksData& links)
    {
        zeno::ParamInfo param;
        return param;
    }
}