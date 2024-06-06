#include <zeno/io/zdareader.h>
#include <zeno/utils/string.h>
#include <zeno/io/iotags.h>
#include <zeno/utils/helper.h>


namespace zenoio
{
    ZENO_API ZdaReader::ZdaReader() : m_bDelayReadGraphData(false) {

    }

    ZENO_API void ZdaReader::setDelayReadGraph(bool bDelay) {
        m_bDelayReadGraphData = bDelay;
    }

    bool ZdaReader::_parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret) {
        if (!doc.HasMember("name") ||
            !doc.HasMember("version") ||
            !doc.HasMember("graph")/* ||
            !doc.HasMember("Parameters")*/)
        {
            return false;
        }

        if (!doc["name"].IsString() ||
            !doc["version"].IsString() ||
            !doc["graph"].IsObject()/* ||
            !doc["Parameters"].IsObject()*/)
        {
            return false;
        }

        m_asset.info.name = doc["name"].GetString();
        const std::string& ver = doc["version"].GetString();
        std::vector<std::string> vervec = zeno::split_str(ver, '.');
        if (vervec.size() == 1) {
            m_asset.info.majorVer = std::stoi(vervec[0]);
        }
        else if (vervec.size() == 2) {
            m_asset.info.majorVer = std::stoi(vervec[0]);
            m_asset.info.minorVer = std::stoi(vervec[1]);
        }

        zeno::AssetsData assets;
        if (!m_bDelayReadGraphData)
        {
            if (!_parseGraph(doc["graph"], assets, ret))
                return false;
        }

        //zeno::NodeData tmp;
        //_parseParams(doc["Parameters"], tmp);

        if (doc.HasMember("subnet-customUi"))
        {
            zeno::NodeData tmp;
            _parseParams(doc["subnet-customUi"], tmp);
            m_asset.primitive_inputs = customUiToParams(tmp.customUi.inputPrims);
            m_asset.object_inputs = tmp.customUi.inputObjs;
            m_asset.primitive_outputs = tmp.customUi.outputPrims;
            m_asset.object_outputs = tmp.customUi.outputObjs;
            m_asset.m_customui = tmp.customUi;
        }

        ret.type = zeno::Subnet_Normal;
        ret.name = m_asset.info.name;
        if (!m_bDelayReadGraphData)
            m_asset.optGraph = ret;
        return true;
    }

    ZENO_API zeno::ZenoAsset ZdaReader::getParsedAsset() const
    {
        return m_asset;
    }

    void ZdaReader::_parseParams(const rapidjson::Value& paramsObj, zeno::NodeData& ret)
    {
        ret.customUi = _parseCustomUI(paramsObj);
        if (paramsObj.HasMember(iotags::params::node_inputs_objs))
        {
            for (const auto& inObj : paramsObj[iotags::params::node_inputs_objs].GetObject())
            {
                const std::string& inSock = inObj.name.GetString();
                const auto& inputObj = inObj.value;

                if (inputObj.IsNull())
                {
                    zeno::ParamObject param;
                    param.name = inSock;
                    ret.customUi.inputObjs.push_back(param);
                }
                else if (inputObj.IsObject())
                {
                    zeno::LinksData links;
                    _parseSocket(true, true, true, "", "", inSock, inputObj, ret, links);
                }
                else
                {
                    //TODO
                }
            }
        }
        if (paramsObj.HasMember(iotags::params::node_outputs_primitive))
        {
            for (const auto& outObj : paramsObj[iotags::params::node_outputs_primitive].GetObject())
            {
                const std::string& outSock = outObj.name.GetString();
                const auto& outputObj = outObj.value;
                if (outputObj.IsNull())
                {
                }
                else if (outputObj.IsObject())
                {
                    zeno::LinksData links;
                    _parseSocket(false, true, false, "", "", outSock, outputObj, ret, links);
                }
                else
                {
                }
            }
        }
        if (paramsObj.HasMember(iotags::params::node_outputs_objs))
        {
            for (const auto& outObj : paramsObj[iotags::params::node_outputs_objs].GetObject())
            {
                const std::string& outSock = outObj.name.GetString();
                const auto& outputObj = outObj.value;
                if (outputObj.IsNull())
                {
                    zeno::ParamObject param;
                    param.name = outSock;
                    ret.customUi.outputObjs.push_back(param);
                }
                else if (outputObj.IsObject())
                {
                    zeno::LinksData links;
                    _parseSocket(false, true, true, "", "", outSock, outputObj, ret, links);
                }
                else
                {
                }
            }
        }
    }
}