#include <zeno/io/zdareader.h>
#include <zeno/utils/string.h>


namespace zenoio
{
    ZENO_API ZdaReader::ZdaReader() : m_bDelayReadGraphData(false) {

    }

    ZENO_API void ZdaReader::setDelayReadGraph(bool bDelay) {
        m_bDelayReadGraphData = bDelay;
    }

    bool ZdaReader::_parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret) {
        if (!doc.HasMember("name") || !doc.HasMember("version") || !doc.HasMember("graph") ||
            !doc.HasMember("Parameters"))
        {
            return false;
        }

        if (!doc["name"].IsString() ||
            !doc["version"].IsString() ||
            !doc["graph"].IsObject() ||
            !doc["Parameters"].IsObject())
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

        zeno::AssetsData assets;//todo
        if (!m_bDelayReadGraphData)
        {
            if (!_parseGraph(doc["graph"], assets, ret))
                return false;
        }

        _parseParams(doc["Parameters"], m_asset.inputs, m_asset.outputs);

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

    void ZdaReader::_parseParams(
        const rapidjson::Value& paramsObj,
        std::vector<zeno::ParamInfo>& inputs,
        std::vector<zeno::ParamInfo>& outputs)
    {
        if (paramsObj.HasMember("inputs"))
        {
            for (const auto& inObj : paramsObj["inputs"].GetObject())
            {
                const std::string& inSock = inObj.name.GetString();
                const auto& inputObj = inObj.value;

                if (inSock == "SRC") {
                    continue;
                }

                if (inputObj.IsNull())
                {
                    zeno::ParamInfo param;
                    param.name = inSock;
                    inputs.push_back(param);
                }
                else if (inputObj.IsObject())
                {
                    zeno::LinksData links;
                    zeno::ParamInfo param = _parseSocket(true, false, "", "", inSock, inputObj, links);
                    inputs.push_back(param);
                }
                else
                {
                }
            }
        }
        if (paramsObj.HasMember("outputs"))
        {
            for (const auto& outObj : paramsObj["outputs"].GetObject())
            {
                const std::string& outSock = outObj.name.GetString();
                const auto& outputObj = outObj.value;

                if (outSock == "SRC") {
                    continue;
                }

                if (outputObj.IsNull())
                {
                    zeno::ParamInfo param;
                    param.name = outSock;
                    outputs.push_back(param);
                }
                else if (outputObj.IsObject())
                {
                    zeno::LinksData links;
                    zeno::ParamInfo param = _parseSocket(false, false, "", "", outSock, outputObj, links);
                    outputs.push_back(param);
                }
                else
                {
                }
            }
        }
    }
}