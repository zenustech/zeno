#include "zsgreader.h"
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "include/iotags.h"
#include <fstream>
#include <filesystem>
#include <zenoio/include/iohelper.h>
#include <zeno/utils/helper.h>


using namespace zeno::iotags;
using namespace zeno::iotags::curve;

namespace zenoio {

    ZsgReader::ZsgReader() : m_bDiskReading(true), m_ioVer(zeno::VER_3) {}

    ZSG_PARSE_RESULT ZsgReader::openFile(const std::string& fn)
    {
        ZSG_PARSE_RESULT result;
        result.bSucceed = false;

        std::filesystem::path filePath(fn);
        if (!std::filesystem::exists(filePath)) {
            zeno::log_error("cannot open zsg file: {} ({})", fn);
            return result;
        }

        rapidjson::Document doc;

        auto szBuffer = std::filesystem::file_size(filePath);
        if (szBuffer == 0)
        {
            zeno::log_error("the zsg file is a empty file");
            return result;
        }

        std::vector<char> dat(szBuffer);
        FILE* fp = fopen(filePath.string().c_str(), "r");
        if (!fp) {
            zeno::log_error("zsg file does not exist");
            return result;
        }

        size_t ret = fread(&dat[0], 1, szBuffer, fp);
        assert(ret == szBuffer);
        fclose(fp);
        fp = nullptr;

        doc.Parse(&dat[0], dat.size());

        if (!doc.IsObject())
        {
            zeno::log_error("zsg json file is corrupted");
            return result;
        }

        if (!_parseMainGraph(doc, result.mainGraph))
            return result;

        if (doc.HasMember("views"))
        {
            _parseViews(doc["views"], result);
        }
        
        result.iover = m_ioVer;
        result.bSucceed = true;
        return result;
    }

    bool ZsgReader::_parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret) {
        return false;
    }

    zeno::NodeData ZsgReader::_parseNode(
        const std::string& subgPath,    //也许无用了，因为边信息不再以path的方式储存（解析麻烦），先保留着
        const std::string& nodeid,
        const rapidjson::Value& nodeObj,
        const std::map<std::string, zeno::GraphData>& subgraphDatas,
        zeno::LinksData& links)    //在parse节点的时候顺带把节点上的边信息也逐个记录到这里
    {
        zeno::NodeData dat;
        return dat;
    }

    zeno::ParamInfo ZsgReader::_parseSocket(
        const bool bInput,
        const std::string& id,
        const std::string& nodeCls,
        const std::string& inSock,
        const rapidjson::Value& sockObj,
        zeno::LinksData& links)
    {
        zeno::ParamInfo info;
        return info;
    }

    void ZsgReader::_parseViews(const rapidjson::Value& jsonViews, zenoio::ZSG_PARSE_RESULT& res)
    {
        if (jsonViews.HasMember("timeline"))
        {
            res.timeline = _parseTimeline(jsonViews["timeline"]);
        }
    }

    zeno::TimelineInfo ZsgReader::_parseTimeline(const rapidjson::Value& jsonTimeline)
    {
        zeno::TimelineInfo timeinfo;
        assert(jsonTimeline.HasMember(timeline::start_frame) && jsonTimeline[timeline::start_frame].IsInt());
        assert(jsonTimeline.HasMember(timeline::end_frame) && jsonTimeline[timeline::end_frame].IsInt());
        assert(jsonTimeline.HasMember(timeline::curr_frame) && jsonTimeline[timeline::curr_frame].IsInt());
        assert(jsonTimeline.HasMember(timeline::always) && jsonTimeline[timeline::always].IsBool());

        timeinfo.beginFrame = jsonTimeline[timeline::start_frame].GetInt();
        timeinfo.endFrame = jsonTimeline[timeline::end_frame].GetInt();
        timeinfo.currFrame = jsonTimeline[timeline::curr_frame].GetInt();
        timeinfo.bAlways = jsonTimeline[timeline::always].GetBool();
        return timeinfo;
    }

    void ZsgReader::_parseInputs(
        const std::string& id,
        const std::string& nodeName,
        const rapidjson::Value& inputs,
        zeno::NodeData& ret,
        zeno::LinksData& links)
    {
        for (const auto& inObj : inputs.GetObject())
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
                ret.inputs.push_back(param);
            }
            else if (inputObj.IsObject())
            {
                zeno::ParamInfo param = _parseSocket(true, id, nodeName, inSock, inputObj, links);
                ret.inputs.push_back(param);
            }
            else
            {
                zeno::log_error("unknown format");
            }
        }
    }

    void ZsgReader::_parseOutputs(
        const std::string& id,
        const std::string& nodeName,
        const rapidjson::Value& outputs,
        zeno::NodeData& ret,
        zeno::LinksData& links)
    {
        for (const auto& outParamObj : outputs.GetObject())
        {
            const std::string& outParam = outParamObj.name.GetString();
            if (outParam == "DST")
                continue;

            const auto& outObj = outParamObj.value;
            if (outObj.IsNull())
            {
                zeno::ParamInfo param;
                param.name = outParam;
                ret.outputs.push_back(param);
            }
            else if (outObj.IsObject())
            {
                zeno::ParamInfo param = _parseSocket(true, id, nodeName, outParam, outObj, links);
                ret.outputs.push_back(param);
            }
            else
            {
                zeno::log_error("unknown format");
            }
        }
    }

    zeno::NodeDescs ZsgReader::_parseDescs(const rapidjson::Value& jsonDescs)
    {
        zeno::NodeDescs _descs;     //不需要系统内置节点的desc，只要读文件的就可以
        zeno::LinksData lnks;       //没用的
        for (const auto& node : jsonDescs.GetObject())
        {
            const std::string& nodeCls = node.name.GetString();
            const auto& objValue = node.value;

            zeno::NodeDesc desc;
            desc.name = nodeCls;
            if (objValue.HasMember("inputs"))
            {
                if (objValue["inputs"].IsArray())
                {
                    //系统节点导出的描述，形如：
                    /*
                    "inputs": [
                        [
                            "ListObject",
                            "keys",
                            ""
                        ],
                        [
                            "",
                            "SRC",
                            ""
                        ]
                    ],
                    */
                    auto inputs = objValue["inputs"].GetArray();
                    for (int i = 0; i < inputs.Size(); i++)
                    {
                        if (inputs[i].IsArray())
                        {
                            auto input_triple = inputs[i].GetArray();
                            std::string socketType, socketName, socketDefl;
                            if (input_triple.Size() > 0 && input_triple[0].IsString())
                                socketType = input_triple[0].GetString();
                            if (input_triple.Size() > 1 && input_triple[1].IsString())
                                socketName = input_triple[1].GetString();
                            if (input_triple.Size() > 2 && input_triple[2].IsString())
                                socketDefl = input_triple[2].GetString();

                            if (!socketName.empty())
                            {
                                zeno::ParamInfo param;
                                param.name = socketName;
                                param.type = zeno::convertToType(socketDefl);
                                param.defl = socketDefl;    //不转了，太麻烦了。..反正普通节点的desc也只是参考

                                desc.inputs.push_back(param);
                            }
                        }
                    }
                }
                else if (objValue["inputs"].IsObject())
                {
                    auto inputs = objValue["inputs"].GetObject();
                    for (const auto& input : inputs)
                    {
                        std::string socketName = input.name.GetString();
                        _parseSocket(true, "", nodeCls, socketName, input.value, lnks);
                    }
                }
            }
            if (objValue.HasMember("params"))
            {
                if (objValue["params"].IsArray())
                {
                    auto params = objValue["params"].GetArray();
                    for (int i = 0; i < params.Size(); i++)
                    {
                        if (params[i].IsArray()) {
                            auto param_triple = params[i].GetArray();
                            std::string socketType, socketName, socketDefl;

                            if (param_triple.Size() > 0 && param_triple[0].IsString())
                                socketType = param_triple[0].GetString();
                            if (param_triple.Size() > 1 && param_triple[1].IsString())
                                socketName = param_triple[1].GetString();
                            if (param_triple.Size() > 2 && param_triple[2].IsString())
                                socketDefl = param_triple[2].GetString();

                            if (!socketName.empty())
                            {
                                zeno::ParamInfo param;
                                param.name = socketName;
                                param.type = zeno::convertToType(socketDefl);
                                param.defl = socketDefl;    //不转了，太麻烦了。..反正普通节点的desc也只是参考
                                desc.inputs.push_back(param);
                            }
                        }
                    }
                }
                else if (objValue["params"].IsObject())
                {
                    auto params = objValue["params"].GetObject();
                    for (const auto& param : params)
                    {
                        std::string socketName = param.name.GetString();
                        _parseSocket(true, "", nodeCls, socketName, param.value, lnks);
                    }
                }
            }
            if (objValue.HasMember("outputs"))
            {
                if (objValue["outputs"].IsArray())
                {
                    auto outputs = objValue["outputs"].GetArray();
                    for (int i = 0; i < outputs.Size(); i++)
                    {
                        if (outputs[i].IsArray()) {
                            auto output_triple = outputs[i].GetArray();
                            std::string socketType, socketName, socketDefl;

                            if (output_triple.Size() > 0 && output_triple[0].IsString())
                                socketType = output_triple[0].GetString();
                            if (output_triple.Size() > 1 && output_triple[1].IsString())
                                socketName = output_triple[1].GetString();
                            if (output_triple.Size() > 2 && output_triple[2].IsString())
                                socketDefl = output_triple[2].GetString();

                            if (!socketName.empty())
                            {
                                zeno::ParamInfo param;
                                param.name = socketName;
                                param.type = zeno::convertToType(socketDefl);
                                param.defl = socketDefl;
                                desc.outputs.push_back(param);
                            }
                        }
                    }
                }
                else if (objValue["outputs"].IsObject())
                {
                    auto outputs = objValue["outputs"].GetObject();
                    for (const auto& output : outputs)
                    {
                        std::string socketName = output.name.GetString();
                        _parseSocket("", nodeCls, socketName, false, output.value, lnks);
                    }
                }
            }
            if (objValue.HasMember("categories") && objValue["categories"].IsArray())
            {
                auto categories = objValue["categories"].GetArray();
                for (int i = 0; i < categories.Size(); i++)
                {
                    desc.categories.push_back(categories[i].GetString());
                }
            }

            _descs.insert(std::make_pair(nodeCls, desc));
        }
        return _descs;
    }


}