#include <zeno/io/zsgreader.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/io/iotags.h>
#include <fstream>
#include <filesystem>
#include <zeno/io/iohelper.h>
#include <zeno/utils/helper.h>


using namespace zeno::iotags;
using namespace zeno::iotags::curve;

namespace zenoio {

    ZsgReader::ZsgReader() : m_bDiskReading(true), m_ioVer(zeno::VER_3) {}

    ZENO_API ZSG_PARSE_RESULT ZsgReader::openFile(const std::string& fn)
    {
        ZSG_PARSE_RESULT result;
        result.code = PARSE_ERROR;

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
        FILE* fp = fopen(filePath.string().c_str(), "rb");
        if (!fp) {
            zeno::log_error("zsg file does not exist");
            return result;
        }

        size_t actualSz = fread(&dat[0], 1, szBuffer, fp);
        if (actualSz != szBuffer) {
            zeno::log_warn("the bytes read from file is different from the size of whole file");
        }
        fclose(fp);
        fp = nullptr;

        doc.Parse(&dat[0], actualSz);

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
        result.code = PARSE_NOERROR;
        return result;
    }

    bool ZsgReader::_parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret) {
        return false;
    }

    void ZsgReader::_parseSocket(
        const bool bInput,
        const bool bSubnetNode,
        const std::string& id,
        const std::string& nodeCls,
        const std::string& inSock,
        const rapidjson::Value& sockObj,
        zeno::NodeData& ret,
        zeno::LinksData& links)
    {
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
                zeno::ParamObject param;
                param.name = inSock;
                //归为对象吧
                ret.customUi.inputObjs.push_back(param);
            }
            else if (inputObj.IsObject())
            {
                bool bSubnet = ret.cls == "Subnet";
                _parseSocket(true, bSubnet, id, nodeName, inSock, inputObj, ret, links);
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
                zeno::ParamObject param;
                param.name = outParam;
                param.socketType = zeno::Socket_Output;
                ret.customUi.outputObjs.push_back(param);
            }
            else if (outObj.IsObject())
            {
                _parseSocket(false, false, id, nodeName, outParam, outObj, ret, links);
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
                                zeno::ParamPrimitive param;
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
                        zeno::NodeData node;
                        _parseSocket(true, false, "", nodeCls, socketName, input.value, node, lnks);
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
                                zeno::ParamPrimitive param;
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
                        zeno::NodeData node;
                        _parseSocket(true, false, "", nodeCls, socketName, param.value, node, lnks);
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
                                zeno::ParamPrimitive param;
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
                        zeno::NodeData node;
                        _parseSocket("", false, nodeCls, socketName, false, output.value, node, lnks);
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