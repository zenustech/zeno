#include "zenreader.h"
#include <zenoio/include/iohelper.h>
#include <zeno/utils/helper.h>


namespace zenoio
{
    ZenReader::ZenReader()
    {
    }

    bool ZenReader::_parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& ret)
    {
        zeno::AssetsData assets;        //todo
        if (doc.HasMember("main"))
        {
            const rapidjson::Value& mainGraph = doc["main"];
            if (_parseGraph(mainGraph, assets, ret))
            {
                ret.name = "main";
                ret.type = zeno::Subnet_Main;
                return true;
            }
        }
        return false;
    }

    bool ZenReader::_parseGraph(const rapidjson::Value& graph, const zeno::AssetsData& assets, zeno::GraphData& ret)
    {
        if (!graph.IsObject() || !graph.HasMember("nodes"))
            return false;

        const auto& nodes = graph["nodes"];
        if (nodes.IsNull())
            return false;

        for (const auto& node : nodes.GetObject())
        {
            const std::string& nodeid = node.name.GetString();
            const zeno::NodeData& nodeData = _parseNode("", nodeid, node.value, assets, ret.links);
            ret.nodes.insert(std::make_pair(nodeid, nodeData));
        }
        return true;
    }

    zeno::NodeData ZenReader::_parseNode(
        const std::string& subgPath,    //也许无用了，因为边信息不再以path的方式储存（解析麻烦），先保留着
        const std::string& nodeid,
        const rapidjson::Value& nodeObj,
        const zeno::AssetsData& assets,
        zeno::LinksData& links)
    {
        zeno::NodeData retNode;

        const auto& objValue = nodeObj;
        const rapidjson::Value& nameValue = objValue["name"];
        const std::string& cls = objValue["class"].GetString();

        retNode.name = nodeid;
        retNode.cls = cls;
        retNode.type = zeno::Node_Normal;

        if (objValue.HasMember("inputs"))
        {
            _parseInputs(nodeid, cls, objValue["inputs"], retNode, links);
        }
        if (objValue.HasMember("outputs"))
        {
            _parseOutputs(nodeid, cls, objValue["outputs"], retNode, links);
        }
        if (objValue.HasMember("customui-panel"))
        {
            //deprecated legacy customui.
        }

        if (objValue.HasMember("uipos"))
        {
            auto uipos = objValue["uipos"].GetArray();
            retNode.uipos = { uipos[0].GetFloat(), uipos[1].GetFloat() };
        }
        if (objValue.HasMember("status"))
        {
            auto optionsArr = objValue["status"].GetArray();
            zeno::NodeStatus opts = zeno::None;
            for (int i = 0; i < optionsArr.Size(); i++)
            {
                assert(optionsArr[i].IsString());

                const std::string& optName = optionsArr[i].GetString();
                if (optName == "VIEW")
                {
                    opts = opts | zeno::View;
                }
                else if (optName == "MUTE")
                {
                    opts = opts | zeno::Mute;
                }
            }
            retNode.status = opts;
        }
        if (objValue.HasMember("collasped"))
        {
            bool bCollasped = objValue["collasped"].GetBool();
            retNode.bCollasped = bCollasped;
        }

        if (cls == "Blackboard")
        {
            zeno::GroupInfo blackboard;
            //use subkey "blackboard" for zeno2 io, but still compatible with zeno1
            const rapidjson::Value& blackBoardValue = objValue.HasMember("blackboard") ? objValue["blackboard"] : objValue;

            if (blackBoardValue.HasMember("special")) {
                blackboard.special = blackBoardValue["special"].GetBool();
            }

            blackboard.title = blackBoardValue.HasMember("title") ? blackBoardValue["title"].GetString() : "";
            blackboard.content = blackBoardValue.HasMember("content") ? blackBoardValue["content"].GetString() : "";

            if (blackBoardValue.HasMember("width") && blackBoardValue.HasMember("height")) {
                float w = blackBoardValue["width"].GetFloat();
                float h = blackBoardValue["height"].GetFloat();
                blackboard.sz = { w,h };
            }
            if (blackBoardValue.HasMember("params")) {
                //todo
            }
            //TODO: import blackboard.
        }
        else if (cls == "Group")
        {
            zeno::GroupInfo group;
            const rapidjson::Value& blackBoardValue = objValue.HasMember("blackboard") ? objValue["blackboard"] : objValue;

            group.title = blackBoardValue.HasMember("title") ? blackBoardValue["title"].GetString() : "";
            group.background = blackBoardValue.HasMember("background") ? blackBoardValue["background"].GetString() : "#3C4645";

            if (blackBoardValue.HasMember("width") && blackBoardValue.HasMember("height")) {
                float w = blackBoardValue["width"].GetFloat();
                float h = blackBoardValue["height"].GetFloat();
                group.sz = { w,h };
            }
            if (blackBoardValue.HasMember("items")) {
                auto item_keys = blackBoardValue["items"].GetArray();
                for (int i = 0; i < item_keys.Size(); i++) {
                    std::string key = item_keys[i].GetString();
                    group.items.push_back(key);
                }
            }
            //TODO: import group.
        }

        if (objValue.HasMember("subnet")) {
            zeno::GraphData subgraph;
            _parseGraph(objValue["subnet"], assets, subgraph);
            retNode.subgraph = subgraph;
            retNode.type = zeno::Node_SubgraphNode;
        }

        if (objValue.HasMember("asset")) {
            zeno::AssetInfo info;
            auto& assetObj = objValue["asset"];
            if (assetObj.HasMember("name") && assetObj.HasMember("version"))
            {
                info.name = assetObj["name"].GetString();
                std::string verStr = assetObj["version"].GetString();
                std::vector<std::string> vec = zeno::split_str(verStr.c_str(), '.');
                if (vec.size() == 1)
                {
                    info.majorVer = std::stoi(vec[0]);
                }
                else if (vec.size() == 2)
                {
                    info.majorVer = std::stoi(vec[0]);
                    info.minorVer = std::stoi(vec[1]);
                }
            }
            retNode.type = zeno::Node_AssetInstance;
            retNode.asset = info;
        }

        return retNode;
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

        std::string sockProp;
        if (sockObj.HasMember("property"))
        {
            //ZASSERT_EXIT(sockObj["property"].IsString());
            sockProp = sockObj["property"].GetString();
        }

        zeno::ParamControl ctrl = zeno::NullControl;

        zeno::SocketProperty prop = zeno::SocketProperty::Socket_Normal;
        if (sockProp == "dict-panel")
            prop = zeno::SocketProperty::Socket_Normal;    //deprecated
        else if (sockProp == "editable")
            prop = zeno::SocketProperty::Socket_Editable;
        else if (sockProp == "group-line")
            prop = zeno::SocketProperty::Socket_Normal;    //deprecated

        param.prop = prop;
        param.name = inSock;

        if (m_bDiskReading &&
            (prop == zeno::SocketProperty::Socket_Editable ||
                nodeCls == "MakeList" || nodeCls == "MakeDict" || nodeCls == "ExtractDict"))
        {
            if (prop == zeno::SocketProperty::Socket_Editable) {
                //like extract dict.
                param.type = zeno::Param_String;
            }
            else {
                param.type = zeno::Param_Null;
            }
        }

        if (sockObj.HasMember("type")) {
            param.type = zeno::convertToType(sockObj["type"].GetString());
        }

        if (sockObj.HasMember("default-value")) {
            param.defl = zenoio::jsonValueToZVar(sockObj["default-value"], param.type);
        }

        //link:
        if (bInput && sockObj.HasMember("links") && sockObj["links"].IsArray())
        {
            auto& arr = sockObj["links"].GetArray();
            for (int i = 0; i < arr.Size(); i++) {
                auto& linkObj = arr[i];
                const std::string outnode = linkObj["out-node"].GetString();
                const std::string outsock = linkObj["out-socket"].GetString();
                const std::string outkey = linkObj["out-key"].GetString();
                const std::string innode = id;
                const std::string insock = inSock;
                const std::string inkey = linkObj["in-key"].GetString();

                zeno::EdgeInfo link = { outnode, outsock, outkey, innode, insock, inkey };
                param.links.push_back(link);
                links.push_back(link);
            }
        }

        if (sockObj.HasMember("control"))
        {
            zeno::ParamControl ctrl = zeno::NullControl;
            zeno::ControlProperty props;
            bool bret = zenoio::importControl(sockObj["control"], ctrl, props);
            if (bret) {
                param.control = ctrl;
                if (ctrl == zeno::NullControl)
                    param.control = zeno::getDefaultControl(param.type);
                if (props.items || props.ranges)
                    param.ctrlProps = props;
            }
        }

        if (sockObj.HasMember("tooltip"))
        {
            param.tooltip = sockObj["tooltip"].GetString();
        }
        return param;
    }
}