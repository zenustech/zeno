#include <rapidjson/document.h>
#include "../include/iohelper.h"
#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/utils/string.h>


namespace zenoio
{
    static zeno::ParamControl getControlByName(const std::string& descName)
    {
        if (descName == "Integer")
        {
            return zeno::Lineedit;
        }
        else if (descName == "Float")
        {
            return zeno::Lineedit;
        }
        else if (descName == "String")
        {
            return zeno::Lineedit;
        }
        else if (descName == "Boolean")
        {
            return zeno::Checkbox;
        }
        else if (descName == "Multiline String")
        {
            return zeno::Multiline;
        }
        else if (descName == "read path")
        {
            return zeno::Pathedit;
        }
        else if (descName == "write path")
        {
            return zeno::Pathedit;
        }
        else if (descName == "Enum")
        {
            return zeno::Combobox;
        }
        else if (descName == "Float Vector 4")
        {
            return zeno::Vec4edit;
        }
        else if (descName == "Float Vector 3")
        {
            return zeno::Vec3edit;
        }
        else if (descName == "Float Vector 2")
        {
            return zeno::Vec2edit;
        }
        else if (descName == "Integer Vector 4")
        {
            return zeno::Vec4edit;
        }
        else if (descName == "Integer Vector 3")
        {
            return zeno::Vec3edit;
        }
        else if (descName == "Integer Vector 2")
        {
            return zeno::Vec2edit;
        }
        else if (descName == "Color")
        {
            return zeno::Heatmap;
        }
        else if (descName == "Pure Color")
        {
            return zeno::Color;
        }
        else if (descName == "Curve")
        {
            return zeno::CurveEditor;
        }
        else if (descName == "SpinBox")
        {
            return zeno::SpinBox;
        }
        else if (descName == "DoubleSpinBox")
        {
            return zeno::DoubleSpinBox;
        }
        else if (descName == "Slider")
        {
            return zeno::Slider;
        }
        else if (descName == "SpinBoxSlider")
        {
            return zeno::SpinBoxSlider;
        }
        else if (descName == "Dict Panel")
        {
            return zeno::NullControl;    //deprecated.
        }
        else if (descName == "group-line")
        {
            return zeno::NullControl;    //deprecated.
        }
        else
        {
            return zeno::NullControl;
        }
    }

    zeno::GraphData fork(
        const std::string& currentPath,
        const std::map<std::string, zeno::GraphData>& sharedSubg,
        const std::string& subnetName)
    {
        zeno::GraphData newGraph;
        zeno::NodesData newDatas;
        zeno::LinksData newLinks;

        std::map<std::string, zeno::NodeData> nodes;
        std::unordered_map<std::string, std::string> old2new;
        zeno::LinksData oldLinks;

        auto it = sharedSubg.find(subnetName);
        if (it == sharedSubg.end())
        {
            return newGraph;
        }

        const zeno::GraphData& subgraph = it->second;
        for (const auto& [name, nodeData] : subgraph.nodes)
        {
            zeno::NodeData nodeDat = nodeData;
            const std::string& snodeId = nodeDat.name;
            const std::string& name = nodeDat.cls;
            const std::string& newId = zeno::generateUUID();
            old2new.insert(std::make_pair(snodeId, newId));

            if (sharedSubg.find(name) != sharedSubg.end())
            {
                const std::string& ssubnetName = name;
                nodeDat.name = newId;

                zeno::LinksData childLinks;
                zeno::GraphData fork_subgraph;
                fork_subgraph = fork(
                    currentPath + "/" + newId,
                    sharedSubg,
                    ssubnetName);
                fork_subgraph.links = childLinks;
                nodeDat.subgraph = fork_subgraph;

                newDatas[newId] = nodeDat;
            }
            else
            {
                nodeDat.name = newId;
                newDatas[newId] = nodeDat;
            }
        }

        for (zeno::EdgeInfo oldLink : subgraph.links)
        {
            zeno::EdgeInfo newLink = oldLink;
            newLink.inNode = old2new[newLink.inNode];
            newLink.outNode = old2new[newLink.outNode];
            newLinks.push_back(newLink);
        }

        newGraph.nodes = nodes;
        newGraph.links = newLinks;
        return newGraph;
    }

    zeno::zvariant jsonValueToZVar(const rapidjson::Value& val, zeno::ParamType const& type)
    {
        zeno::zvariant defl;
        switch (type) {
        case zeno::Param_Int:
        {
            if (val.IsInt()) {
                defl = val.GetInt();
            }
            else if (val.IsFloat()) {
                defl = (int)val.GetFloat();
            }
            else if (val.IsDouble()) {
                defl = (int)val.GetFloat();
            }
            else if (val.IsString()) {
                defl = (int)std::stof(val.GetString());
            }
            else {
                zeno::log_error("error type");
            }
            break;
        }
        case zeno::Param_Float:
        {
            if (val.IsFloat())
                defl = val.GetFloat();
            else if (val.IsInt())
                defl = (float)val.GetInt();
            else if (val.IsDouble())
                defl = val.GetFloat();
            else if (val.IsString())
                defl = std::stof(val.GetString());
            else
                zeno::log_error("error type");
            break;
        }
        case zeno::Param_Bool:
        {
            if (val.IsBool())
                defl = (int)val.GetBool();
            else if (val.IsInt())
                defl = val.GetInt() != 0;
            else if (val.IsFloat())
                defl = val.GetFloat() != 0;
            else
                zeno::log_error("error type");
            break;
        }
        case zeno::Param_String:
        {
            if (val.IsString())
                defl = val.GetString();
            break;
        }
        case zeno::Param_Vec2i:
        case zeno::Param_Vec2f:
        case zeno::Param_Vec3i:
        case zeno::Param_Vec3f:
        case zeno::Param_Vec4i:
        case zeno::Param_Vec4f:
        {
            int dim = 0;
            bool bFloat = false;
            if (zeno::Param_Vec2i == type) dim = 2; bFloat = false;
            if (zeno::Param_Vec2f == type) dim = 2; bFloat = true;
            if (zeno::Param_Vec3i == type) dim = 3; bFloat = false;
            if (zeno::Param_Vec3f == type) dim = 3; bFloat = true;
            if (zeno::Param_Vec4i == type) dim = 4; bFloat = false;
            if (zeno::Param_Vec4f == type) dim = 4; bFloat = true;

            std::vector<float> vecnum;
            std::vector<std::string> vecstr;

            if (val.IsArray()) {
                auto arr = val.GetArray();
                assert(dim == arr.Size());
                for (int i = 0; i < arr.Size(); i++)
                {
                    if (arr[i].IsFloat())
                    {
                        vecnum.push_back(arr[i].GetFloat());
                    }
                    else if (arr[i].IsDouble())
                    {
                        vecnum.push_back(arr[i].GetDouble());
                    }
                    else if (arr[i].IsInt())
                    {
                        vecnum.push_back(arr[i].GetInt());
                    }
                    else if (arr[i].IsString())
                    {
                        //may be a curve json str, or a formula str
                        //but the format of curve in zsg2.0 represents as :
                        /*
                          "position": {
                              "default-value": {
                                   "objectType": "curve",
                                   "x": {...}
                                   "y": {...}
                                   "z": {...}
                              }
                          }

                           it seems that the k-frame is set on the whole vec,
                           may be we just want to k-frame for only one component.
                         */
                        vecstr.push_back(arr[i].GetString());
                    }
                    else {
                        zeno::log_error("unknown type value on vec parsing");
                        break;
                    }
                }
            }
            else if (val.IsString()) {
                std::string strinfo = val.GetString();
                std::vector<std::string> lst = zeno::split_str(strinfo);
                for (auto& num : lst)
                {
                    if (num.empty()) {
                        vecnum.push_back(0);
                    }
                    else {
                        //don't support formula.
                        vecnum.push_back(std::stof(num));
                    }
                }
            }

            if (vecnum.size() == dim) {
                if (zeno::Param_Vec2i == type) {
                    defl = zeno::vec2i(vecnum[0], vecnum[1]);
                }
                if (zeno::Param_Vec2f == type) {
                    defl = zeno::vec2f(vecnum[0], vecnum[1]);
                }
                if (zeno::Param_Vec3i == type) {
                    defl = zeno::vec3i(vecnum[0], vecnum[1], vecnum[2]);
                }
                if (zeno::Param_Vec3f == type) {
                    defl = zeno::vec3f(vecnum[0], vecnum[1], vecnum[2]);
                }
                if (zeno::Param_Vec4i == type) {
                    defl = zeno::vec4i(vecnum[0], vecnum[1], vecnum[2], vecnum[3]);
                }
                if (zeno::Param_Vec4f == type) {
                    defl = zeno::vec4f(vecnum[0], vecnum[1], vecnum[2], vecnum[3]);
                }
            }
            else if (vecstr.size() == dim) {
                if (zeno::Param_Vec2i == type || zeno::Param_Vec2f == type) {
                    defl = zeno::vec2s(vecstr[0], vecstr[1]);
                }
                if (zeno::Param_Vec3i == type || zeno::Param_Vec3f == type) {
                    defl = zeno::vec3s(vecstr[0], vecstr[1], vecstr[2]);
                }
                if (zeno::Param_Vec4i == type || zeno::Param_Vec4f == type) {
                    defl = zeno::vec4s(vecstr[0], vecstr[1], vecstr[2], vecstr[3]);
                }
            }
            else {
                zeno::log_error("unknown type value on vec parsing");
            }
            break;
        }
        case zeno::Param_Curve:
        {
            //todo: wrap the json object as string, and parse it when calculate,
            //by the method of parseCurve on ParseObjectFromUi.cpp
            break;
        }
        }
        return defl;
    }

    bool importControl(const rapidjson::Value& controlObj, zeno::ParamControl& ctrl, zeno::ControlProperty& props)
    {
        if (!controlObj.IsObject())
            return false;

        if (!controlObj.HasMember("name"))
            return false;

        const rapidjson::Value& nameObj = controlObj["name"];
        if (!nameObj.IsString())
            return false;

        const std::string& ctrlName = nameObj.GetString();
        ctrl = getControlByName(ctrlName);

        if (controlObj.HasMember("min") && controlObj.HasMember("max") &&
            controlObj.HasMember("step"))
        {
            if (controlObj["min"].IsNumber() && controlObj["max"].IsNumber() && controlObj["step"].IsNumber())
            {
                props.ranges = {
                    controlObj["min"].GetFloat(),
                    controlObj["max"].GetFloat(),
                    controlObj["step"].GetFloat()
                };
            }
        }
        if (controlObj.HasMember("items"))
        {
            if (controlObj["items"].IsArray())
            {
                auto& arr = controlObj["items"].GetArray();
                for (int i = 0; i < arr.Size(); i++)
                {
                    props.items->push_back(arr[i].GetString());
                }
            }
        }
        return true;
    }

}