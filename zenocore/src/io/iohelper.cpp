#include <rapidjson/document.h>
#include <zeno/io/iohelper.h>
#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/utils/string.h>
#include <zeno/utils/helper.h>
#include <filesystem>


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
            return zeno::ReadPathEdit;
        }
        else if (descName == "write path")
        {
            return zeno::WritePathEdit;
        }
        else if (descName == "directory")
        {
            return zeno::DirectoryPathEdit;
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

    ZENO_API zeno::ZSG_VERSION getVersion(const std::string& fn)
    {
        std::filesystem::path filePath(fn);
        if (!std::filesystem::exists(filePath)) {
            zeno::log_error("cannot open zsg file: {} ({})", fn);
            return zeno::UNKNOWN_VER;
        }

        rapidjson::Document doc;

        auto szBuffer = std::filesystem::file_size(filePath);
        if (szBuffer == 0)
        {
            zeno::log_error("the zsg file is a empty file");
            return zeno::UNKNOWN_VER;
        }

        std::vector<char> dat(szBuffer);
        FILE* fp = fopen(filePath.string().c_str(), "rb");
        if (!fp) {
            zeno::log_error("zsg file does not exist");
            return zeno::UNKNOWN_VER;
        }

        size_t actualSz = fread(&dat[0], 1, szBuffer, fp);
        if (actualSz != szBuffer) {
            zeno::log_warn("the bytes read from file is different from the size of whole file");
        }
        fclose(fp);
        fp = nullptr;

        doc.Parse(&dat[0], actualSz);
        std::string ver = doc["version"].GetString();

        if (ver == "v2")
            return zeno::VER_2;
        else if (ver == "v2.5")
            return zeno::VER_2_5;
        else if (ver == "v3")
            return zeno::VER_3;
        else
            return zeno::UNKNOWN_VER;
    }

    zeno::GraphData fork(
        const std::map<std::string, zeno::GraphData>& sharedSubg,
        const std::string& subnetClassName)
    {
        zeno::GraphData newGraph;
        newGraph.templateName = subnetClassName;

        std::unordered_map<std::string, std::string> old2new;
        zeno::LinksData oldLinks;

        auto it = sharedSubg.find(subnetClassName);
        if (it == sharedSubg.end())
        {
            return newGraph;
        }

        std::unordered_map<std::string, int> node_idx_set;

        const zeno::GraphData& subgraph = it->second;
        for (const auto& [name, nodeData] : subgraph.nodes)
        {
            zeno::NodeData nodeDat = nodeData;
            const std::string& oldName = nodeDat.name;

            bool isSubgraph = nodeDat.subgraph.has_value();
            const std::string& cls = isSubgraph ? nodeDat.subgraph->templateName : nodeDat.cls;
            if (isSubgraph) {
                assert(sharedSubg.find(cls) != sharedSubg.end());
            }

            if (node_idx_set.find(cls) == node_idx_set.end()) {
                node_idx_set[cls] = 1;
            }
            int newIdNum = node_idx_set[cls]++;
            std::string newName;;
            if (cls == "SubInput" || cls == "SubOutput")
            {
                for (const auto& info : nodeData.inputs)
                {
                    if (info.name == "name")
                    {
                        newName = std::get<std::string>(info.defl);
                        break;
                    }
                }
            }
            else
                newName = cls + std::to_string(newIdNum);

            old2new.insert(std::make_pair(oldName, newName));

            if (isSubgraph)
            {
                nodeDat.name = newName;

                zeno::GraphData fork_subgraph;
                fork_subgraph = fork(sharedSubg, cls);
                fork_subgraph.name = newName;
                nodeDat.subgraph = fork_subgraph;
                
                newGraph.nodes[newName] = nodeDat;
            }
            else
            {
                nodeDat.name = newName;
                newGraph.nodes[newName] = nodeDat;
            }
        }

        for (zeno::EdgeInfo oldLink : subgraph.links)
        {
            zeno::EdgeInfo newLink = oldLink;
            newLink.inNode = old2new[newLink.inNode];
            newLink.outNode = old2new[newLink.outNode];
            newGraph.links.push_back(newLink);
        }

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
                std::string sval(val.GetString());
                if (!sval.empty())
                    defl = std::stof(sval);
                else
                    defl = 0;
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
            {
                std::string sval(val.GetString());
                if (!sval.empty())
                    defl = std::stof(sval);
                else
                    defl = (float)0.0;
            }
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
            if (val.IsString())
                defl = val.GetString();
            if (val.IsObject())
            {
                rapidjson::StringBuffer sbBuf;
                RAPIDJSON_WRITER jWriter(sbBuf);
                val.Accept(jWriter);
                defl = std::string(sbBuf.GetString());
            }
            break;
        }
        case zeno::Param_Heatmap:
        {
            if (val.IsString())
            {
                defl = val.GetString();
            }
            break;
        }
        case zeno::Param_Null:
        {
            if (val.IsString())
            {
                defl = val.GetString();
            }
            else if (val.IsInt())
            {
                defl = val.GetInt();
            }
            else if (val.IsFloat())
            {
                defl = val.GetFloat();
            }
            else if (val.IsBool())
            {
                defl = val.GetBool();
            }
        }
        }
        return defl;
    }

    void writeZVariant(zeno::zvariant defl, zeno::ParamType type, RAPIDJSON_WRITER& writer)
    {
        switch (type)
        {
            case zeno::Param_Int:
            {
                int val = 0;
                if (std::holds_alternative<int>(defl))
                {
                    val = (std::get<int>(defl));
                }
                else if (std::holds_alternative<float>(defl))
                {
                    val = (std::get<float>(defl));
                }
                writer.Int(val);
                break;
            }
            case zeno::Param_Float:
            {
                float val = 0;
                if (std::holds_alternative<int>(defl))
                {
                    val = (std::get<int>(defl));
                }
                else if (std::holds_alternative<float>(defl))
                {
                    val = (std::get<float>(defl));
                }
                writer.Double(val);
                break;
            }
            case zeno::Param_Bool:
            {
                int val = 0;
                if (std::holds_alternative<int>(defl))
                {
                    val = (std::get<int>(defl));
                }
                writer.Bool(val != 0);
                break;
            }
            case zeno::Param_String:
            case zeno::Param_Curve:
            case zeno::Param_Heatmap:
            {
                std::string val;
                if (std::holds_alternative<std::string>(defl))
                {
                    val = (std::get<std::string>(defl));
                }
                writer.String(val.c_str());
                break;
            }
            case zeno::Param_Vec2i:
            case zeno::Param_Vec2f:
            case zeno::Param_Vec3i:
            case zeno::Param_Vec3f:
            case zeno::Param_Vec4i:
            case zeno::Param_Vec4f:
            {
                if (std::holds_alternative<zeno::vec2f>(defl))
                {
                    auto vec = std::get<zeno::vec2f>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.Double(elem);
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec2i>(defl))
                {
                    auto vec = std::get<zeno::vec2i>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.Int(elem);
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec2s>(defl))
                {
                    auto vec = std::get<zeno::vec2s>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.String(elem.c_str());
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec3i>(defl))
                {
                    auto vec = std::get<zeno::vec3i>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.Int(elem);
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec3f>(defl))
                {
                    auto vec = std::get<zeno::vec3f>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.Double(elem);
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec3s>(defl))
                {
                    auto vec = std::get<zeno::vec3s>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.String(elem.c_str());
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec4i>(defl))
                {
                    auto vec = std::get<zeno::vec4i>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.Int(elem);
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec4f>(defl))
                {
                    auto vec = std::get<zeno::vec4f>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.Double(elem);
                    }
                    writer.EndArray();
                }
                else if (std::holds_alternative<zeno::vec4s>(defl))
                {
                    auto vec = std::get<zeno::vec4s>(defl);
                    writer.StartArray();
                    for (auto elem : vec) {
                        writer.String(elem.c_str());
                    }
                    writer.EndArray();
                }
                else
                {
                    writer.Null();
                }
                break;
            }
            default:
            {
                writer.Null();
                break;
            }
        }
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
                props.items = std::vector<std::string>();
                for (int i = 0; i < arr.Size(); i++)
                {
                    props.items->push_back(arr[i].GetString());
                }
            }
        }
        return true;
    }

    void dumpControl(zeno::ParamType type, zeno::ParamControl ctrl, std::optional<zeno::ControlProperty> ctrlProps, RAPIDJSON_WRITER& writer)
    {
        writer.StartObject();

        writer.Key("name");
        std::string controlDesc = zeno::getControlDesc(ctrl, type);
        writer.String(controlDesc.c_str());

        if (ctrlProps.has_value())
        {
            zeno::ControlProperty props = ctrlProps.value();
            if (props.items.has_value()) {
                writer.Key("items");
                writer.StartArray();
                for (auto item : props.items.value())
                    writer.String(item.c_str());
                writer.EndArray();
            }
            else if (props.ranges.has_value()) {
                writer.Key("min");
                writer.Double(props.ranges.value()[0]);
                writer.Key("max");
                writer.Double(props.ranges.value()[1]);
                writer.Key("step");
                writer.Double(props.ranges.value()[2]);
            }
        }

        writer.EndObject();
    }

}