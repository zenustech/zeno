#include <rapidjson/document.h>
#include <zeno/io/iohelper.h>
#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/utils/string.h>
#include <zeno/utils/helper.h>
#include <filesystem>
#include <zeno/io/iotags.h>
#include "zeno_types/reflect/reflection.generated.hpp"


using namespace zeno::reflect;

namespace zenoio
{
    class JsonObjBatch
    {
    public:
        JsonObjBatch(RAPIDJSON_WRITER& writer)
            : m_writer(writer)
        {
            m_writer.StartObject();
        }
        ~JsonObjBatch()
        {
            m_writer.EndObject();
        }
    private:
        RAPIDJSON_WRITER& m_writer;
    };

    class JsonArrayBatch
    {
    public:
        JsonArrayBatch(RAPIDJSON_WRITER& writer)
            : m_writer(writer)
        {
            m_writer.StartArray();
        }
        ~JsonArrayBatch()
        {
            m_writer.EndArray();
        }
    private:
        RAPIDJSON_WRITER& m_writer;
    };

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
        else if (descName == "CodeEditor")
        {
            return zeno::CodeEditor;
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

    static zeno::CurveData parseCurve(rapidjson::Value const& jsonCurve, bool& bSucceed)
    {
        using namespace iotags::curve;
        zeno::CurveData dat;
        if (!jsonCurve.HasMember(key_range))
        {
            bSucceed = false;
            return dat;
        }

        const rapidjson::Value& rgObj = jsonCurve[key_range];
        if (!rgObj.HasMember(key_xFrom) || !rgObj.HasMember(key_xTo) ||
            !rgObj.HasMember(key_yFrom) || !rgObj.HasMember(key_yTo))
        {
            bSucceed = false;
            return dat;
        }

        if (!rgObj[key_xFrom].IsDouble() || !rgObj[key_xTo].IsDouble() ||
            !rgObj[key_yFrom].IsDouble() || !rgObj[key_yTo].IsDouble())
        {
            bSucceed = false;
            return dat;
        }

        //CURVE_RANGE rg;
        dat.rg.xFrom = rgObj[key_xFrom].GetDouble();
        dat.rg.xTo = rgObj[key_xTo].GetDouble();
        dat.rg.yFrom = rgObj[key_yFrom].GetDouble();
        dat.rg.yTo = rgObj[key_yTo].GetDouble();

        //todo: id

        if (!jsonCurve.HasMember(key_nodes)) {
            bSucceed = false;
            return dat;
        }

        for (const rapidjson::Value& nodeObj : jsonCurve[key_nodes].GetArray())
        {
            if (!nodeObj.HasMember("x") || !nodeObj["x"].IsDouble() ||
                !nodeObj.HasMember("y") || !nodeObj["y"].IsDouble() ||
                !nodeObj.HasMember(key_left_handle) || !nodeObj[key_left_handle].IsObject() ||
                !nodeObj.HasMember(key_right_handle) || !nodeObj[key_right_handle].IsObject())
            {
                bSucceed = false;
                return dat;
            }

            float x = nodeObj["x"].GetDouble();
            float y = nodeObj["y"].GetDouble();

            auto leftHdlObj = nodeObj[key_left_handle].GetObject();
            if (!leftHdlObj.HasMember("x") || !leftHdlObj.HasMember("y"))
            {
                bSucceed = false;
                return dat;
            }
            float leftX = leftHdlObj["x"].GetDouble();
            float leftY = leftHdlObj["y"].GetDouble();

            auto rightHdlObj = nodeObj[key_right_handle].GetObject();
            if (!rightHdlObj.HasMember("x") || !rightHdlObj.HasMember("y"))
            {
                bSucceed = false;
                return dat;
            }
            float rightX = rightHdlObj["x"].GetDouble();
            float rightY = rightHdlObj["y"].GetDouble();

            zeno::CurveData::PointType pointType = zeno::CurveData::kBezier;
            if (nodeObj.HasMember(key_type) && nodeObj[key_type].IsString())
            {
                std::string type = nodeObj[key_type].GetString();
                if (type == "bezier") {
                    pointType = zeno::CurveData::kBezier;
                }
                else if (type == "kConstant") {
                    pointType = zeno::CurveData::kConstant;
                }
                else if (type == "linear") {
                    pointType = zeno::CurveData::kLinear;
                }
            }
            zeno::CurveData::HANDLE_TYPE handleType = zeno::CurveData::HDL_FREE;
            if (nodeObj.HasMember(key_handle_type) && nodeObj[key_handle_type].IsString())
            {
                std::string type = nodeObj[key_handle_type].GetString();
                if (type == "aligned")
                    handleType = zeno::CurveData::HDL_ALIGNED;
                else if (type == "asym")
                    handleType = zeno::CurveData::HDL_ASYM;
                else if (type == "free")
                    handleType = zeno::CurveData::HDL_FREE;
                else if (type == "vector")
                    handleType = zeno::CurveData::HDL_VECTOR;
            }

            //todo
            bool bLockX = (nodeObj.HasMember("lockX") && nodeObj["lockX"].IsBool());
            bool bLockY = (nodeObj.HasMember("lockY") && nodeObj["lockY"].IsBool());

            dat.addPoint(x, y, pointType, { leftX, leftY }, { rightX, rightY }, handleType);
        }

        if (!jsonCurve.HasMember("nodes")) {
            bSucceed = false;
            return dat;
        }

        if (jsonCurve.HasMember(key_cycle_type) && jsonCurve[key_cycle_type].IsString()) {
            zeno::CurveData::CycleType cycleType = zeno::CurveData::kClamp;
            std::string type = jsonCurve[key_cycle_type].GetString();
            if (type == "kClamp")
                cycleType = zeno::CurveData::kClamp;
            else if (type == "KCycle")
                cycleType = zeno::CurveData::kCycle;
            else if (type == "KMirror")
                cycleType = zeno::CurveData::kMirror;
            dat.cycleType = cycleType;
        }
        if (jsonCurve.HasMember(key_visible) && jsonCurve[key_visible].IsBool()) {
            dat.visible = jsonCurve[key_visible].GetBool();
        }
        if (jsonCurve.HasMember(key_timeline) && jsonCurve[key_timeline].IsBool()) {
            dat.timeline = jsonCurve[key_timeline].GetBool();
        }

        bSucceed = true;
        return dat;
    }

    static zeno::CurvesData parseObjectFromJson(rapidjson::Value const& x, bool& bSucceed)
    {
        bSucceed = true;
        zeno::CurvesData curves;
        for (auto i = x.MemberBegin(); i != x.MemberEnd(); i++) {
            if (i->value.IsObject())
            {
                zeno::CurveData dat = parseCurve(i->value, bSucceed);
                if (!bSucceed) {
                    bSucceed = false;
                    break;
                }
                else {
                    curves.keys.insert({ i->name.GetString(), dat });
                }
            }
        }
        return curves;
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

    zeno::SocketType getSocketTypeByDesc(const std::string& sockType)
    {
        if (sockType == iotags::params::socket_none) {
            return zeno::NoSocket;
        }
        else if (sockType == iotags::params::socket_readonly) {
            return zeno::Socket_ReadOnly;
        }
        else if (sockType == iotags::params::socket_clone) {
            return zeno::Socket_Clone;
        }
        else if (sockType == iotags::params::socket_output) {
            return zeno::Socket_Output;
        }
        else if (sockType == iotags::params::socket_owning) {
            return zeno::Socket_Owning;
        }
        else if (sockType == iotags::params::socket_primitive) {
            return zeno::Socket_Primitve;
        }
        else if (sockType == iotags::params::socket_wildcard) {
            return zeno::Socket_WildCard;
        }
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
                auto primparams = customUiToParams(nodeData.customUi.inputPrims);
                for (const auto& info : primparams)
                {
                    if (info.name == "name")
                    {
                        zeno::zeno_get_if(info.defl, newName);
                        break;
                    }
                }
            }
            else {
                newName = cls + std::to_string(newIdNum);
            }

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

    zeno::reflect::Any jsonValueToAny(const rapidjson::Value& val, zeno::ParamType const& type)
    {
        zeno::reflect::Any defl;
        switch (type) {
        case gParamType_Int:
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
                    defl = sval;
                else
                    defl = 0;
            }
            else {
                zeno::log_error("error type");
            }
            break;
        }
        case gParamType_Float:
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
                    defl = sval;
                else
                    defl = (float)0.0;
            }
            else
                zeno::log_error("error type");
            break;
        }
        case gParamType_Bool:
        {
            if (val.IsBool())
                defl = val.GetBool();
            else if (val.IsInt())
                defl = val.GetInt() != 0;
            else if (val.IsFloat())
                defl = val.GetFloat() != 0;
            else
                zeno::log_error("error type");
            break;
        }
        case gParamType_String:
        {
            if (val.IsString())
                defl = (std::string)val.GetString();
            break;
        }
        case gParamType_Vec2i:
        case gParamType_Vec2f:
        case gParamType_Vec3i:
        case gParamType_Vec3f:
        case gParamType_Vec4i:
        case gParamType_Vec4f:
        {
            int dim = 0;
            bool bFloat = false;
            if (gParamType_Vec2i == type) dim = 2; bFloat = false;
            if (gParamType_Vec2f == type) dim = 2; bFloat = true;
            if (gParamType_Vec3i == type) dim = 3; bFloat = false;
            if (gParamType_Vec3f == type) dim = 3; bFloat = true;
            if (gParamType_Vec4i == type) dim = 4; bFloat = false;
            if (gParamType_Vec4f == type) dim = 4; bFloat = true;

            zeno::vecvar editvec;
            auto arr = val.GetArray();
            assert(dim == arr.Size());
            for (int i = 0; i < arr.Size(); i++)
            {
                if (arr[i].IsFloat())
                {
                    editvec.push_back(arr[i].GetFloat());
                }
                else if (arr[i].IsDouble())
                {
                    editvec.push_back(static_cast<float>(arr[i].GetDouble()));
                }
                else if (arr[i].IsInt())
                {
                    editvec.push_back(arr[i].GetInt());
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
                    editvec.push_back(arr[i].GetString());
                }
                else {
                    zeno::log_error("unknown type value on vec parsing");
                    break;
                }
            }
            defl = editvec;
            break;
        }
        case gParamType_Curve:
        {
            //todo: wrap the json object as string, and parse it when calculate,
            //by the method of parseCurve on ParseObjectFromUi.cpp
            if (val.IsObject())
            {
                bool bSucceed = false;
                zeno::CurvesData curves = parseObjectFromJson(val, bSucceed);
                assert(bSucceed);
                defl = curves;
            }
            break;
        }
        case gParamType_Heatmap: {
            break;
        }
        case Param_Null:
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

    zeno::zvariant jsonValueToZVar(const rapidjson::Value& val, zeno::ParamType const& type)
    {
        zeno::zvariant defl;
        switch (type) {
        case gParamType_Int:
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
                    defl = sval;
                else
                    defl = 0;
            }
            else {
                zeno::log_error("error type");
            }
            break;
        }
        case gParamType_Float:
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
                    defl = sval;
                else
                    defl = (float)0.0;
            }
            else
                zeno::log_error("error type");
            break;
        }
        case gParamType_Bool:
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
        case gParamType_String:
        {
            if (val.IsString())
                defl = val.GetString();
            break;
        }
        case gParamType_Vec2i:
        case gParamType_Vec2f:
        case gParamType_Vec3i:
        case gParamType_Vec3f:
        case gParamType_Vec4i:
        case gParamType_Vec4f:
        {
            int dim = 0;
            bool bFloat = false;
            if (gParamType_Vec2i == type) dim = 2; bFloat = false;
            if (gParamType_Vec2f == type) dim = 2; bFloat = true;
            if (gParamType_Vec3i == type) dim = 3; bFloat = false;
            if (gParamType_Vec3f == type) dim = 3; bFloat = true;
            if (gParamType_Vec4i == type) dim = 4; bFloat = false;
            if (gParamType_Vec4f == type) dim = 4; bFloat = true;

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
                if (gParamType_Vec2i == type) {
                    defl = zeno::vec2i(vecnum[0], vecnum[1]);
                }
                if (gParamType_Vec2f == type) {
                    defl = zeno::vec2f(vecnum[0], vecnum[1]);
                }
                if (gParamType_Vec3i == type) {
                    defl = zeno::vec3i(vecnum[0], vecnum[1], vecnum[2]);
                }
                if (gParamType_Vec3f == type) {
                    defl = zeno::vec3f(vecnum[0], vecnum[1], vecnum[2]);
                }
                if (gParamType_Vec4i == type) {
                    defl = zeno::vec4i(vecnum[0], vecnum[1], vecnum[2], vecnum[3]);
                }
                if (gParamType_Vec4f == type) {
                    defl = zeno::vec4f(vecnum[0], vecnum[1], vecnum[2], vecnum[3]);
                }
            }
            else if (vecstr.size() == dim) {
                if (gParamType_Vec2i == type || gParamType_Vec2f == type) {
                    defl = zeno::vec2s(vecstr[0], vecstr[1]);
                }
                if (gParamType_Vec3i == type || gParamType_Vec3f == type) {
                    defl = zeno::vec3s(vecstr[0], vecstr[1], vecstr[2]);
                }
                if (gParamType_Vec4i == type || gParamType_Vec4f == type) {
                    defl = zeno::vec4s(vecstr[0], vecstr[1], vecstr[2], vecstr[3]);
                }
            }
            else {
                zeno::log_error("unknown type value on vec parsing");
            }
            break;
        }
        case gParamType_Curve:
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
        case gParamType_Heatmap:
        {
            if (val.IsString())
            {
                defl = val.GetString();
            }
            break;
        }
        case Param_Null:
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

    void dumpCurve(const zeno::CurveData& curve, RAPIDJSON_WRITER& writer)
    {
        auto rg = curve.rg;

        JsonObjBatch scope(writer);
        writer.Key(iotags::curve::key_range);
        {
            JsonObjBatch scope2(writer);
            writer.Key(iotags::curve::key_xFrom);
            writer.Double(rg.xFrom);
            writer.Key(iotags::curve::key_xTo);
            writer.Double(rg.xTo);
            writer.Key(iotags::curve::key_yFrom);
            writer.Double(rg.yFrom);
            writer.Key(iotags::curve::key_yTo);
            writer.Double(rg.yTo);
        }

        writer.Key(iotags::curve::key_nodes);
        {
            JsonArrayBatch arrBatch(writer);
            assert(curve.cpoints.size() == curve.cpbases.size());
            for (int i = 0; i < curve.cpoints.size(); i++) {
                auto& pt = curve.cpoints[i];
                std::pair<float, float> pos = { curve.cpbases[i], curve.cpoints[i].v };
                const auto& leftPos = pt.left_handler;
                const auto& rightPos = pt.right_handler;
                bool bLockX = false;    //todo: lock io
                bool bLockY = false;

                JsonObjBatch scope2(writer);
                writer.Key("x");
                writer.Double(pos.first);
                writer.Key("y");
                writer.Double(pos.second);

                writer.Key(iotags::curve::key_left_handle);
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(leftPos[0]);
                    writer.Key("y");
                    writer.Double(leftPos[1]);
                }
                writer.Key(iotags::curve::key_right_handle);
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(rightPos[0]);
                    writer.Key("y");
                    writer.Double(rightPos[1]);
                }

                writer.Key(iotags::curve::key_type);
                switch (pt.cp_type) {
                case zeno::CurveData::kBezier: writer.String("bezier"); break;
                case zeno::CurveData::kLinear: writer.String("linear"); break;
                case zeno::CurveData::kConstant: writer.String("constant"); break;
                default:
                    assert(false);
                    writer.String("unknown");
                    break;
                }

                writer.Key(iotags::curve::key_handle_type);
                switch (pt.controlType) {
                case zeno::CurveData::HDL_ALIGNED: writer.String("aligned"); break;
                case zeno::CurveData::HDL_ASYM: writer.String("asym"); break;
                case zeno::CurveData::HDL_FREE: writer.String("free"); break;
                case zeno::CurveData::HDL_VECTOR: writer.String("vector"); break;
                default:
                    assert(false);
                    writer.String("unknown");
                    break;
                }

                writer.Key(iotags::curve::key_lockX);
                writer.Bool(bLockX);
                writer.Key(iotags::curve::key_lockY);
                writer.Bool(bLockY);
            }
        }
        writer.Key(iotags::curve::key_cycle_type);
        switch (curve.cycleType) {
        case zeno::CurveData::CycleType::kClamp:
            writer.String("kClamp"); break;
        case zeno::CurveData::CycleType::kCycle:
            writer.String("KCycle"); break;
        case zeno::CurveData::CycleType::kMirror:
            writer.String("KMirror"); break;
        default:
            writer.String("unknown"); break;
        }
        writer.Key(iotags::curve::key_visible);
        writer.Bool(curve.visible);
        writer.Key(iotags::curve::key_timeline);
        writer.Bool(curve.timeline);
    }

    void dumpCurves(const zeno::CurvesData* curves, RAPIDJSON_WRITER& writer)
    {
        writer.StartObject();
        writer.Key(iotags::key_objectType);
        writer.String("curve");
        for (auto& [key, curve] : curves->keys) {
            writer.Key(key.c_str());
            dumpCurve(curve, writer);
        }
        writer.EndObject();
    }

    void writeAny(const zeno::reflect::Any& any, zeno::ParamType coreType, RAPIDJSON_WRITER& writer)
    {
        if (!any.has_value()) {
            writer.Null();
            return;
        }

        zeno::ParamType anyType = any.type().hash_code();
        switch (coreType)
        {
            case gParamType_PrimVariant:
            case gParamType_VecEdit:
                assert(false);
                zeno::log_error("core type cannot be edit type");
                break;
            case gParamType_Int:
            {
                if (anyType == gParamType_PrimVariant) {
                    std::visit([&](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, int>) {
                            writer.Int(arg);
                        }
                        else {
                            assert(false);
                            zeno::log_error("type error when writing param defl value");
                            writer.Null();
                        }
                    }, any_cast<zeno::PrimVar>(any));
                }
                else {
                    assert(anyType == gParamType_Int);
                    writer.Int(any_cast<int>(any));
                }
                break;
            }
            case gParamType_Float:
            {
                if (anyType == gParamType_PrimVariant) {
                    std::visit([&](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, float>) {
                            writer.Double(arg);
                        }
                        else {
                            assert(false);
                            zeno::log_error("type error when writing param defl value");
                            writer.Null();
                        }
                    }, any_cast<zeno::PrimVar>(any));
                }
                else {
                    assert(anyType == gParamType_Float);
                    writer.Double(any_cast<float>(any));
                }
                break;
            }
            case gParamType_Bool:
            {
                assert(anyType == gParamType_Bool);
                writer.Bool(any_cast<bool>(any));
                break;
            }
            case gParamType_String:
            {
                assert(anyType == gParamType_String);
                writer.String(any_cast<std::string>(any).c_str());
                break;
            }
            case gParamType_Curve:
            {
                if (auto pCurves = zeno::reflect::any_cast<zeno::CurvesData>(&any)) {
                    //后续会采用序列化进行读写，现在先转为json字符串储存，以便复用以前的代码
                    dumpCurves(pCurves, writer);
                }
                break;
            }
            case gParamType_Heatmap:
            {
                //TODO:
                writer.Null();
                break;
            }
            case gParamType_Vec2i:
            case gParamType_Vec2f:
            case gParamType_Vec3i:
            case gParamType_Vec3f:
            case gParamType_Vec4i:
            case gParamType_Vec4f:
            {
                if (anyType == gParamType_VecEdit) {
                    const zeno::vecvar& vec = any_cast<zeno::vecvar>(any);
                    writer.StartArray();
                    for (const zeno::PrimVar& elem : vec) {
                        std::visit([&](auto&& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, float>) {
                                writer.Double(arg);
                            }
                            else if constexpr (std::is_same_v<T, int>) {
                                writer.Int(arg);
                            }
                            else if constexpr (std::is_same_v<T, std::string>) {
                                writer.String(arg.c_str());
                            }
                            else if constexpr (std::is_same_v<T, zeno::CurveData>) {
                                //TODO
                            }
                        }, any_cast<zeno::PrimVar>(elem));
                    }
                    writer.EndArray();
                }
                else {
                    //其实defl肯定都是edit类型的
                    assert(false);
                    writer.Null();
                }
                break;
            }
            default:
            {
                //TODO: custom type
                writer.Null();
                break;
            }
        }
    }

    void writeZVariant(zeno::zvariant defl, zeno::ParamType type, RAPIDJSON_WRITER& writer)
    {
        switch (type)
        {
            case gParamType_Int:
            {
                int val = 0;
                if (std::holds_alternative<int>(defl))
                {
                    val = (std::get<int>(defl));
                    writer.Int(val);
                }
                else if (std::holds_alternative<float>(defl))
                {
                    val = (std::get<float>(defl));
                    writer.Int(val);
                }
                else if (std::holds_alternative<std::string>(defl))
                {
                    std::string str = (std::get<std::string>(defl));
                    writer.String(str.c_str());
                }
                break;
            }
            case gParamType_Float:
            {
                float val = 0;
                if (std::holds_alternative<int>(defl))
                {
                    val = (std::get<int>(defl));
                    writer.Double(val);
                }
                else if (std::holds_alternative<float>(defl))
                {
                    val = (std::get<float>(defl));
                    writer.Double(val);
                }
                else if (std::holds_alternative<std::string>(defl))
                {
                    std::string str = (std::get<std::string>(defl));
                    writer.String(str.c_str());
                }
                break;
            }
            case gParamType_Bool:
            {
                int val = 0;
                if (std::holds_alternative<int>(defl))
                {
                    val = (std::get<int>(defl));
                }
                writer.Bool(val != 0);
                break;
            }
            case gParamType_String:
            case gParamType_Curve:
            case gParamType_Heatmap:
            {
                std::string val;
                if (std::holds_alternative<std::string>(defl))
                {
                    val = (std::get<std::string>(defl));
                }
                writer.String(val.c_str());
                break;
            }
            case gParamType_Vec2i:
            case gParamType_Vec2f:
            case gParamType_Vec3i:
            case gParamType_Vec3f:
            case gParamType_Vec4i:
            case gParamType_Vec4f:
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

    bool importControl(const rapidjson::Value& controlObj, zeno::ParamControl& ctrl, zeno::reflect::Any& props)
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
                std::vector<float> ranges = {
                    controlObj["min"].GetFloat(),
                    controlObj["max"].GetFloat(),
                    controlObj["step"].GetFloat()
                };
                props = ranges;
            }
        }
        if (controlObj.HasMember("items"))
        {
            if (controlObj["items"].IsArray())
            {
                auto& arr = controlObj["items"].GetArray();
                std::vector<std::string> items;
                for (int i = 0; i < arr.Size(); i++)
                {
                    items.push_back(arr[i].GetString());
                }
                props = items;
            }
        }
        return true;
    }

    void dumpControl(zeno::ParamType type, zeno::ParamControl ctrl, const zeno::reflect::Any& ctrlProps, RAPIDJSON_WRITER& writer)
    {
        writer.StartObject();

        writer.Key("name");
        std::string controlDesc = zeno::getControlDesc(ctrl, type);
        writer.String(controlDesc.c_str());

        if (ctrlProps.has_value())
        {
            //TODO
#if 0
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
#endif
        }

        writer.EndObject();
    }

}