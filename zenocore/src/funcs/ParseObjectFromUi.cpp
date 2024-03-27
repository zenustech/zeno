#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/types/CurveObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/utils/string.h>

namespace zeno {
namespace {

    using namespace rapidjson;
    using namespace iotags;

    CurveData parseCurve(Value const& jsonCurve, bool& bSucceed)
    {
        using namespace iotags::curve;
        CurveData dat;
        if (!jsonCurve.HasMember(key_range))
        {
            bSucceed = false;
            return dat;
        }

        const Value &rgObj = jsonCurve[key_range];
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

        if (!jsonCurve.HasMember("nodes")) {
            bSucceed = false;
            return dat;
        }

        for (const Value &nodeObj : jsonCurve["nodes"].GetArray())
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

            CurveData::PointType type = CurveData::kBezier;
            if (nodeObj.HasMember("type") && nodeObj["type"].IsString())
            {
                std::string type = nodeObj["type"].GetString();
                if (type == "aligned") {
                    type = CurveData::kBezier;
                } else if (type == "asym") {
                    type = CurveData::kBezier;
                } else if (type == "free") {
                    type = CurveData::kBezier;
                } else if (type == "vector") {
                    type = CurveData::kLinear;
                }
            }

            //todo
            bool bLockX = (nodeObj.HasMember("lockX") && nodeObj["lockX"].IsBool());
            bool bLockY = (nodeObj.HasMember("lockY") && nodeObj["lockY"].IsBool());

            dat.addPoint(x, y, type, {leftX, leftY}, {rightX, rightY});
        }

        bSucceed = true;
        return dat;
    }

} // namespace


zany parseObjectFromUi(Value const& x)
{
    bool bSucceed = false;
    auto curve = std::make_shared<zeno::CurveObject>();
    for (auto i = x.MemberBegin(); i != x.MemberEnd(); i++) {
        if (i->value.IsObject())
        {
               CurveData dat = parseCurve(i->value, bSucceed);
            if (!bSucceed) {
                return nullptr;
            } else {
                curve->keys.insert({i->name.GetString(), dat});
            }
        }
    }
    return curve;
}

zany parseCurveObj(const std::string& curveJson)
{
    auto curve = std::make_shared<zeno::CurveObject>();

    rapidjson::Document doc;
    doc.Parse(curveJson.c_str());

    if (!doc.IsObject())
        return nullptr;
    for (auto iter = doc.MemberBegin(); iter != doc.MemberEnd(); iter++) {
        if (iter->value.IsObject()) {
            bool bSucceed = false;
            CurveData dat = parseCurve(iter->value, bSucceed);
            if (!bSucceed) {
                return nullptr;
            }
            else {
                curve->keys.insert({ iter->name.GetString(), dat });
            }
        }
    }
    
    return curve;
}
zany parseHeatmapObj(const std::string& json)
{
    auto heatmap = std::make_shared<zeno::HeatmapObject>();
    rapidjson::Document doc;
    doc.Parse(json.c_str());

    if (!doc.IsObject() || !doc.HasMember("nres") || !doc.HasMember("color"))
        return nullptr;
    int nres = doc["nres"].GetInt();
    std::string ramps = doc["color"].GetString();
    std::stringstream ss(ramps);
    std::vector<std::pair<float, zeno::vec3f>> colors;
    int count;
    ss >> count;
    for (int i = 0; i < count; i++) {
        float f = 0.f, x = 0.f, y = 0.f, z = 0.f;
        ss >> f >> x >> y >> z;
        //printf("%f %f %f %f\n", f, x, y, z);
        colors.emplace_back(
            f, zeno::vec3f(x, y, z));
    }

    for (int i = 0; i < nres; i++) {
        float fac = i * (1.f / nres);
        zeno::vec3f clr;
        for (int j = 0; j < colors.size(); j++) {
            auto [f, rgb] = colors[j];
            if (f >= fac) {
                if (j != 0) {
                    auto [last_f, last_rgb] = colors[j - 1];
                    auto intfac = (fac - last_f) / (f - last_f);
                    //printf("%f %f %f %f\n", fac, last_f, f, intfac);
                    clr = zeno::mix(last_rgb, rgb, intfac);
                }
                else {
                    clr = rgb;
                }
                break;
            }
        }
        heatmap->colors.push_back(clr);
    }
    return heatmap;
}

}
