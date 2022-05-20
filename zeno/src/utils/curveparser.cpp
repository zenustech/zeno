#include <zeno/utils/curveparser.h>
#include <zeno/utils/iotag.h>

namespace zeno {

    using namespace curve;

    CurveData parseCurve(Value const& jsonCurve, bool& bSucceed)
    {
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

}