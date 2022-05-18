#include <zeno/utils/curveparser.h>

namespace zeno {

    CurveData parseCurve(Value const& jsonCurve, bool& bSucceed)
    {
        CurveData dat;
        if (!jsonCurve.HasMember("range"))
        {
            bSucceed = false;
            return dat;
        }

        const Value& rgObj = jsonCurve["range"];
        if (!rgObj.HasMember("xFrom") || !rgObj.HasMember("xTo") ||
            !rgObj.HasMember("yFrom") || !rgObj.HasMember("yTo"))
        {
            bSucceed = false;
            return dat;
        }

        if (!rgObj["xFrom"].IsDouble() || !rgObj["xTo"].IsDouble() ||
            !rgObj["yFrom"].IsDouble() || !rgObj["yTo"].IsDouble())
        {
            bSucceed = false;
            return dat;
        }

        //CURVE_RANGE rg;
        dat.rg.xFrom = rgObj["xFrom"].GetDouble();
        dat.rg.xTo = rgObj["xTo"].GetDouble();
        dat.rg.yFrom = rgObj["yFrom"].GetDouble();
        dat.rg.yTo = rgObj["yTo"].GetDouble();

        //todo: id

        if (!jsonCurve.HasMember("nodes")) {
            bSucceed = false;
            return dat;
        }

        for (const Value &nodeObj : jsonCurve["nodes"].GetArray())
        {
            if (!nodeObj.HasMember("x") || !nodeObj["x"].IsDouble() ||
                !nodeObj.HasMember("y") || !nodeObj["y"].IsDouble() ||
                !nodeObj.HasMember("left-handle") || !nodeObj["left-handle"].IsObject() ||
                !nodeObj.HasMember("right-handle") || !nodeObj["right-handle"].IsObject())
            {
                bSucceed = false;
                return dat;
            }

            float x = nodeObj["x"].GetDouble();
            float y = nodeObj["y"].GetDouble();

            auto leftHdlObj = nodeObj["left-handle"].GetObject();
            if (!leftHdlObj.HasMember("x") || !leftHdlObj.HasMember("y"))
            {
                bSucceed = false;
                return dat;
            }
            float leftX = leftHdlObj["x"].GetDouble();
            float leftY = leftHdlObj["y"].GetDouble();

            auto rightHdlObj = nodeObj["right-handle"].GetObject();
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