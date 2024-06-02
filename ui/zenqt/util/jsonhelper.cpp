#include "jsonhelper.h"
#include "variantptr.h"
#include "model/curvemodel.h"
#include "zeno/utils/logger.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include "uihelper.h"
#include "zassert.h"
#include "util/curveutil.h"

using namespace zeno::iotags;
using namespace zeno::iotags::curve;

namespace JsonHelper
{
    void AddStringList(const QStringList& list, RAPIDJSON_WRITER& writer)
    {
        writer.StartArray();
        for (auto item : list)
		{       // TODO: luzh, can we support UFT-8 (chinese char) here?
            auto s = item.toStdString();
			writer.String(s.data(), s.size());
        }
        writer.EndArray();
    }

    void AddVariantList(const QVariantList& list, const QString& valType, RAPIDJSON_WRITER& writer)
    {
        writer.StartArray();
        for (const QVariant& value : list)
        {
            //the valType is only availble for "value", for example, in phrase ["setNodeInput", "xxx-cube", "pos", (variant value)],
            // valType is only used for value, and the other phrases are parsed as string.
            AddVariant(value, valType, writer);
		}
        writer.EndArray();
    }

    void AddParams(const QString& op,
        const QString& ident,
        const QString& name,
        const QVariant& defl,
        const QString& descType,
        RAPIDJSON_WRITER& writer
    )
    {
        writer.StartArray();
        AddVariant(op, "string", writer);
        AddVariant(ident, "string", writer);
        AddVariant(name, "string", writer);
        AddVariant(defl, descType, writer);
        writer.EndArray();
    }

    bool importControl(const rapidjson::Value& controlObj, zeno::ParamControl& ctrl, QVariant& props)
    {
        if (!controlObj.IsObject())
            return false;

        if (!controlObj.HasMember("name"))
            return false;

        const rapidjson::Value& nameObj = controlObj["name"];
        ZASSERT_EXIT(nameObj.IsString(), false);

        const QString& ctrlName = nameObj.GetString();
        ctrl = UiHelper::getControlByDesc(ctrlName);

        QVariantMap ctrlProps;
        for (const auto& keyObj : controlObj.GetObject())
        {
            const QString& key = keyObj.name.GetString();
            const rapidjson::Value& value = keyObj.value;
            if (key == "name")
                continue;

            QVariant varItem = UiHelper::parseJson(value);
            ctrlProps.insert(key, varItem);
        }
        props = ctrlProps;
        return true;
    }

    void dumpControl(zeno::ParamType type, zeno::ParamControl ctrl, const QVariant& props, RAPIDJSON_WRITER& writer)
    {
        writer.StartObject();

        writer.Key("name");
        QString controlDesc = UiHelper::getControlDesc(ctrl, type);
        writer.String(controlDesc.toUtf8());

        QVariantMap ctrlProps = props.toMap();
        for (QString keyName : ctrlProps.keys()) {
            writer.Key(keyName.toUtf8());
            JsonHelper::WriteVariant(ctrlProps[keyName], writer);
        }

        writer.EndObject();
    }

    void WriteVariant(const QVariant& value, RAPIDJSON_WRITER& writer)
    {
        QVariant::Type varType = value.type();
        if (varType == QVariant::Double)
        {
            writer.Double(value.toDouble());
        }
        else if (varType == QMetaType::Float)
        {
            writer.Double(value.toFloat());
        }
        else if (varType == QVariant::Int)
        {
            writer.Int(value.toInt());
        }
        else if (varType == QVariant::String)
        {
            auto s = value.toString().toStdString();
            writer.String(s.data(), s.size());
        }
        else if (varType == QVariant::Bool)
        {
            writer.Bool(value.toBool());
        }
        else if (varType == QVariant::StringList)
        {
            writer.StartArray();
            auto lst = value.toStringList();
            for (auto item : lst)
            {
                writer.String(item.toUtf8());
            }
            writer.EndArray();
        }
    }

    bool AddVariant(const QVariant& value, const QString& type, RAPIDJSON_WRITER& writer)
    {
        QVariant::Type varType = value.type();
        if (varType == QVariant::Double)
        {
            return writer.Double(value.toDouble());
        }
        else if (varType == QMetaType::Float)
        {
            return writer.Double(value.toFloat());
        }
        else if (varType == QVariant::Int)
        {
            return writer.Int(value.toInt());
        }
        else if (varType == QVariant::String)
        {
            auto s = value.toString().toStdString();
            writer.String(s.data(), s.size());
        }
        else if (varType == QVariant::Bool)
        {
            writer.Bool(value.toBool());
        }
        else if (varType == QVariant::UserType)
        {
            if (value.userType() == QMetaTypeId<UI_VECTYPE>::qt_metatype_id())
            {
                UI_VECTYPE vec = value.value<UI_VECTYPE>();
                if (!vec.isEmpty())
                {
                    writer.StartArray();

                    int dim = -1;
                    bool bFloat = false;
                    UiHelper::parseVecType(type, dim, bFloat);
                    // ^^^ use regexp, but not include colorvec3f
                    // vvv so specify here
                    if (type == "colorvec3f") {
                        bFloat = true;
                    }

                    for (int i = 0; i < vec.size(); i++) {
                        if (!bFloat)
                            writer.Int(vec[i]);
                        else
                            writer.Double(vec[i]);
                    }
                    writer.EndArray();
                }
                else {
                    writer.Null();
                }
            }
            else if (value.userType() == QMetaTypeId<UI_VECSTRING>::qt_metatype_id()) {
                UI_VECSTRING vec = value.value<UI_VECSTRING>();
                if (!vec.isEmpty()) {
                    writer.StartArray();

                    for (int i = 0; i < vec.size(); i++) {
                        auto s = vec[i].toStdString();
                        writer.String(s.data(), s.size());
                    }

                    writer.EndArray();
                }
                else {
                    writer.Null();
                }
            }
            else if (value.userType() == QMetaTypeId<CURVES_DATA>::qt_metatype_id())
            {
                if (type == "curve" || value.canConvert<CURVES_DATA>())
                {
                    CURVES_DATA curves = value.value<CURVES_DATA>();
                    writer.StartObject();
                    writer.Key(key_objectType);
                    writer.String("curve");
                    writer.Key(key_timeline);
                    if (curves.size() == 0)
                    {
                        writer.Bool(false);
                    } else {
                        writer.Bool((*curves.begin()).timeline);
                    }
                    for (auto curve : curves) {
                        writer.Key(curve.key.toUtf8());
                        dumpCurve(curve, writer);
                    }
                    writer.EndObject();
                }
                else
                {
                    writer.Null();
                    ZASSERT_EXIT(false, true);
                }
            }
            else
            {
                //todo: color custom type.
                writer.Null();
            }
        } 
        else if (varType == QVariant::Color) 
        {
            auto s = value.value<QColor>().name().toStdString();
            writer.String(s.data(), s.size());
        }
        //todo: qlineargradient.
        else if (varType != QVariant::Invalid)
        {
            if (varType == QMetaType::VoidStar)
            {
                //todo: color custom type.
                writer.Null();
            }
            else
            {
                writer.Null();
                zeno::log_warn("bad qt variant type {}", value.typeName() ? value.typeName() : "(null)");
                //Q_ASSERT(false);
            }
        }
        else if (varType == QVariant::Invalid)
        {
            writer.Null();
        }
        else {
            writer.Null();
        }
        return true;
    }

    void AddVariantToStringList(const QVariantList& list, RAPIDJSON_WRITER& writer)
    {
        writer.StartArray();
        for (const QVariant& value : list)
        {
            QVariant::Type varType = value.type();
            if (varType == QVariant::Double)
            {
                writer.String(std::to_string(value.toDouble()).c_str());
            }
            else if (varType == QVariant::Int)
            {
                writer.String(std::to_string(value.toInt()).c_str());
            }
            else if (varType == QVariant::String)
            {
                writer.String(value.toString().toUtf8());
            }
            else if (varType == QVariant::Bool)
            {
                writer.String(std::to_string((int)value.toBool()).c_str());
            }
            else 
            {
                if (varType != QVariant::Invalid)
                    zeno::log_trace("bad param info qvariant type {}", value.typeName() ? value.typeName() : "(null)");
                writer.String("");
            }
        }
        writer.EndArray();
    }

    CURVE_DATA parseCurve(QString channel, const rapidjson::Value& jsonCurve)
    {
        CURVE_DATA curve;

        ZASSERT_EXIT(jsonCurve.HasMember(key_range), curve);
        const rapidjson::Value& rgObj = jsonCurve[key_range];
        ZASSERT_EXIT(rgObj.HasMember(key_xFrom) && rgObj.HasMember(key_xTo) && rgObj.HasMember(key_yFrom) && rgObj.HasMember(key_yTo), curve);

        CURVE_RANGE rg;
        ZASSERT_EXIT(rgObj[key_xFrom].IsDouble() && rgObj[key_xTo].IsDouble() && rgObj[key_yFrom].IsDouble() && rgObj[key_yTo].IsDouble(), curve);
        rg.xFrom = rgObj[key_xFrom].GetDouble();
        rg.xTo = rgObj[key_xTo].GetDouble();
        rg.yFrom = rgObj[key_yFrom].GetDouble();
        rg.yTo = rgObj[key_yTo].GetDouble();

        curve.rg = rg;
        curve.key = channel;

        ZASSERT_EXIT(jsonCurve.HasMember(key_nodes), curve);
        for (const rapidjson::Value& nodeObj : jsonCurve[key_nodes].GetArray())
        {
            ZASSERT_EXIT(nodeObj.HasMember("x") && nodeObj["x"].IsDouble(), curve);
            ZASSERT_EXIT(nodeObj.HasMember("y") && nodeObj["y"].IsDouble(), curve);
            QPointF pos(nodeObj["x"].GetDouble(), nodeObj["y"].GetDouble());

            ZASSERT_EXIT(nodeObj.HasMember(key_left_handle) && nodeObj[key_left_handle].IsObject(), curve);
            auto leftHdlObj = nodeObj[key_left_handle].GetObject();
            ZASSERT_EXIT(leftHdlObj.HasMember("x") && leftHdlObj.HasMember("y"), curve);
            qreal leftX = leftHdlObj["x"].GetDouble();
            qreal leftY = leftHdlObj["y"].GetDouble();
            QPointF leftOffset(leftX, leftY);

            ZASSERT_EXIT(nodeObj.HasMember(key_right_handle) && nodeObj[key_right_handle].IsObject(), curve);
            auto rightHdlObj = nodeObj[key_right_handle].GetObject();
            ZASSERT_EXIT(rightHdlObj.HasMember("x") && rightHdlObj.HasMember("y"), curve);
            qreal rightX = rightHdlObj["x"].GetDouble();
            qreal rightY = rightHdlObj["y"].GetDouble();
            QPointF rightOffset(rightX, rightY);

            HANDLE_TYPE hdlType = HDL_ASYM;
            if (nodeObj.HasMember(key_type) && nodeObj[key_type].IsString())
            {
                QString type = nodeObj[key_type].GetString();
                if (type == "aligned") {
                    hdlType = HDL_ALIGNED;
                }
                else if (type == "asym") {
                    hdlType = HDL_ASYM;
                }
                else if (type == "free") {
                    hdlType = HDL_FREE;
                }
                else if (type == "vector") {
                    hdlType = HDL_VECTOR;
                }
            }

            bool bLockX = (nodeObj.HasMember(key_lockX) && nodeObj[key_lockX].IsBool());
            bool bLockY = (nodeObj.HasMember(key_lockY) && nodeObj[key_lockY].IsBool());

            CURVE_POINT pt;
            pt.point = pos;
            pt.leftHandler = leftOffset;
            pt.rightHandler = rightOffset;
            pt.controlType = hdlType;

            curve.points.append(pt);
        }
        if (jsonCurve.HasMember(key_visible)) 
        {
            curve.visible = jsonCurve[key_visible].GetBool();
        }
        return curve;
    }

    CURVES_DATA JsonHelper::parseCurves(const QString& curveJson)
    {
        CURVES_DATA curves;
        rapidjson::Document doc;
        doc.Parse(curveJson.toStdString().c_str());

        if (!doc.IsObject())
            return curves;
        for (auto iter = doc.MemberBegin(); iter != doc.MemberEnd(); iter++) {
            if (iter->value.IsObject()) {
                bool bSucceed = false;
                CURVE_DATA dat = parseCurve(iter->name.GetString(), iter->value);
                curves[iter->name.GetString()] = dat;
            }
        }
        return curves;
    }

    CURVES_DATA parseCurves(const QVariant& val)
    {
        if (val.userType() == QMetaTypeId<UI_VECSTRING>::qt_metatype_id())
        {
            UI_VECSTRING strVec = val.value<UI_VECSTRING>();
            CURVES_DATA curves;
            for (int i = 0; i < strVec.size(); i++)
            {
                QString key = curve_util::getCurveKey(i);
                bool ok = false;
                QString str = strVec.at(i);
                str.toFloat(&ok);
                if (ok)
                    continue;
                rapidjson::Document doc;
                doc.Parse(str.toStdString().c_str());
                if (!doc.IsObject())
                    continue;
                CURVES_DATA data = parseCurves(str);
                curves[key] = data[key];
            }
            return curves;
        }
        else if (val.type() == QVariant::String)
        {
            return parseCurves(val.toString());
        }
        else
        {
            return CURVES_DATA();
        }
    }

    CurveModel* _parseCurveModel(QString channel, const rapidjson::Value& jsonCurve, QObject* parentRef)
    {
        ZASSERT_EXIT(jsonCurve.HasMember(key_range), nullptr);
        const rapidjson::Value& rgObj = jsonCurve[key_range];
        ZASSERT_EXIT(rgObj.HasMember(key_xFrom) && rgObj.HasMember(key_xTo) && rgObj.HasMember(key_yFrom) && rgObj.HasMember(key_yTo), nullptr);

        CURVE_RANGE rg;
        ZASSERT_EXIT(rgObj[key_xFrom].IsDouble() && rgObj[key_xTo].IsDouble() && rgObj[key_yFrom].IsDouble() && rgObj[key_yTo].IsDouble(), nullptr);
        rg.xFrom = rgObj[key_xFrom].GetDouble();
        rg.xTo = rgObj[key_xTo].GetDouble();
        rg.yFrom = rgObj[key_yFrom].GetDouble();
        rg.yTo = rgObj[key_yTo].GetDouble();

        CurveModel *pModel = new CurveModel(channel, rg, parentRef);

        //if (jsonCurve.HasMember(key_timeline) && jsonCurve[key_timeline].IsBool())
        //{
        //    bool bTimeline = jsonCurve[key_timeline].GetBool();
        //    pModel->setTimeline(bTimeline);
        //}

        ZASSERT_EXIT(jsonCurve.HasMember(key_nodes), nullptr);
        for (const rapidjson::Value& nodeObj : jsonCurve[key_nodes].GetArray())
        {
            ZASSERT_EXIT(nodeObj.HasMember("x") && nodeObj["x"].IsDouble(), nullptr);
            ZASSERT_EXIT(nodeObj.HasMember("y") && nodeObj["y"].IsDouble(), nullptr);
            QPointF pos(nodeObj["x"].GetDouble(), nodeObj["y"].GetDouble());

            ZASSERT_EXIT(nodeObj.HasMember(key_left_handle) && nodeObj[key_left_handle].IsObject(), nullptr);
            auto leftHdlObj = nodeObj[key_left_handle].GetObject();
            ZASSERT_EXIT(leftHdlObj.HasMember("x") && leftHdlObj.HasMember("y"), nullptr);
            qreal leftX = leftHdlObj["x"].GetDouble();
            qreal leftY = leftHdlObj["y"].GetDouble();
            QPointF leftOffset(leftX, leftY);

            ZASSERT_EXIT(nodeObj.HasMember(key_right_handle) && nodeObj[key_right_handle].IsObject(), nullptr);
            auto rightHdlObj = nodeObj[key_right_handle].GetObject();
            ZASSERT_EXIT(rightHdlObj.HasMember("x") && rightHdlObj.HasMember("y"), nullptr);
            qreal rightX = rightHdlObj["x"].GetDouble();
            qreal rightY = rightHdlObj["y"].GetDouble();
            QPointF rightOffset(rightX, rightY);

            HANDLE_TYPE hdlType = HDL_ASYM;
            if (nodeObj.HasMember(key_type) && nodeObj[key_type].IsString())
            {
                QString type = nodeObj[key_type].GetString();
                if (type == "aligned") {
                    hdlType = HDL_ALIGNED;
                }
                else if (type == "asym") {
                    hdlType = HDL_ASYM;
                }
                else if (type == "free") {
                    hdlType = HDL_FREE;
                }
                else if (type == "vector") {
                    hdlType = HDL_VECTOR;
                }
            }

            bool bLockX = (nodeObj.HasMember(key_lockX) && nodeObj[key_lockX].IsBool());
            bool bLockY = (nodeObj.HasMember(key_lockY) && nodeObj[key_lockY].IsBool());

            QStandardItem* pItem = new QStandardItem;
            pItem->setData(pos, ROLE_NODEPOS);
            pItem->setData(leftOffset, ROLE_LEFTPOS);
            pItem->setData(rightOffset, ROLE_RIGHTPOS);
            pItem->setData(hdlType, ROLE_TYPE);
            pModel->appendRow(pItem);
        }
        return pModel;
    }

    void dumpCurve(const CURVE_DATA& curve, RAPIDJSON_WRITER& writer)
    {
        CURVE_RANGE rg = curve.rg;

        JsonObjBatch scope(writer);
        writer.Key(key_range);
        {
            JsonObjBatch scope2(writer);
            writer.Key(key_xFrom);
            writer.Double(rg.xFrom);
            writer.Key(key_xTo);
            writer.Double(rg.xTo);
            writer.Key(key_yFrom);
            writer.Double(rg.yFrom);
            writer.Key(key_yTo);
            writer.Double(rg.yTo);
        }

        writer.Key(key_nodes);
        {
            JsonArrayBatch arrBatch(writer);
            for (CURVE_POINT pt : curve.points) {
                const QPointF &pos = pt.point;
                const QPointF &leftPos = pt.leftHandler;
                const QPointF &rightPos = pt.rightHandler;
                HANDLE_TYPE hdlType = (HANDLE_TYPE)pt.controlType;
                bool bLockX = pt.bLockX;    //todo: lock io
                bool bLockY = pt.bLockY;

                JsonObjBatch scope2(writer);
                writer.Key("x");
                writer.Double(pos.x());
                writer.Key("y");
                writer.Double(pos.y());

                writer.Key(key_left_handle);
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(leftPos.x());
                    writer.Key("y");
                    writer.Double(leftPos.y());
                }
                writer.Key(key_right_handle);
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(rightPos.x());
                    writer.Key("y");
                    writer.Double(rightPos.y());
                }

                writer.Key(key_type);
                switch (hdlType) {
                case HDL_ALIGNED: writer.String("aligned"); break;
                case HDL_ASYM: writer.String("asym"); break;
                case HDL_FREE: writer.String("free"); break;
                case HDL_VECTOR: writer.String("vector"); break;
                default:
                    Q_ASSERT(false);
                    writer.String("unknown");
                    break;
                }

                writer.Key(key_lockX);
                writer.Bool(bLockX);
                writer.Key(key_lockY);
                writer.Bool(bLockY);
            }
        }
        writer.Key(key_visible);
        writer.Bool(curve.visible);
    }

    QString JsonHelper::dumpCurves(const CURVES_DATA& curves)
    {
        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);
        writer.StartObject();
        for (auto curve : curves) {
            writer.Key(curve.key.toUtf8());
            dumpCurve(curve, writer);
        }
        writer.EndObject();
        return QString::fromUtf8(s.GetString());
    }

    void JsonHelper::dumpCurves(const CURVES_DATA& curves, QVariant& val)
    {
        if (curves.isEmpty())
            return;
        UI_VECSTRING vec;
        if (val.canConvert<UI_VECSTRING>())
        {
            vec = val.value<UI_VECSTRING>();
        }
        else if (val.canConvert<UI_VECTYPE>())
        {
            UI_VECTYPE data = val.value<UI_VECTYPE>();
            for (int i = 0; i < data.size(); i++)
            {
                vec << UiHelper::variantToString(data.at(i));
            }
        }
        else
        {
            vec << UiHelper::variantToString(val);
        }
        for (auto curve : curves) {
            CURVES_DATA data;
            data[curve.key] = curve;
            QString val = dumpCurves(data);
            QString key = curve.key;
            int idx = key == "x" ? 0 : key == "y" ? 1 : key == "z" ? 2 : 3;
            if (vec.size() > idx)
            {
                vec[idx] = val;
            }
        }

        if (vec.size() == 1)
            val = QVariant::fromValue(vec.first());
        else
            val = QVariant::fromValue(vec);
    }

    bool JsonHelper::parseHeatmap(const QString& json, int& nres, QString& grad)
    {
        rapidjson::Document doc;
        doc.Parse(json.toStdString().c_str());

        if (!doc.IsObject() || !doc.HasMember("nres") || !doc.HasMember("color"))
            return false;
        nres = doc["nres"].GetInt();
        grad = doc["color"].GetString();
        return true;
    }

    QString JsonHelper::dumpHeatmap(int nres, const QString& grad)
    {
        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);
        writer.StartObject();
        writer.Key("nres");
        writer.Int(nres);
        writer.Key("color");
        writer.String(grad.toUtf8());
        writer.EndObject();
        return QString::fromUtf8(s.GetString());
    }

    void dumpCurveModel(const CurveModel* pModel, RAPIDJSON_WRITER& writer)
    {
        if (!pModel) {
            return;
        }

        CURVE_RANGE rg = pModel->range();

        JsonObjBatch scope(writer);
        writer.Key(key_range);
        {
            JsonObjBatch scope2(writer);
            writer.Key(key_xFrom);
            writer.Double(rg.xFrom);
            writer.Key(key_xTo);
            writer.Double(rg.xTo);
            writer.Key(key_yFrom);
            writer.Double(rg.yFrom);
            writer.Key(key_yTo);
            writer.Double(rg.yTo);
        }

        writer.Key(key_nodes);
        {
            JsonArrayBatch arrBatch(writer);
            for (int i = 0; i < pModel->rowCount(); i++)
            {
                const QModelIndex &idx = pModel->index(i, 0);
                const QPointF &pos = idx.data(ROLE_NODEPOS).toPointF();
                const QPointF &leftPos = idx.data(ROLE_LEFTPOS).toPointF();
                const QPointF &rightPos = idx.data(ROLE_RIGHTPOS).toPointF();
                HANDLE_TYPE hdlType = (HANDLE_TYPE)idx.data(ROLE_TYPE).toInt();
                bool bLockX = idx.data(ROLE_LOCKX).toBool();
                bool bLockY = idx.data(ROLE_LOCKY).toBool();

                JsonObjBatch scope2(writer);
                writer.Key("x");
                writer.Double(pos.x());
                writer.Key("y");
                writer.Double(pos.y());

                writer.Key(key_left_handle);
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(leftPos.x());
                    writer.Key("y");
                    writer.Double(leftPos.y());
                }
                writer.Key(key_right_handle);
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(rightPos.x());
                    writer.Key("y");
                    writer.Double(rightPos.y());
                }

                writer.Key(key_type);
                switch (hdlType)
                {
                case HDL_ALIGNED: writer.String("aligned"); break;
                case HDL_ASYM: writer.String("asym"); break;
                case HDL_FREE: writer.String("free"); break;
                case HDL_VECTOR: writer.String("vector"); break;
                default:
                    Q_ASSERT(false);
                    writer.String("unknown");
                    break;
                }

                writer.Key(key_lockX);
                writer.Bool(bLockX);
                writer.Key(key_lockY);
                writer.Bool(bLockY);
            }
        }
    }
}
