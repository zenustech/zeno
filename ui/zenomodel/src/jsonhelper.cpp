#include "jsonhelper.h"
#include "variantptr.h"
#include "curvemodel.h"
#include "zeno/utils/logger.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include "uihelper.h"
#include "zassert.h"

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

    void AddVariantList(const QVariantList& list, const QString& valType, RAPIDJSON_WRITER& writer, bool fillInvalid)
    {
        writer.StartArray();
        for (const QVariant& value : list)
        {
            //the valType is only availble for "value", for example, in phrase ["setNodeInput", "xxx-cube", "pos", (variant value)],
            // valType is only used for value, and the other phrases are parsed as string.
            AddVariant(value, valType, writer, fillInvalid);
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
        AddVariant(op, "string", writer, true);
        AddVariant(ident, "string", writer, true);
        AddVariant(name, "string", writer, true);
        AddVariant(defl, descType, writer, true);
        writer.EndArray();
    }

    bool importControl(const rapidjson::Value& controlObj, PARAM_CONTROL& ctrl, QVariant& props)
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

    void dumpControl(PARAM_CONTROL ctrl, const QVariant& props, RAPIDJSON_WRITER& writer)
    {
        writer.StartObject();

        writer.Key("name");
        QString controlDesc = UiHelper::getControlDesc(ctrl);
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

    void AddVariant(const QVariant& value, const QString& type, RAPIDJSON_WRITER& writer, bool fillInvalid)
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

                    for (int i = 0; i < vec.size(); i++) {
                        if (!bFloat)
                            writer.Int(vec[i]);
                        else
                            writer.Double(vec[i]);
                    }
                    writer.EndArray();
                }
            }
            else if (value.userType() == QMetaTypeId<CURVES_DATA>::qt_metatype_id())
            {
                if (type == "curve") {
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
                } else {
                    ZASSERT_EXIT(false);
                    writer.Null();
                }
            }
            else
            {
                //todo: color custom type.
                writer.Null();
            }
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
            if (fillInvalid)
                writer.Null();
        }
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
        return curve;
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
                bool bLockX = false;    //todo: lock io
                bool bLockY = false;

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

    QVariant importDescriptor(const rapidjson::Value &value, const QString &socketName, int type, QObject *parentRef) 
    {
        QVariant var;
        QString socketType;
        if (value.HasMember("type")) 
        {
            socketType = value["type"].GetString();
        }
        QVariant defaultValue;
        if (value.HasMember("default-value")) 
        {
            if (!value["default-value"].IsNull()) 
            {
                defaultValue = UiHelper::parseJsonByType(socketType, value["default-value"], parentRef);
            }
        }
        PARAM_CONTROL ctrl = CONTROL_NONE;
        QVariant props;
        if (value.HasMember("control")) 
        {
            JsonHelper::importControl(value["control"], ctrl, props);
        }
        if (!socketName.isEmpty()) 
        {
            if (type == PARAM_INPUT) 
            {
                INPUT_SOCKET inputSocket;
                inputSocket.info = SOCKET_INFO("", socketName);
                inputSocket.info.type = socketType;
                inputSocket.info.control = ctrl;
                inputSocket.info.ctrlProps = props.toMap();
                inputSocket.info.defaultValue = defaultValue;
                var = QVariant::fromValue(inputSocket);
            } 
            else if (type == PARAM_PARAM) 
            {
                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.control = ctrl;
                paramInfo.controlProps = props.toMap();
                paramInfo.name = socketName;
                paramInfo.typeDesc = socketType;
                paramInfo.defaultValue = defaultValue;
                var = QVariant::fromValue(paramInfo);
            }
            else if (type == PARAM_OUTPUT) 
            {
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = SOCKET_INFO("", socketName);
                outputSocket.info.type = socketType;
                outputSocket.info.control = ctrl;
                outputSocket.info.ctrlProps = props.toMap();
                outputSocket.info.defaultValue = defaultValue;
                var = QVariant::fromValue(outputSocket);
            } 
        }
        return var;
    }
}
