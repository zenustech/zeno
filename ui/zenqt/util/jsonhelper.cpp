#include "jsonhelper.h"
#include "variantptr.h"
#include "model/curvemodel.h"
#include "zeno/utils/logger.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include "uihelper.h"
#include "zassert.h"
#include "util/curveutil.h"
#include <zeno/io/iocommon.h>


using namespace zenoio::iotags::curve;

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

    CurveModel* _parseCurveModel(QString channel, const rapidjson::Value& jsonCurve, QObject* parentRef)
    {
        ZASSERT_EXIT(jsonCurve.HasMember(key_range), nullptr);
        const rapidjson::Value& rgObj = jsonCurve[key_range];
        ZASSERT_EXIT(rgObj.HasMember(key_xFrom) && rgObj.HasMember(key_xTo) && rgObj.HasMember(key_yFrom) && rgObj.HasMember(key_yTo), nullptr);

        zeno::CurveData::Range rg;
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

#if 0
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
#endif

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
}
