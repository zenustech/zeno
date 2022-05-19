#include "jsonhelper.h"
#include <zenoui/model/variantptr.h>
#include <zenoui/model/curvemodel.h>
#include <zeno/utils/logger.h>
#include <zenoui/model/curvemodel.h>
#include <zenoui/model/variantptr.h>

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

    void AddVariantList(const QVariantList& list, const QString& type, RAPIDJSON_WRITER& writer, bool fillInvalid)
    {
        writer.StartArray();
        for (const QVariant& value : list)
        {
            QVariant::Type varType = value.type();
            if (varType == QVariant::Double)
            {
                writer.Double(value.toDouble());
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
            //todo: qlineargradient.
            else if (varType != QVariant::Invalid)
            {
				if (varType == QVariant::UserType)
                {
                    //todo: declare a custom metatype
                    QVector<qreal> vec = value.value<QVector<qreal>>();
                    if (!vec.isEmpty())
                    {
                        writer.StartArray();
                        for (int i = 0; i < vec.size(); i++) {
                            if (type == "vec3i")
                                writer.Int(vec[i]);
                            else
                                writer.Double(vec[i]);
                        }
                        writer.EndArray();
                        continue;
                    }
                }
                else if (varType == QMetaType::VoidStar)
                {
                    // TODO: use qobject_cast<CurveModel *>(QVariantPtr<IModel>::asPtr(value))
                    // also btw luzh, will this have a memory leakage? no, we make sure that curvemodel is child of subgraphmodel.
                    if (type == "curve")
                    {
                        auto pModel = QVariantPtr<CurveModel>::asPtr(value);
                        dumpCurveModel(pModel, writer);
                    }
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
        writer.EndArray();
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
                writer.String(value.toString().toLatin1());
            }
            else if (varType == QVariant::Bool)
            {
                writer.String(std::to_string((int)value.toBool()).c_str());
            }
            else 
            {
                if (varType != QVariant::Invalid)  // FIXME: so many QVector<qreal>??? i will give a typedef later, and declare a qt meta type.
                    zeno::log_trace("bad param info qvariant type {}", value.typeName() ? value.typeName() : "(null)");
                writer.String("");
            }
        }
        writer.EndArray();
    }

    void dumpCurveModel(const CurveModel* pModel, RAPIDJSON_WRITER& writer)
    {
        if (!pModel) {
            return;
        }

        CURVE_RANGE rg = pModel->range();

        JsonObjBatch scope(writer);
        writer.Key("range");
        {
            JsonObjBatch scope2(writer);
            writer.Key("xFrom");
            writer.Double(rg.xFrom);
            writer.Key("xTo");
            writer.Double(rg.xTo);
            writer.Key("yFrom");
            writer.Double(rg.yFrom);
            writer.Key("yTo");
            writer.Double(rg.yTo);
        }

        writer.Key("timeline");
        writer.Bool(pModel->isTimeline());

        writer.Key("nodes");
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
                
                writer.Key("left-handle");
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(leftPos.x());
                    writer.Key("y");
                    writer.Double(leftPos.y());
                }
                writer.Key("right-handle");
                {
                    JsonObjBatch scope3(writer);
                    writer.Key("x");
                    writer.Double(rightPos.x());
                    writer.Key("y");
                    writer.Double(rightPos.y());
                }

                writer.Key("type");
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

                writer.Key("lockX");
                writer.Bool(bLockX);
                writer.Key("lockY");
                writer.Bool(bLockY);
            }        
        }
    }
}
