#include "jsonhelper.h"
#include <zenoui/model/variantptr.h>
#include <zenoui/model/zserializable.h>
#include <zeno/utils/logger.h>

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

	void AddVariantList(const QVariantList& list, const QString& type, RAPIDJSON_WRITER& writer)
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
			else if (varType == QMetaType::VoidStar)
			{
                auto pModel = QVariantPtr<ZSerializable>::asPtr(value);
                auto s = pModel->z_serialize();
				writer.String(s.data(), s.size());
            }
			//todo: qlineargradient.
			else if (varType != QVariant::Invalid)
            {
				if (varType == QVariant::UserType) {
					//todo: declare a custom metatype
					QVector<qreal> vec = value.value<QVector<qreal>>();
                    if (!vec.isEmpty()) {
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

                writer.Null();
                zeno::log_warn("bad qt variant type {}", value.typeName() ? value.typeName() : "(null)");
                //Q_ASSERT(false);
			}
		}
		writer.EndArray();
	}

	void AddVariantListWithNull(const QVariantList& list, const QString& type, RAPIDJSON_WRITER& writer)
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
				writer.String(value.toString().toLatin1());
			}
			else if (varType == QVariant::Bool)
			{
				writer.Bool(value.toBool());
			}
			else
			{
                if (varType == QVariant::UserType) {
					//todo: declare a custom metatype
					QVector<qreal> vec = value.value<QVector<qreal>>();
					//for vec:
                    if (!vec.isEmpty()) {
                        writer.StartArray();
                        for (int i = 0; i < vec.size(); i++) {
							//todo: more type.
                            if (type == "vec3i")
                                writer.Int(vec[i]);
                            else
                                writer.Double(vec[i]);
						}
                        writer.EndArray();
						continue;
					}
				}
				if (varType != QVariant::Invalid)
					zeno::log_warn("bad param info qvariant type {}", value.typeName() ? value.typeName() : "(null)");
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
				if (varType != QVariant::Invalid)  // FIXME: so many QVector<qreal>???
					zeno::log_trace("bad param info qvariant type {}", value.typeName() ? value.typeName() : "(null)");
				writer.String("");
			}
		}
		writer.EndArray();
	}
}
