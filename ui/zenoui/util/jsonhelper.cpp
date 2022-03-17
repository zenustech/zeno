#include "jsonhelper.h"
#include <zeno/utils/logger.h>

namespace JsonHelper
{
	void AddStringList(const QStringList& list, RAPIDJSON_WRITER& writer)
	{
		writer.StartArray();
		for (auto item : list)
		{
			writer.String(item.toLatin1());
		}
		writer.EndArray();
	}

	void AddVariantList(const QVariantList& list, RAPIDJSON_WRITER& writer)
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
				if (varType != QVariant::Invalid)
					zeno::log_warn("bad qt variant type {}", value.typeName() ? value.typeName() : "(null)");
				Q_ASSERT(false);
			}
		}
		writer.EndArray();
	}

	void AddVariantListWithNull(const QVariantList& list, RAPIDJSON_WRITER& writer)
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
				if (varType != QVariant::Invalid)
					zeno::log_warn("bad param info qvariant type {}", value.typeName() ? value.typeName() : "(null)");
				writer.String("");
			}
		}
		writer.EndArray();
	}
}