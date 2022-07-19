#ifndef __JSON_HELPER_H__
#define __JSON_HELPER_H__

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#include <QtWidgets>

typedef rapidjson::Writer<rapidjson::StringBuffer> RAPIDJSON_WRITER;

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

class CurveModel;

namespace JsonHelper
{
	void AddStringList(const QStringList& list, RAPIDJSON_WRITER& writer);
	void AddVariantList(const QVariantList& list, const QString& type, RAPIDJSON_WRITER& writer, bool fillInvalid = false);
	void AddVariant(const QVariant& var, const QString& type, RAPIDJSON_WRITER& writer, bool fillInvalid);
	void AddVariantToStringList(const QVariantList& list, RAPIDJSON_WRITER& writer);
	void dumpCurveModel(const CurveModel* pModel, RAPIDJSON_WRITER& writer);
}



#endif