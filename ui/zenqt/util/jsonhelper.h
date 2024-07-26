#ifndef __JSON_HELPER_H__
#define __JSON_HELPER_H__

#include <QObject>
#include <QtWidgets>
#include <rapidjson/document.h>

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#include "uicommon.h"


typedef rapidjson::PrettyWriter<rapidjson::StringBuffer> RAPIDJSON_WRITER;
typedef rapidjson::PrettyWriter<rapidjson::StringBuffer> PRETTY_WRITER;

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
    void WriteVariant(const QVariant& var, RAPIDJSON_WRITER& writer);
    void dumpControl(zeno::ParamType type, zeno::ParamControl ctrl, const QVariant& props, RAPIDJSON_WRITER& writer);
    bool importControl(const rapidjson::Value& controlObj, zeno::ParamControl& ctrl, QVariant& props);
    CurveModel* _parseCurveModel(QString channel, const rapidjson::Value& jsonCurve, QObject* parentRef);
    bool parseHeatmap(const QString& json, int &nres, QString &grad);
    QString dumpHeatmap(int nres, const QString& grad);
}



#endif