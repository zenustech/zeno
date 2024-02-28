#ifndef __IO_COMMON_H__
#define __IO_COMMON_H__

#include <zeno/core/data.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>

namespace zenoio {
    //already declared in zeno/core/common.h
    //enum ZSG_VERSION
    //{
    //    VER_2,          //old version io
    //    VER_2_5,        //new version io
    //};

    typedef rapidjson::PrettyWriter<rapidjson::StringBuffer> RAPIDJSON_WRITER;

    class JsonObjScope
    {
    public:
        JsonObjScope(RAPIDJSON_WRITER& writer)
            : m_writer(writer)
        {
            m_writer.StartObject();
        }
        ~JsonObjScope()
        {
            m_writer.EndObject();
        }
    private:
        RAPIDJSON_WRITER& m_writer;
    };

    struct AppSettings
    {
        zeno::TimelineInfo timeline;
        //todo: other settings.
    };

    struct ZSG_PARSE_RESULT {
        zeno::GraphData mainGraph;
        zeno::ZSG_VERSION iover;
        zeno::TimelineInfo timeline;
        bool bSucceed;
    };
}

#endif