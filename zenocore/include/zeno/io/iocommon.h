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

    namespace iotags {
        constexpr const char* key_objectType = "objectType";
        constexpr const char* sZencache_lockfile_prefix = "zencache_lockfile_";

        namespace curve {
            constexpr const char* key_timeline = "timeline"; // whether curve type is timeline
            constexpr const char* key_nodes = "nodes";       //curve node.
            constexpr const char* key_range = "range";       //curve range.
            constexpr const char* key_xFrom = "xFrom";
            constexpr const char* key_xTo = "xTo";
            constexpr const char* key_yFrom = "yFrom";
            constexpr const char* key_yTo = "yTo";
            constexpr const char* key_left_handle = "left-handle";
            constexpr const char* key_right_handle = "right-handle";
            constexpr const char* key_type = "type";
            constexpr const char* key_lockX = "lockX";
            constexpr const char* key_lockY = "lockY";
            constexpr const char* key_visible = "visible";
        }

        namespace timeline {
            constexpr const char* start_frame = "start-frame";
            constexpr const char* end_frame = "end-frame";
            constexpr const char* curr_frame = "curr-frame";
            constexpr const char* always = "always";
            constexpr const char* timeline_fps = "timeline-fps";
        }

        namespace recordinfo {
            constexpr const char* record_path = "record-path";
            constexpr const char* videoname = "video-name";
            constexpr const char* fps = "fps";
            constexpr const char* bitrate = "bitrate";
            constexpr const char* numMSAA = "numMSAA";
            constexpr const char* numOptix = "numOptix";
            constexpr const char* width = "width";
            constexpr const char* height = "height";
            constexpr const char* bExportVideo = "export-video";
            constexpr const char* needDenoise = "need-denoise";
            constexpr const char* bAutoRemoveCache = "auto-remove-cache";
            constexpr const char* bAov = "aov";
            constexpr const char* bExr = "exr";
            constexpr const char* exePath = "exePath";
        }

        namespace layoutinfo {
            constexpr const char* layout = "layout";
        }

        namespace userdatainfo {
            constexpr const char* optixShowBackground = "optix-show-background";
        }

        namespace params {
            constexpr const char* node_inputs = "inputs";
            constexpr const char* node_inputs_objs = "object_inputs";
            constexpr const char* node_inputs_primitive = "primitive_inputs";
            constexpr const char* node_outputs_objs = "object_outputs";
            constexpr const char* node_outputs_primitive = "primitive_outputs";
            constexpr const char* node_params = "params";
            constexpr const char* node_outputs = "outputs";
            constexpr const char* panel_root = "root";
            constexpr const char* panel_default_tab = "Default";
            constexpr const char* panel_inputs = "In Sockets";
            constexpr const char* panel_params = "Parameters";
            constexpr const char* panel_outputs = "Out Sockets";
            constexpr const char* params_valueKey = "value";

            //socket type
            constexpr const char* socket_none = "none";
            constexpr const char* socket_readonly = "readonly";
            constexpr const char* socket_clone = "clone";
            constexpr const char* socket_output = "output";
            constexpr const char* socket_owning = "owning";
            constexpr const char* socket_primitive = "primitive";
            constexpr const char* socket_wildcard = "wildcard";

            //legacy desc params
            constexpr const char* legacy_inputs = "legacy_inputs";
            constexpr const char* legacy_params = "legacy_params";
            constexpr const char* legacy_outputs = "legacy_outputs";
        }
    }

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

    enum ERR_CODE {
        PARSE_NOERROR,
        PARSE_VERSION_UNKNOWN,
        PARSE_ERROR
    };

    struct ZSG_PARSE_RESULT {
        zeno::GraphData mainGraph;
        zeno::ZSG_VERSION iover;
        zeno::TimelineInfo timeline;
        ERR_CODE code;
    };
}

#endif