#pragma once

#include <zeno/zeno.h>
#include <rapidjson/document.h>

namespace zeno {

zany parseObjectFromUi(rapidjson::Value const &x);

namespace iotags {
    constexpr const char *key_objectType = "objectType";
    constexpr const char* sZencache_lockfile_prefix = "zencache_lockfile_";

    namespace curve {
        constexpr const char *key_timeline = "timeline"; // whether curve type is timeline
        constexpr const char *key_nodes = "nodes";       //curve node.
        constexpr const char *key_range = "range";       //curve range.
        constexpr const char *key_xFrom = "xFrom";
        constexpr const char *key_xTo = "xTo";
        constexpr const char *key_yFrom = "yFrom";
        constexpr const char *key_yTo = "yTo";
        constexpr const char *key_left_handle = "left-handle";
        constexpr const char *key_right_handle = "right-handle";
        constexpr const char *key_type = "type";
        constexpr const char *key_lockX = "lockX";
        constexpr const char *key_lockY = "lockY";
        constexpr const char *key_visible = "visible";
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
    }

    namespace layoutinfo {
        constexpr const char* layout = "layout";
    }

    namespace userdatainfo {
        constexpr const char* optixShowBackground = "optix-show-background";
    }
}

}
