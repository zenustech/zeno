#pragma once

#include <zeno/zeno.h>
#include <rapidjson/document.h>

namespace zeno {

zany parseObjectFromUi(rapidjson::Value const &x);

namespace iotags {
    constexpr const char *key_objectType = "objectType";

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
    }
}

}
