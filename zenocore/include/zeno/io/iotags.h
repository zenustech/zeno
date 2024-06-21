#ifndef __ENUM_IO_H__
#define __ENUM_IO_H__

namespace iotags {
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

#endif