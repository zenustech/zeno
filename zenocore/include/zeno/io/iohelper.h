#ifndef __IO_HELPER_H__
#define __IO_HELPER_H__

#include <zeno/utils/uuid.h>
#include <zeno/core/data.h>
#include <zeno/io/iocommon.h>

namespace zenoio {

    zeno::GraphData fork(
        const std::map<std::string, zeno::GraphData>& sharedSubg,
        const std::string& subnetName);

    zeno::zvariant jsonValueToZVar(const rapidjson::Value& val, zeno::ParamType const& type);
    zeno::reflect::Any jsonValueToAny(const rapidjson::Value& val, zeno::ParamType const& type);

    void writeZVariant(zeno::zvariant var, zeno::ParamType type, RAPIDJSON_WRITER& writer);
    void writeAny(const zeno::reflect::Any& any, zeno::ParamType type, RAPIDJSON_WRITER& writer);
    bool importControl(const rapidjson::Value& controlObj, zeno::ParamControl& ctrl, zeno::ControlProperty& props);
    void dumpControl(zeno::ParamType type, zeno::ParamControl ctrl, std::optional<zeno::ControlProperty> ctrlProps, RAPIDJSON_WRITER& writer);
    ZENO_API zeno::ZSG_VERSION getVersion(const std::string& fn);
    zeno::SocketType getSocketTypeByDesc(const std::string& desc);
}

#endif