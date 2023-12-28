#ifndef __IO_HELPER_H__
#define __IO_HELPER_H__

#include <common/data.h>
#include <zeno/utils/uuid.h>

namespace zenoio {

    zeno::GraphData fork(
        const std::string& currentPath,
        const zeno::AssetsData& subgraphDatas,
        const std::string& subnetName);

    zeno::zvariant jsonValueToZVar(const rapidjson::Value& val, zeno::ParamType const& type);

    bool importControl(const rapidjson::Value& controlObj, zeno::ParamControl& ctrl, zeno::ControlProperty& props);
}

#endif