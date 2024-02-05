#ifndef __HELPER_H__
#define __HELPER_H__

#include <rapidjson/document.h>
#include <zeno/core/data.h>
#include <zeno/utils/string.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/IParam.h>

namespace zeno {

    ZENO_API ParamType convertToType(std::string const& type);
    ZENO_API std::string paramTypeToString(ParamType type);
    ZENO_API zvariant str2var(std::string const& defl, ParamType const& type);
    ZENO_API std::string getControlDesc(zeno::ParamControl ctrl, zeno::ParamType type);
    ZENO_API zeno::ParamControl getDefaultControl(const zeno::ParamType type);
    bool isEqual(const zvariant& lhs, const zvariant& rhs, ParamType const type);
    zany strToZAny(std::string const& defl, ParamType const& type);
    EdgeInfo getEdgeInfo(std::shared_ptr<ILink> spLink);
}


#endif