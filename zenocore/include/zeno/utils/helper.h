#ifndef __HELPER_H__
#define __HELPER_H__

#include <rapidjson/document.h>
#include <common/data.h>
#include <zeno/utils/string.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/log.h>

namespace zeno {

    ParamType convertToType(std::string const& type);
    zvariant str2var(std::string const& defl, ParamType const& type);
    zany strToZAny(std::string const& defl, ParamType const& type);
}


#endif