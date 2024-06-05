#ifndef __HELPER_H__
#define __HELPER_H__

#include <rapidjson/document.h>
#include <zeno/core/data.h>
#include <zeno/utils/string.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/CoreParam.h>

namespace zeno {

    ZENO_API ParamType convertToType(std::string const& type);
    ZENO_API std::string paramTypeToString(ParamType type);
    ZENO_API zvariant str2var(std::string const& defl, ParamType const& type);
    ZENO_API zvariant initDeflValue(ParamType const& type);
    ZENO_API std::string getControlDesc(zeno::ParamControl ctrl, zeno::ParamType type);
    ZENO_API zeno::ParamControl getDefaultControl(const zeno::ParamType type);
    bool isEqual(const zvariant& lhs, const zvariant& rhs, ParamType const type);
    zany strToZAny(std::string const& defl, ParamType const& type);
    EdgeInfo getEdgeInfo(std::shared_ptr<ObjectLink> spLink);
    EdgeInfo getEdgeInfo(std::shared_ptr<PrimitiveLink> spLink);
    std::string generateObjKey(std::shared_ptr<IObject> spObject);
    ZENO_API std::string objPathToStr(ObjPath path);
    ObjPath strToObjPath(const std::string& str);
    bool getParamInfo(const CustomUI& customui, std::vector<ParamPrimitive>& inputs, std::vector<ParamPrimitive>& outputs);
    bool isPrimitiveType(const zeno::ParamType type);
    CustomUI descToCustomui(const Descriptor& desc);
    ZENO_API PrimitiveParams customUiToParams(const CustomUIParams& customparams);
    void initControlsByType(CustomUI& ui);
    std::string absolutePath(std::string currentPath, const std::string& path);
    std::string relativePath(std::string currentPath, const std::string& path);
    std::set<std::string> getReferPath(const std::string& path);
    std::set<std::string> getReferPaths(const zvariant& val);
    formula_tip_info getNodesByPath(const std::string& nodeabspath, const std::string& graphpath, const std::string& prefix);
}


#endif