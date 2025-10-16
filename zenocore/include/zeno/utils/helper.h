#ifndef __HELPER_H__
#define __HELPER_H__

#include <rapidjson/document.h>
#include <zeno/core/data.h>
#include <zeno/core/coredata.h>
#include <zeno/utils/string.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/CoreParam.h>
#include <zeno/core/reflectdef.h>
#include <reflect/container/any>
#include <tinygltf/json.hpp>
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno {

    struct DictObject;
    struct ListObject;
    struct SubnetNode;

    ZENO_API ParamType convertToType(std::string const& type, const std::string_view& param_name = "");
    ZENO_API bool isAnyEqual(const zeno::reflect::Any& lhs, const zeno::reflect::Any& rhs);
    ZENO_API std::string paramTypeToString(ParamType type);
    ZENO_API std::string any_cast_to_string(const Any& value);
    ZENO_API zvariant str2var(std::string const& defl, ParamType const& type);
    ZENO_API zeno::reflect::Any str2any(std::string const& defl, ParamType const& type);
    ZENO_API bool convertToEditVar(zeno::reflect::Any& var, const ParamType type);
    ZENO_API bool convertToOriginalVar(zeno::reflect::Any& editvar, const ParamType type);
    ZENO_API bool ListHasPrimObj(zeno::ListObject* list);
    ZENO_API bool DictHasPrimObj(zeno::DictObject* dict);
    ZENO_API zvariant initDeflValue(ParamType type);
    ZENO_API zeno::reflect::Any initAnyDeflValue(ParamType const& type);
    ZENO_API zvariant AnyToZVariant(zeno::reflect::Any const& var);
    ZENO_API NumericValue AnyToNumeric(const zeno::reflect::Any& any, bool& bSucceed);
    ZENO_API std::string getControlDesc(zeno::ParamControl ctrl, zeno::ParamType type);
    ZENO_API zeno::ParamControl getDefaultControl(const zeno::ParamType type);
    ZENO_API std::string editVariantToStr(const PrimVar& var);
    ZENO_API std::vector<zany> fromZenCache(const std::string& cachedir, int frameid);
    ZENO_API std::wstring s2ws();
    ZENO_API void merge_json(nlohmann::json& target, const nlohmann::json& source);
    bool isEqual(const zvariant& lhs, const zvariant& rhs, ParamType const type);
    zany strToZAny(std::string const& defl, ParamType const& type);
    EdgeInfo getEdgeInfo(std::shared_ptr<ObjectLink> spLink);
    EdgeInfo getEdgeInfo(std::shared_ptr<PrimitiveLink> spLink);
    std::string generateObjKey(std::shared_ptr<IObject> spObject);
    std::string uniqueName(std::string prefix, std::set<std::string> existing);
    ZENO_API std::string objPathToStr(ObjPath path);
    ObjPath strToObjPath(const std::string& str);
    bool getParamInfo(const CustomUI& customui, std::vector<ParamPrimitive>& inputs, std::vector<ParamPrimitive>& outputs);
    zeno::ParamType findParamType(const CustomUI& customui, bool bInput, const std::string& name);
    bool isPrimitiveType(const ParamType type);
    ZENO_API PrimitiveParams customUiToParams(const CustomUIParams& customparams);
    ZENO_API void parseUpdateInfo(const CustomUI& customui, ParamsUpdateInfo& infos);
    void initControlsByType(CustomUI& ui);
    std::string absolutePath(std::string currentPath, const std::string& path);
    std::string relativePath(std::string currentPath, const std::string& path);
    std::set<std::string> getReferPath(const std::string& path);
    std::set<std::string> getReferPaths(const zvariant& val);
    formula_tip_info getNodesByPath(const std::string& nodeabspath, const std::string& graphpath, const std::string& prefix);

    bool isObjectType(const zeno::reflect::RTTITypeInfo& type, bool& isConstPtr);
    bool isObjectType(ParamType type);
    bool isNumericType(zeno::ParamType type);
    bool isNumericVecType(zeno::ParamType type);
    bool isSameDimensionNumericVecType(zeno::ParamType left, zeno::ParamType right);
    ZENO_API bool outParamTypeCanConvertInParamType(zeno::ParamType outType, zeno::ParamType inType, NodeDataGroup outGroup, NodeDataGroup inGroup);
    ZENO_API bool isPrimVarType(zeno::ParamType type);
    ZENO_API zeno::reflect::Any convertNumericAnyType(zeno::ParamType outType, zeno::ParamType inType, zeno::reflect::Any outputVal);

    void getNameMappingFromReflectUI(
        zeno::reflect::TypeBase* typeBase,
        NodeImpl* node,
        std::map<std::string, std::string>& inputParams,
        std::vector<std::string>& outputParams);
    ZENO_API bool isDerivedFromSubnetNodeName(const std::string& clsname);    //判断clsname是否为继承自subnetNode的节点

    ZENO_API zeno::SubnetNode* getSubnetNode(zeno::NodeImpl* pAdapter);

    //变量传播dirty相关
    ZENO_API void propagateDirty(NodeImpl* spCurrNode, std::string varName);
    void getUpstreamNodes(NodeImpl* spCurrNode, std::set<ObjPath>& upstreamDepNodes, std::set<ObjPath>& upstreams, std::string outParamName = "");
    void mark_dirty_by_dependNodes(NodeImpl* spCurrNode, bool bOn, std::set<ObjPath> nodesRange, std::string inParamName = "");
    bool isSubnetInputOutputParam(NodeImpl* spParentnode, std::string paramName);

    void update_list_root_key(ListObject* listobj, const std::string& key);
    void update_dict_root_key(DictObject* dictobj, const std::string& key);

    zany clone_by_key(IObject* pObject, const std::string& prefix);
    void add_prefix_key(IObject* pObject, const std::string& prefix);
    ZENO_API std::vector<std::string> get_obj_paths(IObject* pObject);

    AttrVar abiAnyToAttrVar(const zeno::reflect::Any& anyval);
}


#endif