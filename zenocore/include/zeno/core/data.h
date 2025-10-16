#ifndef __UTILS_DATA_H__
#define __UTILS_DATA_H__

#include <zeno/core/common.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <unordered_map>
#include <memory>
#include <optional>
#include <zeno/utils/api.h>
#include <reflect/container/any>
#include <reflect/type.hpp>
#include <zeno/types/CurveObject.h>
#include <zeno/core/IObject.h>
#include <zeno/extra/GlobalError.h>


namespace zeno {

    struct EdgeInfo {
        std::string outNode;
        std::string outParam;
        std::string outKey;

        std::string inNode;
        std::string inParam;
        std::string inKey;

        std::string targetParam;

        bool bObjLink = true;

        bool operator==(const EdgeInfo& rhs) const {
            return outNode == rhs.outNode && outParam == rhs.outParam && outKey == rhs.outKey &&
                inNode == rhs.inNode && inParam == rhs.inParam && inKey == rhs.inKey &&
                bObjLink == rhs.bObjLink;
        }
        bool operator<(const EdgeInfo& rhs) const {

            if (outNode != rhs.outNode) {
                return outNode < rhs.outNode;
            }
            else if (outParam != rhs.outParam) {
                return outParam < rhs.outParam;
            }
            else if (outKey != rhs.outKey) {
                return outKey < rhs.outKey;
            }
            else if (inNode != rhs.inNode) {
                return inNode < rhs.inNode;
            }
            else if (inParam != rhs.inParam) {
                return inParam < rhs.inParam;
            }
            else if (inKey != rhs.inKey) {
                return inKey < rhs.inKey;
            }
            else {
                return false;
            }
        }
    };

    struct CommonParam {
        std::string name;
        ParamType type = Param_Null;
        SocketType socketType = Socket_Clone;    //为了兼容zeno老版本的使用习惯，clone可能有内存性能风险，owning可能概念不明
        SocketProperty sockProp = Socket_Normal;
        std::string wildCardGroup;
        std::string constrain;
        std::string tooltip;
        std::vector<EdgeInfo> links;

        bool bInput = true;
        bool bSocketVisible = false;
        bool bVisible = true;       //在参数面板是否可见
        bool bEnable = true;        //在参数面板是否可用
        bool bWildcard = false;     //是否为wildcard参数

        CommonParam() {}
        CommonParam(std::string name) : name(name) {}
        CommonParam(std::string name, SocketType socketType, std::string wildCardGroup = "", std::string constrain = "", std::string tooltip = "")
            : name(name)
            , socketType(socketType)
            , wildCardGroup(wildCardGroup)
            , constrain(constrain)
            , tooltip(tooltip)
        {
        }
        CommonParam(std::string name, ParamType type, std::string constrain = "", std::string tooltip = "", bool socketVisible = false)
            : name(name)
            , type(type)
            , constrain(constrain)
            , tooltip(tooltip)
            , bSocketVisible(socketVisible)
        {
        }
        CommonParam(std::string name, SocketType socketType, SocketProperty sockProp)
            : name(name)
            , socketType(socketType)
            , sockProp(sockProp)
        {
        }
    };

    struct ParamObject : CommonParam {
        ParamObject() {}
        ParamObject(std::string name) : CommonParam(name) {}
        ParamObject(std::string name, ParamType type, std::string constrain = "", std::string tooltip = "")
            : CommonParam(name, type, constrain, tooltip) {}
        ParamObject(std::string name, SocketType socketType)
            : CommonParam(name, socketType) {}
        ParamObject(std::string name, SocketType socketType, std::string wildGrp, std::string constrain)
            : CommonParam(name, socketType, wildGrp, constrain) {}
    };

    //primitive
    struct ParamPrimitive : CommonParam {
        zeno::reflect::Any defl;
        zeno::reflect::Any result;    //run result.
        ParamControl control = NullControl;
        zeno::reflect::Any ctrlProps;

        ParamPrimitive() {}
        ParamPrimitive(std::string name, zeno::reflect::Any defl = zeno::reflect::Any(), ParamControl control = NullControl, zeno::reflect::Any ctrlProps = zeno::reflect::Any(), std::string wildcardGrp = "", std::string constrain = "")
            : CommonParam(name, Socket_Primitve, wildcardGrp, constrain)
            , defl(defl)
            , control(control)
            , ctrlProps(ctrlProps)
        {}
        ParamPrimitive(std::string name, ParamType type, zeno::reflect::Any defl = zeno::reflect::Any(), ParamControl control = NullControl, zeno::reflect::Any ctrlProps = zeno::reflect::Any(), std::string constrain = "", bool socketVisible = false)
            : CommonParam(name, type, constrain, "", socketVisible)
            , defl(defl)
            , control(control)
            , ctrlProps(ctrlProps)
        {
        }
    };

    using ObjectParams = std::vector<ParamObject>;
    using PrimitiveParams = std::vector<ParamPrimitive>;

    struct ParamGroup {
        std::string name = "Group1";
        PrimitiveParams params;
    };

    struct ParamTab {
        std::string name = "Tab1";
        std::vector<ParamGroup> groups;
    };

    using CustomUIParams = std::vector<ParamTab>;   //custom ui for input primitive params

    struct NodeUIStyle {
        std::string iconResPath;        //是否可以直接用svg的字符串，这样不用管理资源文件
        std::string background;
    };

    //CustomUI is structure for input params of primitive types, like vec3f int string, etc.
    struct CustomUI {
        ObjectParams inputObjs;
        CustomUIParams inputPrims;
        PrimitiveParams outputPrims;
        ObjectParams outputObjs;

        NodeUIStyle uistyle;

        std::string category;
        std::string nickname;
        std::string doc;

        CustomUI() = default;
        CustomUI(
            const ObjectParams& inObjs,
            const CustomUIParams& inPrims,
            const PrimitiveParams& outPrims,
            const ObjectParams& outObjs,
            const NodeUIStyle& uisty,
            const std::string& _category,
            const std::string& _nickname,
            const std::string& _doc
            )
            : inputObjs(inObjs)
            , inputPrims(inPrims)
            , outputPrims(outPrims)
            , outputObjs(outObjs)
            , uistyle(uisty)
            , category(_category)
            , nickname(_nickname)
            , doc(_doc)
        {}
    };

    struct ParamUpdateInfo {
        std::variant<zeno::ParamPrimitive, zeno::ParamObject> param;
        std::string oldName;
    };

    using ParamsUpdateInfo = std::vector<ParamUpdateInfo>;

    struct NodeData;

    using NodesData = std::map<std::string, NodeData>;
    using LinksData = std::vector<EdgeInfo>;
    using ReferencesData = std::map<std::string, std::set<std::string>>;    /* refer-node-name, set(params) */

    struct GraphData {
        std::string name;   //the name like "subnet1", "subnet2", not a template name.
        std::string templateName;
        NodesData nodes;
        LinksData links;
        ReferencesData references;  //potenial reference node.
        SubnetType type;
    };

    struct AssetInfo
    {
        std::string name;
        std::string path;
        int majorVer;
        int minorVer;
    };

    struct ZenoAsset {
        ObjectParams object_inputs;
        PrimitiveParams primitive_inputs;
        PrimitiveParams primitive_outputs;
        ObjectParams object_outputs;
        AssetInfo info;
        CustomUI m_customui;
        std::optional<GraphData> optGraph;
    };

    using AssetsData = std::map<std::string, ZenoAsset>;

    struct GroupInfo
    {
        vec2f sz;
        std::string title;
        std::string content;
        //params
        bool special = false;
        std::vector<std::string> items;
        vec3f background;     //hex format
    };

    struct NodeData {
        std::string name;
        std::string cls;

        CustomUI customUi;   //custom ui for input params,just a template or mapping of input params data.

        //if current node is a subgraph node, which means type =NodeStatus::SubgraphNode.
        std::optional<GraphData> subgraph;
        std::optional<AssetInfo> asset;
        std::optional<GroupInfo> group;

        std::pair<float, float> uipos;
        bool bView = false;
        NodeType type;
        bool bCollasped = false;
        bool bLocked = true;   //asset instance.
        bool bypass = false;
        bool bnocache = false;
        bool bclearsbn = false;
    };

    struct NodeDesc {
        std::string name;
        std::vector<ParamPrimitive> inputs;
        std::vector<ParamPrimitive> outputs;
        std::vector<std::string> categories;
        bool is_subgraph = false;
    };
    using NodeDescs = std::map<std::string, NodeDesc>;

    enum LoadStatus
    {
        ZModule_Loaded,
        ZModule_UnLoaded
    };

    struct NodeInfo
    {
        std::string name;           //node class name
        std::string module_path;    //比如zenvdb.dll, 或者aaa.zda
        std::string cate;
        LoadStatus status = ZModule_Loaded;
    };

    //using NodeCates = std::map<std::string, std::vector<NodeInfo>>;
    using NodeRegistry = std::vector<NodeInfo>;

    struct TimelineInfo {
        int beginFrame = 0;
        int endFrame = 100;
        int currFrame = 0;
        bool bAlways = true;
        int timelinefps = 24;
    };

    struct params_change_info {
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::set<std::string> new_inputs;
        std::set<std::string> new_outputs;
        std::set<std::pair<std::string, std::string>> rename_inputs;
        std::set<std::pair<std::string, std::string>> rename_outputs;    //pair: <old_name, new_name>
        std::set<std::string> remove_inputs;
        std::set<std::string> remove_outputs;
    };

    struct HeatmapData {
        std::vector<float> facs;
        std::vector<zeno::vec3f> colors;

        bool operator==(const HeatmapData& rhs) {
            return rhs.facs == facs && rhs.colors == colors;
        }

        std::vector<zeno::vec3f> toVecColors(int nres) const {
            std::vector<zeno::vec3f> clrs;
            for (int i = 0; i < nres; i++) {
                float fac = i * (1.f / nres);
                zeno::vec3f clr;
                for (int j = 0; j < colors.size(); j++) {
                    float f = facs[j];
                    auto rgb = colors[j];
                    if (f >= fac) {
                        if (j != 0) {
                            float last_f = facs[j - 1];
                            auto last_rgb = colors[j - 1];
                            auto intfac = (fac - last_f) / (f - last_f);
                            //printf("%f %f %f %f\n", fac, last_f, f, intfac);
                            clr = zeno::mix(last_rgb, rgb, intfac);
                        }
                        else {
                            clr = rgb;
                        }
                        break;
                    }
                }
                clrs.push_back(clr);
            }
            return clrs;
        }
    };

    template<typename T, typename E>
    T zeno_get(const E& container) {
        if constexpr (std::is_same_v<E, zvariant>) {
            return std::visit([](auto&& val)->T {
                using V = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<V, T>) {
                    return val;
                }
                else {
                    throw;
                }
                }, container);
        }
        else if constexpr (std::is_same_v<E, zeno::reflect::Any>) {
            if (zeno::reflect::get_type<T>() == container.type()) {
                return zeno::reflect::any_cast<T>(container);
            }
            else {
                throw;
            }
        }
        else {
            throw;
        }
    }

    template<typename T>
    const T* any_get_if(const zeno::reflect::Any& val) {
        if (zeno::reflect::get_type<T>() == val.type()) {
            return zeno::reflect::any_cast<T>(&val);
        }
        else {
            return nullptr;
        }
    }

    template<typename T, typename E>
    bool zeno_get_if(const T& container, E& ret) {
        if constexpr (std::is_same_v<T, zvariant>) {
            return std::visit([&](auto&& val)->bool {
                using V = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<V, E>) {
                    ret = val;
                    return true;
                }
                else {
                    return false;
                }
                }, container);
        }
        else if constexpr (std::is_same_v<T, zeno::reflect::Any>) {
            if (zeno::reflect::get_type<E>() == container.type()) {
                ret = zeno::reflect::any_cast<E>(container);
                return true;
            }
            else {
                return false;
            }
        }
        else {
            return false;
        }
    }

    enum formula_tip_type {
        FMLA_NO_MATCH,
        FMLA_TIP_FUNC_CANDIDATES,   //function candidates.
        FMLA_TIP_FUNC_ARGS,
        FMLA_TIP_REFERENCE,
    };

    struct ARG_INFO
    {
        std::string name;
        std::string type;
    };

    struct FUNC_INFO
    {
        std::string name;
        std::string tip;
        std::string rettype;
        std::vector<ARG_INFO> args;
    };

    struct formula_args_tip {
        FUNC_INFO func;
        int argidx = -1;
    };

    struct ref_tip_info {
        std::string nodename;
        std::string iconres;
    };

    struct formula_tip_info {
        std::string prefix;
        formula_tip_type type;
        std::vector<std::string> func_candidats;
        formula_args_tip func_args;
        std::vector<ref_tip_info> ref_candidates;
    };

    struct TriangulateInfo
    {
        /*in*/ bool with_uv = true;
        /*in*/ bool has_lines = true;
        /*in*/ bool with_attr = true;
        /*out*/std::vector<vec3i> tris;
        /*out*/std::vector<vec2i> lines;
        /*out*/std::vector<vec3f> uv;
        /*out*/std::vector<vec3f> uv0;
        /*out*/std::vector<vec3f> uv1;
        /*out*/std::vector<vec3f> uv2;
    };

    //可以是公式，曲线，数值，字符串中的一种，适用于所有数值类型和向量类型
    using PrimVar = std::variant<int, float, std::string, CurveData>;
    using vecvar = std::vector<zeno::PrimVar>;
    using ParamPath = std::string;  //uuidpath

    using NumericValue = std::variant<
        int, zeno::vec2i, zeno::vec3i, zeno::vec4i,
        float, zeno::vec2f, zeno::vec3f, zeno::vec4f>;

    using ShaderNodePath = std::string;

    struct ShaderData
    {
        ParamPath curr_param;
        std::variant<NumericValue, ShaderNodePath> data;
    };

    using SharedObjects = std::map<std::string, std::shared_ptr<zeno::IObject>>;

    struct RenderObjsInfo {
        SharedObjects newObjs;
        SharedObjects modifyObjs;
        SharedObjects remObjs;
        SharedObjects lightObjs;    //TODO:
        SharedObjects allObjects;

        bool empty() const {
            return newObjs.empty() && modifyObjs.empty() && remObjs.empty() && lightObjs.empty();
        }
    };

    struct NodeImpl;

    struct ObjectNodeInfo {
        std::optional<IObject*> rootObj;        //list/dict case.
        IObject* transformingObj;
        NodeImpl* spViewNode;
    };

    //一条计算链路的更新信息：
    struct render_update_info {
        UpdateReason reason = Update_Unknown;
        std::string uuidpath_node_objkey;   //节点的uuid路径，同时也是obj的key.
        zany spObject;  //对象所属权可藉此传送到渲染端
        std::vector<std::string> remove_objs;

        render_update_info() {}
        render_update_info(const render_update_info& rhs) {
            reason = rhs.reason;
            uuidpath_node_objkey = rhs.uuidpath_node_objkey;
            if (rhs.spObject)
                spObject = rhs.spObject->clone();
            remove_objs = rhs.remove_objs;
        }

        render_update_info& operator=(const render_update_info& rhs) {
            reason = rhs.reason;
            uuidpath_node_objkey = rhs.uuidpath_node_objkey;
            if (rhs.spObject)
                spObject = rhs.spObject->clone();
            remove_objs = rhs.remove_objs;
            return *this;
        }
    };

    struct render_reload_info {
        render_reload_policy policy;
        std::string current_ui_graph;   //当前用户在编辑器端的ui图层，以普通路径表达
        std::vector<render_update_info> objs;
        GlobalError error;
    };
}

#endif