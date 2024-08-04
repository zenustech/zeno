#ifndef __UTILS_DATA_H__
#define __UTILS_DATA_H__

#include <zeno/core/common.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <unordered_map>
#include <optional>
#include <zeno/utils/api.h>
#include <reflect/container/any>
#include <reflect/type.hpp>
#include <zeno/types/CurveObject.h>


namespace zeno {

    struct EdgeInfo {
        std::string outNode;
        std::string outParam;
        std::string outKey;

        std::string inNode;
        std::string inParam;
        std::string inKey;

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

    struct ParamObject {
        std::string name;
        ParamType type = Param_Null;
        SocketType socketType = Socket_Clone;
        std::vector<EdgeInfo> links;
        SocketProperty prop = Socket_Normal;
        std::string tooltip;
        bool bInput = true;
        std::string wildCardGroup;
    };

    //primitive
    struct ParamPrimitive {
        std::string name;
        ParamType type = Param_Null;
        SocketType socketType = Socket_Primitve;
        zeno::reflect::Any defl;
        zeno::reflect::Any result;    //run result.
        ParamControl control = NullControl;
        zeno::reflect::Any ctrlProps;

        std::vector<EdgeInfo> links;
        SocketProperty prop = Socket_Normal;

        std::string tooltip;
        bool bInput = true;
        bool bVisible = false;
        std::string wildCardGroup;
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

    struct CustomUIParams {
        std::vector<ParamTab> tabs;   //custom ui for input primitive params
    };

    //CustomUI is structure for input params of primitive types, like vec3f int string, etc.
    struct CustomUI {
        ObjectParams inputObjs;
        CustomUIParams inputPrims;
        PrimitiveParams outputPrims;
        ObjectParams outputObjs;

        std::string category;
        std::string nickname;
        std::string iconResPath;
        std::string doc;
    };

    struct ParamUpdateInfo {
        std::variant<zeno::ParamPrimitive, zeno::ParamObject> param;
        std::string oldName;
    };

    using ParamsUpdateInfo = std::vector<ParamUpdateInfo>;

    struct NodeData;

    using NodesData = std::map<std::string, NodeData>;
    using LinksData = std::vector<EdgeInfo>;

    struct GraphData {
        std::string name;   //the name like "subnet1", "subnet2", not a template name.
        std::string templateName;
        NodesData nodes;
        LinksData links;
        SubnetType type;
    };

    struct AssetInfo
    {
        std::string name;
        std::string path;
        int majorVer;
        int minorVer;
    };

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
    };

    struct NodeDesc {
        std::string name;
        std::vector<ParamPrimitive> inputs;
        std::vector<ParamPrimitive> outputs;
        std::vector<std::string> categories;
        bool is_subgraph = false;
    };
    using NodeDescs = std::map<std::string, NodeDesc>;

    using NodeCates = std::map<std::string, std::vector<std::string>>;

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
}

#endif