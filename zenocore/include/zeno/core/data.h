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

namespace zeno {

    struct ControlProperty
    {
        std::optional<std::vector<std::string>> items;  //for combobox
        std::optional<std::array<float, 3>> ranges;       //min, max, step
    };

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
    };

    //primitive
    struct ParamPrimitive {
        std::string name;
        ParamType type = Param_Null;
        SocketType socketType = Socket_Primitve;
        zvariant defl;
        zvariant result;    //run result.
        ParamControl control = NullControl;
        std::optional<ControlProperty> ctrlProps;

        std::vector<EdgeInfo> links;
        SocketProperty prop = Socket_Normal;

        std::string tooltip;
        bool bInput = true;
        bool bVisible = true;

        ParamPrimitive() {}
        ParamPrimitive(std::string name, ParamType type, SocketType sockType, std::string tooltip = "")
            : name(name)
            , type(type)
            , socketType(sockType)
            , tooltip(tooltip)
        {
        }
        ParamPrimitive(std::string name, ParamType type, SocketType sockType, zvariant defl, ParamControl ctrl, ControlProperty props, std::string tooltip = "")
            : name(name)
            , type(type)
            , socketType(sockType)
            , defl(defl)
            , control(ctrl)
            , ctrlProps(props)
            , tooltip(tooltip)
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
}

#endif