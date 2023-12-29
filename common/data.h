#ifndef __UTILS_DATA_H__
#define __UTILS_DATA_H__

#include "common.h"
#include <string>
#include <map>
#include <vector>
#include <unordered_map>
#include <optional>

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

        LinkFunction lnkfunc = Link_Copy;
    };

    struct ParamInfo {
        std::string name;
        std::string tooltip;
        std::vector<EdgeInfo> links;
        zvariant defl;
        ParamControl control = NullControl;
        ParamType type = Param_Null;
        SocketProperty prop = Socket_Normal;
        std::optional<ControlProperty> ctrlProps;
    };

    struct NodeData;

    using NodesData = std::map<std::string, NodeData>;
    using LinksData = std::vector<EdgeInfo>;

    struct GraphData {
        NodesData nodes;
        LinksData links;
    };

    struct NodeData {
        std::string ident;
        std::string name;
        std::string cls;

        std::vector<ParamInfo> inputs;
        std::vector<ParamInfo> outputs;

        //if current node is a subgraph node, which means type =NodeStatus::SubgraphNode.
        std::optional<GraphData> subgraph;

        std::pair<float, float> uipos;
        NodeStatus status = NodeStatus::None;
        bool bAssetsNode = false;
    };



    struct NodeDesc {
        std::string name;
        std::vector<ParamInfo> inputs;
        std::vector<ParamInfo> outputs;
        std::vector<std::string> categories;
        bool is_subgraph = false;
    };
    using NodeDescs = std::map<std::string, NodeDesc>;

    struct GroupInfo
    {
        std::pair<float, float> sz;
        std::string title;
        std::string content;
        //params
        bool special = false;
        std::vector<std::string> items;
        std::string background;     //hex format
    };

    struct TimelineInfo {
        int beginFrame = 0;
        int endFrame = 0;
        int currFrame = 0;
        bool bAlways = true;
        int timelinefps = 24;
    };

    struct ZenoAsset {
        NodeDesc desc;
        GraphData graph;
    };

    using AssetsData = std::map<std::string, ZenoAsset>;

    struct ZSG_PARSE_RESULT {
        GraphData mainGraph;
        ZSG_VERSION iover;
        NodeDescs descs;
        AssetsData assetGraphs;
        TimelineInfo timeline;
    };
}

#endif