#ifndef __UTILS_DATA_H__
#define __UTILS_DATA_H__

#include "common.h"
#include <string>
#include <map>
#include <vector>
#include <unordered_map>

namespace zeno {

    struct ParamInfo {
        std::string name;
        std::string tooltip;
        std::vector<EdgeInfo> links;
        std::map<std::string, ctrlpropvalue> ctrlprops;
        zvariant defl;
        ParamControl control = ParamControl::Null;
        ParamType type = ParamType::Param_Null;
        SocketProperty prop = SocketProperty::Normal;
    };

    struct NodeData {
        std::string ident;
        std::string name;
        std::string cls;

        std::map<std::string, ParamInfo> inputs;
        std::map<std::string, ParamInfo> outputs;

        //if current node is a subgraph node, which means type =NodeStatus::SubgraphNode.
        GraphData subgraph;

        std::pair<float, float> uipos;
        NodeStatus status = NodeStatus::Null;
        bool bAssetsNode = false;
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

    using NodesData = std::map<std::string, NodeData>;
    using LinksData = std::vector<EdgeInfo>;

    struct GraphData {
        NodesData nodes;
        LinksData links;
    };

    struct NodeDesc {
        std::string name;
        std::vector<ParamInfo> inputs;
        std::vector<ParamInfo> outputs;
        std::vector<std::string> categories;
        bool is_subgraph = false;
    };
    using NodeDescs = std::map<std::string, NodeDesc>;

    struct TimelineInfo {
        int beginFrame = 0;
        int endFrame = 0;
        int currFrame = 0;
        bool bAlways = true;
        int timelinefps = 24;
    };

    using AssetsData = std::map<std::string, GraphData>;

    struct ZSG_PARSE_RESULT {
        GraphData mainGraph;
        ZSG_VERSION iover;
        NodeDescs descs;
        AssetsData assetGraphs;
        TimelineInfo timeline;
    };
}

#endif