#ifndef __UTILS_DATA_H__
#define __UTILS_DATA_H__

#include "common.h"

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
        NodeType type = NodeType::Normal;
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
}

#endif