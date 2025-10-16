#pragma once

#ifndef __ZPYGRAPH_H__
#define __ZPYGRAPH_H__

#include <string>
#include <zeno/core/Graph.h>
#include "zpynode.h"
#include "zpyobject.h"

class Zpy_Graph {
public:
    Zpy_Graph(std::shared_ptr<zeno::Graph> graph);
    Zpy_Node createNode(const std::string& nodecls, const std::string& originalname = "", bool bAssets = false);
    Zpy_Node getNode(const std::string& name);
    Zpy_Object getInputObject(const std::string& node_name, const std::string& param);
    void removeNode(const std::string& name);
    std::string getName() const;
    void addEdge(const std::string& out_node, const std::string& out_param,
        const std::string& in_node, const std::string& in_param);
    void removeEdge(const std::string& out_node, const std::string& out_param,
        const std::string& in_node, const std::string& in_param);

    py::object getCamera() const;
    py::object getLight() const;

private:
    std::weak_ptr<zeno::Graph> m_wpGraph;
};

#endif