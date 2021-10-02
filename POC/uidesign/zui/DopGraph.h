#pragma once


#include "DopLazy.h"
#include "DopNode.h"


struct DopGraph {
    ztd::Map<std::string, std::unique_ptr<DopNode>> nodes;

    DopNode *add_node(std::string kind);
    std::string _determine_name(std::string kind);

    bool remove_node(DopNode *node);
    void serialize(std::ostream &ss) const;

    static void set_node_input
        ( DopNode *to_node
        , int to_socket_index
        , DopNode *from_node
        , int from_socket_index
        );
    static void remove_node_input
        ( DopNode *to_node
        , int to_socket_index
        );

    std::any resolve_value(std::string expr, std::set<std::string> &visited);

    inline std::any resolve_value(std::string expr) {
        std::set<std::string> visited;
        return resolve_value(expr, visited);
    }
};
