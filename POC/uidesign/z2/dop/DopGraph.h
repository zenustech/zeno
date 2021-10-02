#pragma once


#include <z2/dop/DopNode.h>


namespace z2::dop {


struct DopGraph {
    ztd::map<std::string, std::unique_ptr<DopNode>> nodes;

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

    std::any resolve_value(std::string expr, DopContext *visited);

    inline std::any resolve_value(std::string expr) {
        DopContext visited;
        return resolve_value(expr, &visited);
    }
};


}  // namespace z2::dop
