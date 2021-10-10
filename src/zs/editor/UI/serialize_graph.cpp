#include <zs/editor/UI/UiDopGraph.h>
#include <zs/editor/UI/UiDopNode.h>
#include <zs/editor/UI/UiDopScene.h>
#include <rapidjson/document.h>


namespace zeno2::UI {

using namespace rapidjson;


static Value dump_string(std::string const &s) {
    Value v;
    v.SetString(s.data(), s.size());
    return v;
}


rapidjson::Value serialize(UiDopGraph const *graph, rapidjson::Document::AllocatorType &alloc) {
    Value v_graph(kObjectType);

    Value v_view(kObjectType);
    v_view.AddMember("scale", Value().SetFloat(graph->scaling), alloc);
    v_view.AddMember("xoffs", Value().SetFloat(graph->translate.x), alloc);
    v_view.AddMember("yoffs", Value().SetFloat(graph->translate.y), alloc);
    v_graph.AddMember("view", v_view, alloc);

    Value v_nodes(kArrayType);
    for (auto const &[_, node]: graph->nodes) {
        Value v_node(kObjectType);
        v_node.AddMember("name", dump_string(node->name), alloc);
        v_node.AddMember("kind", dump_string(node->kind), alloc);
        v_node.AddMember("xpos", Value().SetFloat(node->position.x), alloc);
        v_node.AddMember("ypos", Value().SetFloat(node->position.y), alloc);

        Value v_inputs(kArrayType);
        for (int i = 0; i < node->inputs.size(); i++) {
            auto *socket = node->inputs[i];
            Value v_socket(kObjectType);
            v_socket.AddMember("name", dump_string(socket->name), alloc);
            v_socket.AddMember("value", dump_string(socket->value), alloc);
            v_inputs.PushBack(v_socket, alloc);
        }
        v_node.AddMember("inputs", v_inputs, alloc);

        Value v_outputs(kArrayType);
        for (int i = 0; i < node->inputs.size(); i++) {
            auto *socket = node->inputs[i];
            Value v_socket(kObjectType);
            v_socket.AddMember("name", dump_string(socket->name), alloc);
            v_outputs.PushBack(v_socket, alloc);
        }
        v_node.AddMember("outputs", v_outputs, alloc);

        v_nodes.PushBack(v_node, alloc);
    }
    v_graph.AddMember("nodes", v_nodes, alloc);

    Value v_links(kArrayType);
    for (auto const &link: graph->links) {
        Value v_link(kObjectType);

        auto from_node = link->from_socket->get_parent();
        v_link.AddMember("from_node", dump_string(from_node->name), alloc);
        for (int i = 0; i < from_node->outputs.size(); i++) {
            if (from_node->outputs[i] == link->from_socket) {
                v_link.AddMember("from_socket", Value().SetInt(i), alloc);
                break;
            }
        }

        auto to_node = link->to_socket->get_parent();
        v_link.AddMember("to_node", dump_string(to_node->name), alloc);
        for (int i = 0; i < to_node->inputs.size(); i++) {
            if (to_node->inputs[i] == link->to_socket) {
                v_link.AddMember("to_socket", Value().SetInt(i), alloc);
                break;
            }
        }

        v_links.PushBack(v_link, alloc);
    }
    v_graph.AddMember("links", v_links, alloc);

    return v_graph;
}


template <class Doc>
auto const &load_member(Doc &&doc, const char *name) {
    auto it = doc.FindMember(name);
    if (it == doc.MemberEnd())
        throw ztd::format_error("KeyError: no member named '{}'", name);
    return it->value;
}


void deserialize(UiDopGraph *graph, rapidjson::Value const &v_graph) {
    graph->reset_graph();
    auto v_view = load_member(v_graph, "view").GetObject();
    graph->scaling = load_member(v_view, "scale").GetFloat();
    graph->translate.x = load_member(v_view, "xoffs").GetFloat();
    graph->translate.y = load_member(v_view, "yoffs").GetFloat();

    auto v_nodes = load_member(v_graph, "nodes").GetArray();
    for (auto const &_: v_nodes) {
        auto v_node = _.GetObject();

        auto node = graph->add_child<UiDopNode>();
        node->kind = load_member(v_node, "kind").GetString();
        node->name = load_member(v_node, "name").GetString();
        node->position.x = load_member(v_node, "xpos").GetFloat();
        node->position.y = load_member(v_node, "ypos").GetFloat();

        auto v_inputs = load_member(v_node, "inputs").GetArray();
        for (auto const &_: v_inputs) {
            auto v_socket = _.GetObject();
            auto socket = node->add_input_socket();
            socket->name = load_member(v_socket, "name").GetString();
            socket->value = load_member(v_socket, "value").GetString();
        }

        auto v_outputs = load_member(v_node, "outputs").GetArray();
        for (auto const &_: v_outputs) {
            auto v_socket = _.GetObject();
            auto socket = node->add_output_socket();
            socket->name = load_member(v_socket, "name").GetString();
        }

        node->update_sockets();
        graph->nodes.emplace(node->name, node);
    }

    auto v_links = load_member(v_graph, "links").GetArray();
    for (auto const &_: v_links) {
        auto v_link = _.GetObject();

        auto from_node = load_member(v_link, "from_node").GetString();
        auto from_socket = load_member(v_link, "from_socket").GetInt();
        auto to_node = load_member(v_link, "to_node").GetString();
        auto to_socket = load_member(v_link, "to_socket").GetInt();

        auto f_node = graph->nodes.at(from_node);
        auto f_sock = f_node->outputs.at(from_socket);
        auto t_node = graph->nodes.at(to_node);
        auto t_sock = t_node->inputs.at(to_socket);

        graph->add_link(f_sock, t_sock);
    }
}


}
