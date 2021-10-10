#include <zeno2/UI/UiDopGraph.h>
#include <zeno2/UI/UiDopNode.h>
#include <zeno2/UI/UiDopScene.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>


namespace zeno2::UI {

using namespace rapidjson;


Value dump_string(std::string const &s) {
    Value v(kStringType);
    v.SetString(s.data(), s.size());
    return v;
}


std::string UiDopGraph::serialize_graph() {
    Document doc(kObjectType);
    doc.AddMember("version", dump_string("v2.0"), doc.GetAllocator());

    Value v_nodes(kArrayType);
    for (auto const &[_, node]: nodes) {
        Value v_node(kObjectType);
        v_node.AddMember("name", dump_string(node->name), doc.GetAllocator());
        v_node.AddMember("kind", dump_string(node->kind), doc.GetAllocator());
        v_node.AddMember("xpos", Value().SetFloat(node->position.x), doc.GetAllocator());
        v_node.AddMember("ypos", Value().SetFloat(node->position.y), doc.GetAllocator());

        Value v_inputs(kArrayType);
        for (int i = 0; i < node->inputs.size(); i++) {
            auto *socket = node->inputs[i];
            Value v_socket(kObjectType);
            v_socket.AddMember("name", dump_string(socket->name), doc.GetAllocator());
            v_socket.AddMember("value", dump_string(socket->value), doc.GetAllocator());
            v_inputs.PushBack(v_socket, doc.GetAllocator());
        }
        v_node.AddMember("inputs", v_inputs, doc.GetAllocator());

        Value v_outputs(kArrayType);
        for (int i = 0; i < node->inputs.size(); i++) {
            auto *socket = node->inputs[i];
            Value v_socket(kObjectType);
            v_socket.AddMember("name", dump_string(socket->name), doc.GetAllocator());
            v_outputs.PushBack(v_socket, doc.GetAllocator());
        }
        v_node.AddMember("outputs", v_outputs, doc.GetAllocator());

        v_nodes.PushBack(v_node, doc.GetAllocator());
    }
    doc.AddMember("nodes", v_nodes, doc.GetAllocator());

    Value v_links(kArrayType);
    for (auto const &link: links) {
        Value v_link(kObjectType);

        auto from_node = link->from_socket->get_parent();
        v_link.AddMember("from_node", dump_string(from_node->name), doc.GetAllocator());
        for (int i = 0; i < from_node->outputs.size(); i++) {
            if (from_node->outputs[i] == link->from_socket) {
                v_link.AddMember("from_socket", Value().SetInt(i), doc.GetAllocator());
                break;
            }
        }

        auto to_node = link->to_socket->get_parent();
        v_link.AddMember("to_node", dump_string(to_node->name), doc.GetAllocator());
        for (int i = 0; i < to_node->inputs.size(); i++) {
            if (to_node->inputs[i] == link->to_socket) {
                v_link.AddMember("to_socket", Value().SetInt(i), doc.GetAllocator());
                break;
            }
        }

        v_links.PushBack(v_link, doc.GetAllocator());
    }

    StringBuffer buffer;
    PrettyWriter writer(buffer);
    doc.Accept(writer);
    std::string str = buffer.GetString();
    SPDLOG_INFO("serialize_graph result:\n{}", str);
    return str;
}


template <class Doc>
auto const &load_member(Doc &&doc, const char *name) {
    auto it = doc.FindMember(name);
    if (it == doc.MemberEnd())
        throw ztd::format_error("KeyError: no member named '{}'", name);
    return it->value;
}


void UiDopGraph::deserialize_graph(std::string const &buffer) {
    Document doc;
    doc.Parse(buffer.c_str());

    nodes.clear();
    auto v_nodes = load_member(doc, "nodes").GetArray();
    for (auto const &_: v_nodes) {
        auto v_node = _.GetObject();

        auto node = add_child<UiDopNode>();
        node->kind = load_member(v_node, "kind").GetString();
        node->name = load_member(v_node, "name").GetString();
        node->position.x = load_member(v_node, "xpos").GetFloat();
        node->position.y = load_member(v_node, "xpos").GetFloat();

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
        nodes.emplace(node->name, node);
    }

    auto v_links = load_member(doc, "links").GetArray();
    for (auto const &_: v_links) {
        auto v_link = _.GetObject();

        auto from_node = load_member(v_link, "from_node").GetString();
        auto from_socket = load_member(v_link, "from_socket").GetInt();
        auto to_node = load_member(v_link, "to_node").GetString();
        auto to_socket = load_member(v_link, "to_socket").GetInt();

        auto f_node = nodes.at(from_node);
        auto f_sock = f_node->outputs.at(from_socket);
        auto t_node = nodes.at(to_node);
        auto t_sock = t_node->inputs.at(to_socket);

        add_link(f_sock, t_sock);
    }

    SPDLOG_INFO("deserialize_graph done, {} nodes loaded", nodes.size());
}


}
