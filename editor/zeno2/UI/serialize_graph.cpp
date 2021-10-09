#include <zeno2/UI/UiDopGraph.h>
#include <zeno2/UI/UiDopNode.h>
#include <zeno2/UI/UiDopScene.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>


namespace zeno2::UI {

using namespace rapidjson;


Value parse_string(std::string const &s) {
    Value v(kStringType);
    v.SetString(s.data(), s.size());
    return v;
}


std::string UiDopGraph::serialize_graph() {
    Document doc;
    doc.SetObject();

    Value v_nodes(kArrayType);
    for (auto const &[_, node]: nodes) {
        Value v_node(kObjectType);
        v_node.AddMember("name", parse_string(node->name), doc.GetAllocator());
        v_node.AddMember("kind", parse_string(node->kind), doc.GetAllocator());

        Value v_inputs(kArrayType);
        for (int i = 0; i < node->inputs.size(); i++) {
            auto *socket = node->inputs[i];
            Value v_socket(kObjectType);
            v_socket.AddMember("name", parse_string(socket->name), doc.GetAllocator());
            v_socket.AddMember("value", parse_string(socket->value), doc.GetAllocator());
            v_inputs.PushBack(v_socket, doc.GetAllocator());
        }
        v_node.AddMember("inputs", v_inputs, doc.GetAllocator());

        Value v_outputs(kArrayType);
        for (int i = 0; i < node->inputs.size(); i++) {
            auto *socket = node->inputs[i];
            Value v_socket(kObjectType);
            v_socket.AddMember("name", parse_string(socket->name), doc.GetAllocator());
            v_outputs.PushBack(v_socket, doc.GetAllocator());
        }
        v_node.AddMember("outputs", v_outputs, doc.GetAllocator());

        v_nodes.PushBack(v_node, doc.GetAllocator());
    }

    doc.AddMember("nodes", v_nodes, doc.GetAllocator());

    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    doc.Accept(writer);
    std::string str = buffer.GetString();
    std::cout << str << std::endl;
    return str;
}


}
