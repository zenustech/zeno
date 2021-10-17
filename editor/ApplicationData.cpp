#include "ApplicationData.h"
#include <rapidjson/document.h>
#include <zs/zeno/dop/dop.h>
#include <zs/ztd/format.h>
#include <zs/ztd/error.h>

using namespace zs;


ApplicationData::ApplicationData(QObject *parent)
    : QObject(parent) {}

void ApplicationData::load_scene(QString str) const {
    ztd::print("load_scene: ", str.toStdString());
}

ApplicationData::~ApplicationData() = default;


static rapidjson::Value dump_string(std::string const &s) {
    rapidjson::Value v;
    v.SetString(s.data(), s.size());
    return v;
}


template <class Doc>
auto const &load_member(Doc &&doc, const char *name) {
    auto it = doc.FindMember(name);
    if (it == doc.MemberEnd())
        throw ztd::format_error("KeyError: no member named '{}'", name);
    return it->value;
}


void parse_graph(rapidjson::Value const &v_graph) {
    struct NodeLutData {
        zeno::dop::Node *node_ptr{};
        std::vector<std::string> input_exprs;
    };

    std::map<std::string, NodeLutData> nodeslut;

    auto v_nodes = load_member(v_graph, "nodes").GetArray();
    for (auto const &_: v_nodes) {
        auto v_node = _.GetObject();

        auto kind = load_member(v_node, "kind").GetString();
        auto name = load_member(v_node, "name").GetString();
        auto const &desc = zeno::dop::desc_of(kind);
        auto node = graph->add_node(name, desc);
        node->xpos = load_member(v_node, "x").GetFloat();
        node->ypos = load_member(v_node, "y").GetFloat();

        auto v_inputs = load_member(v_node, "inputs").GetArray();
        ZS_ZTD_ASSERT_EQ(v_inputs.Size(), desc.inputs.size());
        input_exprs.reserve(v_inputs.Size());
        for (auto const &v_input: v_inputs) {
            auto v_socket = v_input.GetObject();
            auto name = load_member(v_socket, "name").GetString();
            auto value = load_member(v_socket, "value").GetString();
            input_exprs.push_back(input);
        }

        auto v_outputs = load_member(v_node, "outputs").GetArray();
        ZS_ZTD_ASSERT_EQ(v_outputs.Size(), desc.outputs.size());
        for (auto const &v_output: v_outputs) {
            auto name = load_member(v_socket, "name").GetString();
            ZS_ZTD_ASSERT_EQ(name, desc.outputs);
        }

        nodeslut.emplace(node->name, node);
    }

    auto v_links = load_member(v_graph, "links").GetArray();
    for (auto const &_: v_links) {
        auto v_link = _.GetObject();

        auto src_node = load_member(v_link, "src_node").GetString();
        auto src_socket = load_member(v_link, "src_socket").GetInt();
        auto dst_node = load_member(v_link, "dst_node").GetString();
        auto dst_socket = load_member(v_link, "dst_socket").GetInt();

        auto s_node = graph->nodes.at(src_node);
        auto s_sock = f_node->outputs.at(src_socket);
        auto d_node = graph->nodes.at(dst_node);
        auto d_sock = t_node->inputs.at(dst_socket);

        graph->add_link(s_sock, d_sock);
    }
}
