#include "ApplicationData.h"
#include <rapidjson/document.h>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <zs/zeno/dop/dop.h>
#include <zs/ztd/format.h>
#include <zs/ztd/assert.h>
#include <zs/ztd/error.h>

using namespace zs;


static std::unique_ptr<zeno::dop::Graph> parse_graph(rapidjson::Value const &v_graph) {
    auto load_member = [] (auto &&doc, const char *name) -> auto const & {
        auto it = doc.FindMember(name);
        if (it == doc.MemberEnd())
            throw ztd::format_error("KeyError: JSON object has no member '{}'", name);
        return it->value;
    };

    struct NodeLutData {
        zeno::dop::Node *node_ptr{};
        std::vector<std::string> input_exprs;
    };

    std::map<std::string, NodeLutData> nodeslut;

    auto graph = std::make_unique<zeno::dop::Graph>();

    auto v_nodes = load_member(v_graph, "nodes").GetArray();
    for (auto const &_: v_nodes) {
        auto v_node = _.GetObject();

        auto kind = load_member(v_node, "kind").GetString();
        auto name = load_member(v_node, "name").GetString();
        auto const &desc = zeno::dop::desc_of(kind);
        auto node = graph->add_node(name, desc);
        node->xpos = load_member(v_node, "x").GetFloat();
        node->ypos = load_member(v_node, "y").GetFloat();

        auto &lutdata = nodeslut[node->name];
        lutdata.node_ptr = node;

        auto v_inputs = load_member(v_node, "inputs").GetArray();
        ZS_ZTD_ASSERT(v_inputs.Size() == desc.inputs.size());
        lutdata.input_exprs.reserve(v_inputs.Size());
        for (auto const &[v_input, desc_input]: ranges::views::zip(
                v_inputs | ranges::views::transform([&] (auto &&x) { return x.GetObject(); }),
                desc.inputs)) {
            auto name = load_member(v_input, "name").GetString();
            auto value = load_member(v_input, "value").GetString();
            ZS_ZTD_ASSERT(name == desc_input.name);
            lutdata.input_exprs.push_back(value);
        }

        auto v_outputs = load_member(v_node, "outputs").GetArray();
        ZS_ZTD_ASSERT(v_outputs.Size() == desc.outputs.size());
        for (auto const &[v_output, desc_output]: ranges::views::zip(v_outputs, desc.outputs)) {
            auto name = load_member(v_output, "name").GetString();
            ZS_ZTD_ASSERT(name == desc_output.name);
        }
    }

    auto v_links = load_member(v_graph, "links").GetArray();
    for (auto const &_: v_links) {
        auto v_link = _.GetObject();

        auto src_node = load_member(v_link, "src_node").GetString();
        auto src_socket = load_member(v_link, "src_socket").GetString();
        auto dst_node = load_member(v_link, "dst_node").GetString();
        auto dst_socket = load_member(v_link, "dst_socket").GetString();

        auto s_node = nodeslut.at(src_node).node_ptr;
        auto d_node = nodeslut.at(dst_node).node_ptr;
        auto i_input = ZS_ZTD_ASSERT(ranges::find_if(d_node->desc->inputs, [&] (auto const &i) {
            return i.name == dst_socket;
        }), != ranges::end(d_node->desc->inputs)) - ranges::begin(d_node->desc->inputs);

        graph->add_link(s_sock, d_sock);
    }

    return graph;
}


ApplicationData::ApplicationData(QObject *parent)
    : QObject(parent) {}

void ApplicationData::load_scene(QString str) const {
    std::string s = str.toStdString();
    ztd::print("load_scene: ", s);
    rapidjson::Document doc;
    doc.Parse(s.c_str());
    parse_graph(doc);
}

ApplicationData::~ApplicationData() = default;
