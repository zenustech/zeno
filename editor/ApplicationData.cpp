#include "ApplicationData.h"
#include <rapidjson/writer.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <zs/zeno/dop/dop.h>
#include <zs/ztd/format.h>
#include <zs/ztd/assert.h>
#include <zs/ztd/error.h>

using namespace zs;


static void dump_descs(rapidjson::Value &v_descs, rapidjson::Document::AllocatorType &alloc) {
    auto dump_string = [] (std::string const &s) {
        rapidjson::Value v;
        v.SetString(s.data(), s.size());
        return v;
    };

    v_descs.SetArray();
    for (auto const &kind: zeno::dop::desc_names()) {
        auto const &desc = zeno::dop::desc_of(kind);

        rapidjson::Value v_desc(rapidjson::kObjectType);

        v_desc.AddMember("kind", dump_string(kind), alloc);
        v_desc.AddMember("cate", dump_string(desc.cate.category), alloc);
        v_desc.AddMember("docs", dump_string(desc.cate.documentation), alloc);

        rapidjson::Value v_inputs(rapidjson::kArrayType);
        for (auto const &input: desc.inputs) {
            rapidjson::Value v_input(rapidjson::kObjectType);
            v_input.AddMember("name", dump_string(input.name), alloc);
            v_inputs.PushBack(v_input, alloc);
        }
        v_desc.AddMember("inputs", v_inputs, alloc);

        rapidjson::Value v_outputs(rapidjson::kArrayType);
        for (auto const &output: desc.outputs) {
            rapidjson::Value v_output(rapidjson::kObjectType);
            v_output.AddMember("name", dump_string(output.name), alloc);
            v_outputs.PushBack(v_output, alloc);
        }
        v_desc.AddMember("outputs", v_outputs, alloc);

        v_descs.PushBack(v_desc, alloc);
    }
}


static std::unique_ptr<zeno::dop::Graph> parse_graph(rapidjson::Value const &v_graph) {
    auto load_member = [] (auto &&doc, const char *name) -> auto const & {
        auto it = doc.FindMember(name);
        if (it == doc.MemberEnd())
            throw ztd::format_error("KeyError: JSON object has no member '{}'", name);
        return it->value;
    };

    std::map<std::string, zeno::dop::Node *> nodeslut;

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

        nodeslut.emplace(node->name, node);

        auto v_inputs = load_member(v_node, "inputs").GetArray();
        ZS_ZTD_ASSERT(v_inputs.Size() == desc.inputs.size());
        for (auto const &[v_input, node_input, desc_input]: ranges::views::zip
                ( v_inputs | ranges::views::transform([&] (auto &&x) { return x.GetObject(); })
                , node->inputs | ranges::views::transform([&] (auto &&x) { return std::ref(x); })
                , desc.inputs
                )) {
            auto name = load_member(v_input, "name").GetString();
            auto value = load_member(v_input, "value").GetString();
            ZS_ZTD_ASSERT(name == desc_input.name);
            node_input.get() = zeno::dop::Input_Value{.value = value};
        }

        auto v_outputs = load_member(v_node, "outputs").GetArray();
        ZS_ZTD_ASSERT(v_outputs.Size() == desc.outputs.size());
        for (auto const &[v_output, desc_output]: ranges::views::zip
                ( v_outputs | ranges::views::transform([&] (auto &&x) { return x.GetObject(); })
                , desc.outputs
            )) {
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

        auto s_node = nodeslut.at(src_node);
        auto d_node = nodeslut.at(dst_node);
        auto d_idx = ZS_ZTD_ASSERT(ranges::find_if(d_node->desc->inputs, [&] (auto const &_) {
            return _.name == dst_socket;
        }), != ranges::end(d_node->desc->inputs)) - ranges::begin(d_node->desc->inputs);
        auto s_idx = ZS_ZTD_ASSERT(ranges::find_if(d_node->desc->outputs, [&] (auto const &_) {
            return _.name == dst_socket;
        }), != ranges::end(d_node->desc->outputs)) - ranges::begin(d_node->desc->outputs);

        d_node->inputs.at(d_idx) = zeno::dop::Input_Link{.node = s_node, .sockid = (int)s_idx};
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

QString ApplicationData::get_descriptors() const {
    rapidjson::Document doc;
    dump_descs(doc, doc.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    std::string s = buffer.GetString();
    ztd::print("get_descriptors: ", s);
    return QString::fromStdString(s);
}

ApplicationData::~ApplicationData() = default;
