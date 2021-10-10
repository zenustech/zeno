#include <zs/editor/UI/UiDopGraph.h>
#include <zs/editor/UI/UiDopNode.h>
#include <zs/editor/UI/UiDopScene.h>
#include <zs/zeno/dop/dop.h>
#include <cctype>


namespace zs::editor::UI {


static ztd::zany parse_any(std::string const &expr) {
    if (!expr.size()) {  // (empty string)
        return {};
    } else if (std::isdigit(expr[0])) {
        if (expr.find('.') != std::string::npos) {  // 3.14
            return std::stof(expr);
        } else {  // 42
            return std::stoi(expr);
        }
    } else {  // string
        return expr;
    }
}


static dop::Input parse_socket
    ( dop::Graph *g
    , ztd::map<std::string, dop::Input_Link> const &exprlut
    , UiDopInputSocket *socket
    ) {
    if (socket->links.size()) {  // (physically linked)
        auto *link = *socket->links.begin();
        auto *outsocket = link->from_socket;
        auto *outnode = outsocket->get_parent();
        auto outn = g->get_node(outnode->name);
        int outid = 0;
        for (int i = 0; i < outnode->outputs.size(); i++) {
            if (outnode->outputs[i]->name == outsocket->name) {
                outid = i;
            }
        }
        return dop::Input_Link{.node = outn, .sockid = outid};

    } else {
        auto expr = socket->value;

        if (expr.starts_with('@')) {
            expr = expr.substr(1);

            auto p = expr.find(':');
            if (p == std::string::npos) {  // @Route1
                auto outnode = g->get_node(expr);
                return dop::Input_Link{.node = outnode, .sockid = 0};

            } else {
                auto sockname = expr.substr(p + 1);
                auto nodename = expr.substr(0, p);
                auto *outn = g->get_node(nodename);

                if (!sockname.size()) {  // @Route1:
                    return dop::Input_Link{.node = outn, .sockid = 0};

                } else if (std::isdigit(sockname[0])) {  // @Route1:0
                    int outid = std::stoi(sockname);
                    return dop::Input_Link{.node = outn, .sockid = outid};

                } else {  // @Route1:value
                    return exprlut.at(expr);
                }
            }

        } else {  // 3.14
            return dop::Input_Value{.value = parse_any(expr)};
        }
    }
}


std::unique_ptr<dop::Graph> dump_graph(UiDopGraph const *graph) {
    auto g = std::make_unique<dop::Graph>();

    ztd::map<std::string, dop::Input_Link> exprlut;
    for (auto const &[_, node]: graph->nodes) {
        auto n = g->add_node(node->name, dop::desc_of(node->kind));
        for (int i = 0; i < node->outputs.size(); i++) {
            auto key = node->name + ':' + node->outputs[i]->name;
            // (todo: really necessary here? may use get_index...)
            // for fast lookup like exprlut.at("Route1:value") in later pass
            exprlut.emplace(key, dop::Input_Link{.node = n, .sockid = i});
        }
        n->xpos = node->position.x;
        n->name = node->name;
        n->desc = &dop::desc_of(node->kind);
    }

    for (auto const &[_, node]: graph->nodes) {
        auto n = g->get_node(node->name);

        for (int i = 0; i < node->inputs.size(); i++) {
            auto *socket = node->inputs[i];
            auto input = parse_socket(g.get(), exprlut, socket);
            n->inputs.at(i) = input;
        }
    }
    return g;
}


}
