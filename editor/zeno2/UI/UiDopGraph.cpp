#include <zeno2/UI/UiDopGraph.h>
#include <zeno2/UI/UiDopNode.h>
#include <zeno2/UI/UiDopScene.h>
#include <zeno2/UI/UiDopEditor.h>
#include <zeno2/UI/dump_graph.h>
#include <zeno2/UI/serialize_graph.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <zeno2/ztd/assert.h>
#include <zeno2/dop/dop.h>


namespace zeno2::UI {


void UiDopGraph::select_child(GraphicsWidget *ptr, bool multiselect) {
    GraphicsView::select_child(ptr, multiselect);
    if (editor) editor->set_selection(dynamic_cast<UiDopNode *>(ptr));
}


bool UiDopGraph::remove_link(UiDopLink *link) {
    if (remove_child(link)) {
        link->from_socket->links.erase(link);
        link->to_socket->links.erase(link);
        auto to_node = link->to_socket->get_parent();
        auto from_node = link->from_socket->get_parent();
        links.erase(link);
        return true;
    } else {
        return false;
    }
}


void UiDopGraph::reset_graph() {
    for (auto const &[_, node]: std::map(nodes)) {
        remove_node(node);
    }
    //remove_all_children();
}


bool UiDopGraph::remove_node(UiDopNode *node) {
    for (auto *socket: node->inputs) {
        for (auto *link: std::set(socket->links)) {
            remove_link(link);
        }
    }
    for (auto *socket: node->outputs) {
        for (auto *link: std::set(socket->links)) {
            remove_link(link);
        }
    }
    if (remove_child(node)) {
        nodes.erase(node->name);
        return true;
    } else {
        return false;
    }
}


UiDopNode *UiDopGraph::add_node(std::string kind) {
    auto p = add_child<UiDopNode>();
    for (int i = 1;; i++) {
        auto name = kind + std::to_string(i);
        if (!nodes.contains(name)) {
            p->name = name;
            break;
        }
    }
    p->kind = kind;
    nodes.emplace(p->name, p);
    return p;
}


UiDopLink *UiDopGraph::add_link(UiDopOutputSocket *from_socket, UiDopInputSocket *to_socket) {
    auto p = add_child<UiDopLink>(from_socket, to_socket);
    auto to_node = to_socket->get_parent();
    auto from_node = from_socket->get_parent();
    links.insert(p);
    return p;
}


// add a new pending link with one side linked to |socket| if no pending link
// create a real link from current pending link's socket to the |socket| otherwise
void UiDopGraph::add_pending_link(UiDopSocket *socket) {
    if (pending_link) {
        if (socket && pending_link->socket) {
            auto socket1 = pending_link->socket;
            auto socket2 = socket;
            auto output1 = dynamic_cast<UiDopOutputSocket *>(socket1);
            auto output2 = dynamic_cast<UiDopOutputSocket *>(socket2);
            auto input1 = dynamic_cast<UiDopInputSocket *>(socket1);
            auto input2 = dynamic_cast<UiDopInputSocket *>(socket2);
            if (output1 && input2) {
                add_link(output1, input2);
            } else if (input1 && output2) {
                add_link(output2, input1);
            }
        } else if (auto another = dynamic_cast<UiDopInputSocket *>(pending_link->socket); another) {
            another->clear_links();
        }
        remove_child(pending_link);
        pending_link = nullptr;

    } else if (socket) {
        pending_link = add_child<UiDopPendingLink>(socket);
    }
}


UiDopGraph::UiDopGraph() {
    add_node("ToView", {400, 128});

    auto apply_btn = add_child<Button>();
    apply_btn->text = "Apply";
    apply_btn->on_clicked.connect([this] {
        auto g = dump_graph(this);
        auto val = dop::resolve(dop::Input_Link{.node = g->get_node("ToView1")});
        get_parent()->set_view_result(val);
    }, this);

    auto save_btn = add_child<Button>();
    save_btn->position = {300, 0};
    save_btn->text = "Save";
    save_btn->on_clicked.connect([this] {
        rapidjson::Document doc;
        doc.SetObject();
        auto v_graph = serialize(this, doc.GetAllocator());
        doc.AddMember("version", rapidjson::Value().SetString("v2", 2), doc.GetAllocator());
        doc.AddMember("graph", v_graph, doc.GetAllocator());
        rapidjson::StringBuffer buffer;
        rapidjson::Writer writer(buffer);
        doc.Accept(writer);
        std::string json = buffer.GetString();
        SPDLOG_INFO("dumped graph to JSON: {}", json);
        glfwSetClipboardString(cur.window, json.c_str());
    }, this);

    auto load_btn = add_child<Button>();
    load_btn->position = {600, 0};
    load_btn->text = "Load";
    load_btn->on_clicked.connect([this] {
        auto json = glfwGetClipboardString(cur.window);
        if (!json) return;
        SPDLOG_INFO("loading graph from JSON: {}", json);
        rapidjson::Document doc;
        doc.Parse(json);
        [[unlikely]] if (
            auto version = ZENO2_ZTD_ASSERT(doc.FindMember("version"), != doc.MemberEnd()
                                           )->value.GetString(); std::strcmp("v2", version)) {
            SPDLOG_WARN("found incompatible version: {} != v2", version);
        }
        auto const &v_graph = ZENO2_ZTD_ASSERT(doc.FindMember("graph"), != doc.MemberEnd())->value;
        deserialize(this, v_graph);
    }, this);
}


void UiDopGraph::paint() const {
    glColor3f(0.2f, 0.2f, 0.2f);
    glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
}


void UiDopGraph::on_event(Event_Mouse e) {
    GraphicsView::on_event(e);

    if (e.down != true)
        return;

    if (e.btn == 1) {
        if (pending_link) {
            remove_child(pending_link);
            pending_link = nullptr;
        } else {
            select_child(nullptr, false);
        }
    }

    if (e.btn != 0)
        return;

    auto item = item_at({cur.x, cur.y}, [=] (Widget *it) {
        // filter away UiDopPendingLink and UiDopLink:
        return !dynamic_cast<GraphicsLineItem *>(it);
    });

    if (auto node = dynamic_cast<UiDopNode *>(item); node) {
        if (pending_link) {
            auto another = pending_link->socket;
            if (dynamic_cast<UiDopInputSocket *>(another) && node->outputs.size()) {
                add_pending_link(node->outputs[0]);
            } else if (dynamic_cast<UiDopOutputSocket *>(another) && node->inputs.size()) {
                add_pending_link(node->inputs[0]);
            } else {
                add_pending_link(nullptr);
            }
        }

#if 0
    } else if (auto link = dynamic_cast<UiDopLink *>(item); link) {
        if (pending_link) {
            auto another = pending_link->socket;
            if (dynamic_cast<UiDopInputSocket *>(another)) {
                add_pending_link(link->from_socket);
            } else if (dynamic_cast<UiDopOutputSocket *>(another)) {
                add_pending_link(link->to_socket);
            } else {
                add_pending_link(nullptr);
            }
        }
#endif

    } else if (auto socket = dynamic_cast<UiDopSocket *>(item); socket) {
        add_pending_link(socket);

    } else {
        add_pending_link(nullptr);
    }
}


UiDopNode *UiDopGraph::add_node(std::string kind, Point pos) {
    auto node = add_node(kind);
    node->position = pos;
    auto const &desc = dop::desc_of(kind);
    for (auto const &sock_info: desc.inputs) {
        auto socket = node->add_input_socket();
        socket->name = sock_info.name;
    }
    for (auto const &sock_info: desc.outputs) {
        auto socket = node->add_output_socket();
        socket->name = sock_info.name;
    }
    node->update_sockets();
    return node;
}


UiDopContextMenu *UiDopGraph::add_context_menu() {
    remove_context_menu();

    menu = get_parent()->add_child<UiDopContextMenu>();
    menu->position = position + translate + Point(cur.x, cur.y) * scaling;
    for (auto const &key: dop::desc_names()) {
        menu->add_entry(key);
    }
    menu->update_entries();

    menu->on_selected.connect([this] {
        add_node(menu->selection, (menu->position - position - translate) * (1.f / scaling));
        remove_context_menu();
    }, this);

    return menu;
}


void UiDopGraph::remove_context_menu() {
    if (menu) {
        get_parent()->remove_child(menu);
        menu = nullptr;
    }
}


void UiDopGraph::on_event(Event_Key e) {
    Widget::on_event(e);

    if (e.down != true)
        return;

    if (e.key == GLFW_KEY_TAB) {
        if (!menu)
            add_context_menu();
        else
            remove_context_menu();

    } else if (e.key == GLFW_KEY_DELETE) {
        for (auto *item: children_selected) {
            if (auto link = dynamic_cast<UiDopLink *>(item); link) {
                remove_link(link);
            } else if (auto node = dynamic_cast<UiDopNode *>(item); node) {
                remove_node(node);
            }
        }
        children_selected.clear();
        select_child(nullptr, false);
    }
}


}  // namespace zeno2::UI
