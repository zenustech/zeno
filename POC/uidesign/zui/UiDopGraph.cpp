#include "UiDopGraph.h"
#include "UiDopNode.h"
#include "UiDopEditor.h"


void UiDopGraph::select_child(GraphicsWidget *ptr, bool multiselect) {
    GraphicsView::select_child(ptr, multiselect);
    if (editor)
        editor->set_selection(dynamic_cast<UiDopNode *>(ptr));
}


bool UiDopGraph::remove_link(UiDopLink *link) {
    if (remove_child(link)) {
        link->from_socket->links.erase(link);
        link->to_socket->links.erase(link);
        auto to_node = link->to_socket->get_parent();
        auto from_node = link->from_socket->get_parent();
        bk_graph->remove_node_input(to_node->bk_node,
                link->to_socket->get_index());
        links.erase(link);
        return true;
    } else {
        return false;
    }
}

bool UiDopGraph::remove_node(UiDopNode *node) {
    bk_graph->remove_node(node->bk_node);
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
        nodes.erase(node);
        return true;
    } else {
        return false;
    }
}

UiDopNode *UiDopGraph::add_node(std::string kind) {
    auto p = add_child<UiDopNode>();
    p->bk_node = bk_graph->add_node(kind);
    p->name = p->bk_node->name;
    p->kind = p->bk_node->kind;
    nodes.insert(p);
    return p;
}

UiDopLink *UiDopGraph::add_link(UiDopOutputSocket *from_socket, UiDopInputSocket *to_socket) {
    auto p = add_child<UiDopLink>(from_socket, to_socket);
    auto to_node = to_socket->get_parent();
    auto from_node = from_socket->get_parent();
    bk_graph->set_node_input(to_node->bk_node, to_socket->get_index(),
            from_node->bk_node, from_socket->get_index());
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
    auto c = add_node("readvdb", {100, 256});
    auto d = add_node("vdbsmooth", {450, 256});

    add_link(c->outputs[0], d->inputs[0]);

    auto btn = add_child<Button>();
    btn->text = "Apply";
    btn->on_clicked.connect([this] () {
        bk_graph->nodes.at("vdbsmooth1")->get_output_by_name("grid")();
    });
}

void UiDopGraph::paint() const {
    glColor3f(0.2f, 0.2f, 0.2f);
    glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
}

void UiDopGraph::on_event(Event_Mouse e) {
    GraphicsView::on_event(e);

    if (e.down != true)
        return;
    if (e.btn != 0)
        return;

    auto item = item_at({cur.x, cur.y});

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

    } else if (auto socket = dynamic_cast<UiDopSocket *>(item); socket) {
        add_pending_link(socket);

    } else {
        add_pending_link(nullptr);
    }
}

UiDopNode *UiDopGraph::add_node(std::string kind, Point pos) {
    auto node = add_node(kind);
    node->position = pos;
    node->kind = kind;
    node->add_input_socket()->name = "path";
    node->add_input_socket()->name = "type";
    node->add_output_socket()->name = "grid";
    node->update_sockets();
    return node;
}

UiDopContextMenu *UiDopGraph::add_context_menu() {
    remove_context_menu();
    menu = add_child<UiDopContextMenu>();

    menu->add_entry("vdbsmooth");
    menu->add_entry("readvdb");
    menu->add_entry("vdberode");
    menu->update_entries();

    menu->on_selected.connect([this] {
        add_node(menu->selection, menu->position);
        remove_context_menu();
    });

    return menu;
}

void UiDopGraph::remove_context_menu() {
    if (menu) {
        remove_child(menu);
        menu = nullptr;
    }
}

void UiDopGraph::on_event(Event_Key e) {
    Widget::on_event(e);

    if (e.down != true)
        return;

    if (e.key == GLFW_KEY_TAB) {
        add_context_menu();

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
