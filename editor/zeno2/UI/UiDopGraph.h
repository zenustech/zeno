#pragma once


#include <zeno2/UI/UiDopNode.h>
#include <zeno2/UI/UiDopLink.h>
#include <zeno2/UI/GraphicsView.h>
#include <zeno2/UI/UiDopContextMenu.h>
#include <zeno2/UI/UiDopSocket.h>
#include <zeno2/dop/dop.h>


namespace zeno2::UI {


struct UiDopEditor;
struct UiDopScene;


struct UiDopGraph : GraphicsView {
    UiDopPendingLink *pending_link = nullptr;
    std::map<std::string, UiDopNode *> nodes;
    std::set<UiDopLink *> links;

    UiDopEditor *editor = nullptr;
    UiDopContextMenu *menu = nullptr;

    UiDopGraph();

    void paint() const override;

    // add a new pending link with one side linked to |socket| if no pending link
    // create a real link from current pending link's socket to the |socket| otherwise
    void add_pending_link(UiDopSocket *socket);
    UiDopNode *add_node(std::string kind, Point pos);
    UiDopNode *add_node(std::string kind);
    // must invoke these two functions rather than operate on |links| and
    // |remove_child| directly to prevent bad pointer
    bool remove_link(UiDopLink *link);
    bool remove_node(UiDopNode *node);
    UiDopLink *add_link(UiDopOutputSocket *from_socket, UiDopInputSocket *to_socket);

    UiDopContextMenu *add_context_menu();
    void remove_context_menu();

    void select_child(GraphicsWidget *ptr, bool multiselect) override;

    void on_event(Event_Mouse e) override;
    void on_event(Event_Key e) override;

    std::unique_ptr<dop::Graph> dump_graph();
    std::string serialize_graph();

    inline UiDopScene *get_parent() const {
        return (UiDopScene *)parent;
    }
};


}  // namespace zeno2::UI
