#pragma once


#include <z2/UI/UiDopNode.h>
#include <z2/UI/UiDopLink.h>
#include <z2/UI/UiDopContextMenu.h>
#include <z2/UI/UiDopSocket.h>
#include <z2/dop/DopGraph.h>


namespace z2::UI {


struct UiDopEditor;


struct UiDopGraph : GraphicsView {
    std::set<UiDopNode *> nodes;
    std::set<UiDopLink *> links;
    UiDopPendingLink *pending_link = nullptr;

    std::unique_ptr<dop::DopGraph> bk_graph = std::make_unique<dop::DopGraph>();

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
};


}  // namespace z2::UI
