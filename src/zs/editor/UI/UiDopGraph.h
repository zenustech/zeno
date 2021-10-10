#pragma once


#include <zs/editor/UI/UiDopNode.h>
#include <zs/editor/UI/UiDopLink.h>
#include <zs/editor/UI/GraphicsView.h>
#include <zs/editor/UI/UiDopContextMenu.h>
#include <zs/editor/UI/UiDopSocket.h>


namespace zs::editor::UI {


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

    UiDopNode *add_node(std::string kind, Point pos);
    UiDopNode *add_node(std::string kind);
    UiDopLink *add_link(UiDopOutputSocket *from_socket, UiDopInputSocket *to_socket);
    // add a new pending link with one side linked to |socket| if no pending link
    // create a real link from current pending link's socket to the |socket| otherwise
    void add_pending_link(UiDopSocket *socket);
    // must invoke these two functions rather than operate on |links| and
    // |remove_child| directly to prevent bad pointer
    bool remove_node(UiDopNode *node);
    bool remove_link(UiDopLink *link);
    // clear all nodes and links, i.e. reset to initial state
    void reset_graph();

    UiDopContextMenu *add_context_menu();
    void remove_context_menu();

    void select_child(GraphicsWidget *ptr, bool multiselect) override;

    void on_event(Event_Mouse e) override;
    void on_event(Event_Key e) override;

    inline UiDopScene *get_parent() const {
        return (UiDopScene *)parent;
    }
};


}  // namespace zs::editor::UI
