#pragma once


#include <zs/editor/UI/Widget.h>
#include <zs/editor/UI/Button.h>
#include <zs/editor/UI/Event.h>


namespace zs::editor::UI {


struct UiDopContextMenu : Widget {
    std::vector<Button *> entries;
    std::string selection;
    float scroll = 0;

    SignalSlot on_selected;

    UiDopContextMenu();
    Button *add_entry(std::string const &name);
    void update_entries();
    void on_event(Event_Scroll e) override;
};


}  // namespace zs::editor::UI
