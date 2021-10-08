#pragma once


#include <zeno2/UI/Widget.h>
#include <zeno2/UI/Button.h>
#include <zeno2/UI/Event.h>


namespace zeno2::UI {


struct UiDopContextMenu : Widget {
    static constexpr float EH = 32.f, EW = 210.f, FH = 20.f;

    std::vector<Button *> entries;
    std::string selection;

    SignalSlot on_selected;

    UiDopContextMenu();
    Button *add_entry(std::string name);
    void update_entries();
};


}  // namespace zeno2::UI
