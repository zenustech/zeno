#pragma once


#include <zeno/UI/Widget.h>
#include <zeno/UI/Button.h>
#include <zeno/UI/Event.h>


namespace zeno::UI {


struct UiDopContextMenu : Widget {
    static constexpr float EH = 32.f, EW = 210.f, FH = 20.f;

    std::vector<Button *> entries;
    std::string selection;

    SignalSlot on_selected;

    UiDopContextMenu();
    Button *add_entry(std::string name);
    void update_entries();
};


}  // namespace zeno::UI
