#pragma once


#include "Widget.h"
#include "Button.h"
#include "Event.h"


struct UiDopContextMenu : Widget {
    static constexpr float EH = 32.f, EW = 210.f, FH = 20.f;

    std::vector<Button *> entries;
    std::string selection;

    SignalSlot on_selected;

    UiDopContextMenu();
    Button *add_entry(std::string name);
    void update_entries();
};
