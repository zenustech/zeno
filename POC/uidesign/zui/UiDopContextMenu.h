#pragma once


#include "Widget.h"
#include "Button.h"
#include "Event.h"


struct UiDopContextMenu : Widget {
    static constexpr float EH = 32.f, EW = 210.f, FH = 20.f;

    std::vector<Button *> entries;
    std::string selection;

    SignalSlot on_selected;

    UiDopContextMenu() {
        position = {cur.x, cur.y};
        zvalue = 10.f;
    }

    Button *add_entry(std::string name) {
        auto btn = add_child<Button>();
        btn->text = name;
        btn->bbox = {0, 0, EW, EH};
        btn->font_size = FH;
        btn->on_clicked.connect([=, this] {
            selection = name;
            on_selected();
        });
        entries.push_back(btn);
        return btn;
    }

    void update_entries() {
        for (int i = 0; i < entries.size(); i++) {
            entries[i]->position = {0, -(i + 1) * EH};
        }
        bbox = {0, entries.size() * -EH, EW, entries.size() * EH};
    }
};
