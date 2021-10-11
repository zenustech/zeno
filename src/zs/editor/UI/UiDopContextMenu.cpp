#include <zs/editor/UI/UiDopContextMenu.h>


namespace zs::editor::UI {


static constexpr float EH = 32.f, EW = 210.f, FH = 20.f, VS = 16.0f;



UiDopContextMenu::UiDopContextMenu() {
    zvalue = 10.f;
}


Button *UiDopContextMenu::add_entry(std::string const &name) {
    auto btn = add_child<Button>();
    btn->text = name;
    btn->bbox = {0, 0, EW, EH};
    btn->font_size = FH;
    btn->on_clicked.connect([=, this] {
        selection = name;
        on_selected();
    }, this);
    entries.push_back(btn);
    return btn;
}


void UiDopContextMenu::update_entries() {
    for (int i = 0; i < entries.size(); i++) {
        entries[i]->position = {0, scroll - (i + 1) * EH};
        entries[i]->bbox = {0, 0, EW, EH};
    }
    bbox = {0, scroll - entries.size() * EH, EW, entries.size() * EH};
}


void UiDopContextMenu::on_event(Event_Scroll e) {
    scroll -= e.dy * VS;
    update_entries();
}


}  // namespace zs::editor::UI
