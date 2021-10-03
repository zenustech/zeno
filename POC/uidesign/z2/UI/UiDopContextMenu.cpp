#include <z2/UI/UiDopContextMenu.h>


namespace z2::UI {


UiDopContextMenu::UiDopContextMenu() {
    zvalue = 10.f;
}


Button *UiDopContextMenu::add_entry(std::string name) {
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


void UiDopContextMenu::update_entries() {
    for (int i = 0; i < entries.size(); i++) {
        entries[i]->position = {0, -(i + 1) * EH};
        entries[i]->bbox = {0, 0, EW, EH};
    }
    bbox = {0, entries.size() * -EH, EW, entries.size() * EH};
}


}  // namespace z2::UI
