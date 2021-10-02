#pragma once


#include <zeno/UI/Label.h>


namespace zeno::UI {


struct TextEdit : Label {
    int cursor = 0;
    int sellen = 0;
    bool disabled = false;

    float font_size = 20.f;

    void _insert_text(std::string content) {
        text = text.substr(0, cursor) + content + text.substr(cursor + sellen);
        cursor += content.size();
        sellen = 0;
    }

    SignalSlot on_editing_finished;

    void on_event(Event_Hover e) override;
    void on_event(Event_Mouse e) override;
    void on_event(Event_Key e) override;
    void on_event(Event_Char e) override;

    void paint() const override;
};


}  // namespace zeno::UI
