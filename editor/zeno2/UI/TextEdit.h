#pragma once


#include <zeno2/UI/Label.h>


namespace zeno2::UI {


struct TextEdit : Label {
    int cursor = 0;
    int sellen = 0;
    bool disabled = false;

    float font_size = 20.f;

    void insert_text(std::string content);

    SignalSlot on_editing_finished;

    void on_event(Event_Hover e) override;
    void on_event(Event_Mouse e) override;
    void on_event(Event_Key e) override;
    void on_event(Event_Char e) override;

    void paint() const override;
};


}  // namespace zeno2::UI
