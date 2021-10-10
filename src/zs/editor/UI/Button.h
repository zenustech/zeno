#pragma once


#include <zs/editor/UI/Label.h>


namespace zeno2::UI {


struct Button : Label {
    SignalSlot on_clicked;

    void on_event(Event_Mouse e) override;
    void paint() const override;
};


}  // namespace zeno2::UI
