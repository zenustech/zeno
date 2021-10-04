#pragma once


#include <z2/UI/Label.h>


namespace z2::UI {


struct Button : Label {
    SignalSlot on_clicked;

    void on_event(Event_Mouse e) override;
    void paint() const override;
};


}  // namespace z2::UI
