#pragma once


#include <zeno/UI/Label.h>


namespace zeno::UI {


struct Button : Label {
    SignalSlot on_clicked;

    void on_event(Event_Mouse e) override;
    void paint() const override;
};


}  // namespace zeno::UI
