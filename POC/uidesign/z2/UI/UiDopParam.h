#pragma once


#include <z2/UI/Label.h>
#include <z2/UI/TextEdit.h>
#include <z2/UI/UiDopSocket.h>
#include <z2/UI/Font.h>


namespace z2::UI {


struct UiDopParam : Widget {
    Label *label;
    TextEdit *edit;

    UiDopParam();
    void set_socket(UiDopSocket *socket);
};


}  // namespace z2::UI
