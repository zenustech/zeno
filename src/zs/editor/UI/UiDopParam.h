#pragma once


#include <zeno2/UI/Label.h>
#include <zeno2/UI/TextEdit.h>
#include <zeno2/UI/UiDopSocket.h>
#include <zeno2/UI/Font.h>


namespace zeno2::UI {


struct UiDopParam : Widget {
    Label *label;
    TextEdit *edit;

    UiDopParam();
    void set_socket(UiDopInputSocket *socket);
};


}  // namespace zeno2::UI
