#pragma once


#include <zs/editor/UI/Label.h>
#include <zs/editor/UI/TextEdit.h>
#include <zs/editor/UI/UiDopSocket.h>
#include <zs/editor/UI/Font.h>


namespace zs::editor::UI {


struct UiDopParam : Widget {
    Label *label;
    TextEdit *edit;

    UiDopParam();
    void set_socket(UiDopInputSocket *socket);
};


}  // namespace zs::editor::UI
