#pragma once


#include <z2/UI/Label.h>
#include <z2/UI/TextEdit.h>
#include <z2/UI/UiDopSocket.h>
#include <z2/UI/Font.h>


namespace z2::UI {


struct UiDopParam : Widget {
    Label *label;
    TextEdit *edit;

    UiDopParam() {
        bbox = {0, 0, 500, 50};
        label = add_child<Label>();
        label->position = {0, 5};
        label->bbox = {0, 0, 100, 40};
        edit = add_child<TextEdit>();
        edit->position = {100, 5};
        edit->bbox = {0, 0, 400, 40};
    }
};


}  // namespace z2::UI
