#pragma once


#include <z2/UI/Label.h>
#include <z2/UI/TextEdit.h>
#include <z2/UI/UiDopSocket.h>
#include <z2/UI/Font.h>
#include <z2/dop/DopNode.h>
#include <z2/dop/DopSocket.h>


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

    void set_bk_socket
        ( UiDopInputSocket *socket
        , dop::DopInputSocket *bk_socket
        , dop::DopNode *bk_node
        ) {
        label->text = socket->name;
        edit->text = bk_socket->value;
        edit->disabled = socket->links.size();

        edit->on_editing_finished.connect([=, this] {
            if (bk_socket->value != edit->text) {
                bk_socket->value = edit->text;
                bk_node->invalidate();
            }
        });
    }
};


}  // namespace z2::UI
