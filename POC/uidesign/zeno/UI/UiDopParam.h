#pragma once


#include <zeno/UI/Label.h>
#include <zeno/UI/TextEdit.h>
#include <zeno/UI/UiDopSocket.h>
#include <zeno/UI/Font.h>


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
        , DopInputSocket *bk_socket
        , DopNode *bk_node
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
