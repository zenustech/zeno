#include <z2/UI/UiDopParam.h>
#include <z2/UI/UiDopSocket.h>


namespace z2::UI {


UiDopParam::UiDopParam() {
    bbox = {0, 0, 500, 50};
    label = add_child<Label>();
    label->position = {0, 5};
    label->bbox = {0, 0, 100, 40};
    edit = add_child<TextEdit>();
    edit->position = {100, 5};
    edit->bbox = {0, 0, 400, 40};
}


void UiDopParam::set_socket(UiDopSocket *socket) {
    edit->on_editing_finished([=, this] {
        socket->value = edit->text;
    });
}


}
