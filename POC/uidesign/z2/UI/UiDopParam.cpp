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


void UiDopParam::set_socket(UiDopInputSocket *socket) {
    label->text = socket->name;
    edit->text = socket->value;
    edit->disabled = socket->links.size() != 0;
    edit->on_editing_finished.connect([=, this] {
        socket->value = edit->text;
    });
}


}
