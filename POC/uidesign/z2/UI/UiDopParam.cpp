#include <z2/UI/UiDopParam.h>


namespace z2::UI {


void UiDopParam::set_bk_socket
    ( UiDopInputSocket *socket
    , dop::Input *bk_socket
    , dop::Node *bk_node
    ) {
    label->text = socket->name;
    edit->text = unevaluate(bk_socket->value);
    edit->disabled = socket->links.size();

    edit->on_editing_finished.connect([=, this] {
        bk_socket->value = evaluate(edit->text);
        bk_node->invalidate();
    });
}


}
