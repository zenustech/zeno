#include "UiDopSocket.h"
#include "UiDopNode.h"


int UiDopInputSocket::get_index() const {
    auto node = get_parent();
    for (int i = 0; i < node->inputs.size(); i++) {
        if (node->inputs[i] == this) {
            return i;
        }
    }
    throw ztd::makeException("Cannot find index of input node");
}

int UiDopOutputSocket::get_index() const {
    auto node = get_parent();
    for (int i = 0; i < node->outputs.size(); i++) {
        if (node->outputs[i] == this) {
            return i;
        }
    }
    throw ztd::makeException("Cannot find index of output node");
}
