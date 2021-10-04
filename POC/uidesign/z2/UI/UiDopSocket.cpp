#include <z2/UI/UiDopSocket.h>
#include <z2/UI/UiDopNode.h>
#include <z2/UI/UiDopGraph.h>


namespace z2::UI {


int UiDopInputSocket::get_index() const {
    auto node = get_parent();
    for (int i = 0; i < node->inputs.size(); i++) {
        if (node->inputs[i] == this) {
            return i;
        }
    }
    throw ztd::make_error("Cannot find index of input node");
}

int UiDopOutputSocket::get_index() const {
    auto node = get_parent();
    for (int i = 0; i < node->outputs.size(); i++) {
        if (node->outputs[i] == this) {
            return i;
        }
    }
    throw ztd::make_error("Cannot find index of output node");
}


bool UiDopSocket::is_parent_active() const {
    return get_parent()->hovered;
}

void UiDopSocket::clear_links() {
    auto graph = get_parent()->get_parent();
    if (links.size()) {
        for (auto link: std::set(links)) {
            graph->remove_link(link);
        }
    }
}


}  // namespace z2::UI
