#include <z2/UI/UiDopSocket.h>
#include <z2/UI/UiDopNode.h>
#include <z2/UI/UiDopGraph.h>


namespace z2::UI {


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
