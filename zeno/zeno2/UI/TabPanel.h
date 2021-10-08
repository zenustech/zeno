#pragma once


#include <zeno2/UI/Widget.h>


namespace zeno2::UI {


struct TabPanel : Widget {  // TODO: WIP, complete to unlock a multi-dop-graph editor
    std::set<Widget *> elements;

    template <class T, class ...Ts>
    Widget *add_element(Ts &&...ts) {
        auto elm = this->add_child<T>(std::forward<Ts>(ts)...);
        elements.insert(elm);
        return elm;
    }

    bool remove_element(Widget *elm) {
        if (remove_child(elm)) {
            elements.erase(elm);
            return true;
        } else {
            return false;
        }
    }
};

}
