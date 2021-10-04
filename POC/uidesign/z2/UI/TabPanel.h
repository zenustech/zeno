#pragma once


#include <z2/UI/Widget.h>


namespace z2::UI {


struct TabPanel : Widget {
    std::set<Widget *> elements;

    TabPanel();

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
