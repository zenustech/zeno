#pragma once


#include <zeno2/UI/Widget.h>


namespace zeno2::UI {


template <class Element = Widget>
struct TabPanel : Widget {
    std::set<Element *> elements;
    Element *current = nullptr;

    template <class T = Element, class ...Ts>
    Element *add_element(Ts &&...ts) {
        auto elm = this->add_child<T>(std::forward<Ts>(ts)...);
        elm->hidden = true;
        elements.insert(elm);
        return elm;
    }

    void show_element(Element *elm) {
        for (auto *e: elements) {
            e->hidden = e != elm;
        }
        bbox = elm->bbox + elm->position;
        current = elm;
    }

    bool remove_element(Element *elm) {
        if (remove_child(elm)) {
            elements.erase(elm);
            return true;
        } else {
            return false;
        }
    }
};

}
