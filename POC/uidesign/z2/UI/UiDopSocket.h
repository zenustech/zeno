#pragma once


#include <z2/UI/GraphicsRectItem.h>
#include <z2/UI/Font.h>


namespace z2::UI {


struct UiDopLink;
struct UiDopNode;

struct UiDopSocket : GraphicsRectItem {
    static constexpr float BW = 4, R = 15, FH = 19, NW = 200;

    std::string name;
    std::set<UiDopLink *> links;
    bool failed = false;

    UiDopSocket();

    void paint() const override;

    UiDopNode *get_parent() const {
        return (UiDopNode *)(parent);
    }

    void clear_links();
};


struct UiDopInputSocket : UiDopSocket {
    std::string value;

    void paint() const override;

    void attach_link(UiDopLink *link) {
        clear_links();
        links.insert(link);
    }
};


struct UiDopOutputSocket : UiDopSocket {
    void paint() const override;

    void attach_link(UiDopLink *link) {
        links.insert(link);
    }
};


}  // namespace z2::UI
