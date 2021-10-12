#pragma once


#include <zs/editor/UI/GraphicsRectItem.h>
#include <zs/editor/UI/Font.h>
#include <zs/editor/UI/Color.h>


namespace zs::editor::UI {


struct UiDopLink;
struct UiDopNode;

struct UiDopSocket : GraphicsRectItem {
    static constexpr float BW = 4, R = 15, FH = 19, NW = 200;

    std::string name;
    std::set<UiDopLink *> links;
    bool failed = false;

    UiDopSocket();

    void paint() const override;
    virtual Color get_color() const;

    UiDopNode *get_parent() const {
        return (UiDopNode *)(parent);
    }

    void clear_links();
};


struct UiDopInputSocket : UiDopSocket {
    std::string value;

    void paint() const override;
    Color get_color() const override;

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


}  // namespace zs::editor::UI
