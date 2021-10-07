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

    UiDopSocket() {
        bbox = {-R, -R, 2 * R, 2 * R};
        zvalue = 2.f;
    }

    void paint() const override {
        glColor3f(0.75f, 0.75f, 0.75f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
        if (hovered) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (links.size()) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0 + BW, bbox.y0 + BW, bbox.x0 + bbox.nx - BW, bbox.y0 + bbox.ny - BW);
    }

    UiDopNode *get_parent() const {
        return (UiDopNode *)(parent);
    }

    bool is_parent_active() const;
    void clear_links();
};


struct UiDopInputSocket : UiDopSocket {
    std::string value;

    void paint() const override {
        UiDopSocket::paint();

        if (is_parent_active()) {
            Font font("regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_LEFT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(R * 1.3f, -R + FH * 0.15f, name);
        }
    }

    void attach_link(UiDopLink *link) {
        clear_links();
        links.insert(link);
    }
};


struct UiDopOutputSocket : UiDopSocket {
    void paint() const override {
        UiDopSocket::paint();

        if (is_parent_active()) {
            Font font("regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_RIGHT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(-NW - R * 1.5f, -R + FH * 0.15f, name);
        }
    }

    void attach_link(UiDopLink *link) {
        links.insert(link);
    }
};


}  // namespace z2::UI
