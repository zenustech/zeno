#include <z2/UI/UiDopSocket.h>
#include <z2/UI/UiDopNode.h>
#include <z2/UI/UiDopGraph.h>


namespace z2::UI {


UiDopSocket::UiDopSocket() {
    bbox = {-R, -R, 2 * R, 2 * R};
    zvalue = 2.f;
}


void UiDopSocket::clear_links() {
    auto graph = get_parent()->get_parent();
    if (links.size()) {
        for (auto link: std::set(links)) {
            graph->remove_link(link);
        }
    }
}


void UiDopSocket::paint() const {
    glColor3f(0.75f, 0.75f, 0.75f);
    glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    if (hovered) {
        glColor3f(0.75f, 0.5f, 0.375f);
    } else if (failed) {
        glColor3f(0.825f, 0.225f, 0.125f);
    } else if (links.size()) {
        glColor3f(0.375f, 0.5f, 1.0f);
    } else {
        glColor3f(0.375f, 0.375f, 0.375f);
    }
    glRectf(bbox.x0 + BW, bbox.y0 + BW, bbox.x0 + bbox.nx - BW, bbox.y0 + bbox.ny - BW);
}


void UiDopInputSocket::paint() const {
    UiDopSocket::paint();

    if (get_parent()->hovered) {
        Font font("regular.ttf");
        font.set_font_size(FH);
        font.set_fixed_height(2 * R);
        font.set_fixed_width(NW, FTGL::ALIGN_LEFT);
        glColor3f(1.f, 1.f, 1.f);
        font.render(R * 1.3f, -R + FH * 0.15f, name);
    }
}


void UiDopOutputSocket::paint() const {
    UiDopSocket::paint();

    if (get_parent()->hovered) {
        Font font("regular.ttf");
        font.set_font_size(FH);
        font.set_fixed_height(2 * R);
        font.set_fixed_width(NW, FTGL::ALIGN_RIGHT);
        glColor3f(1.f, 1.f, 1.f);
        font.render(-NW - R * 1.5f, -R + FH * 0.15f, name);
    }
}


}  // namespace z2::UI
