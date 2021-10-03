#include <z2/UI/GraphicsWidget.h>


namespace z2::UI {


void GraphicsView::select_child(GraphicsWidget *ptr, bool multiselect) {
    if (!(multiselect || (ptr && ptr->selected))) {
        for (auto const &child: children_selected) {
            child->selected = false;
        }
        children_selected.clear();
    }
    if (ptr) {
        if (ptr->selected && multiselect) {
            children_selected.erase(ptr);
            ptr->selected = false;
        } else {
            children_selected.insert(ptr);
            ptr->selected = true;
        }
    }
}

void GraphicsView::on_event(Event_Motion e) {
    Widget::on_event(e);
    if (cur.mmb) {
        translate.x += cur.dx;
        translate.y += cur.dy;

    } else if (cur.lmb) {
        for (auto const &child: children_selected) {
            if (child->draggable) {
                child->position.x += cur.dx;
                child->position.y += cur.dy;
            }
        }
    }
}

void GraphicsView::on_event(Event_Mouse e) {
    Widget::on_event(e);

    if (e.down != true)
        return;
    if (e.btn != 0)
        return;

    if (auto item = item_at({cur.x, cur.y}); item) {
        auto it = dynamic_cast<GraphicsWidget *>(item);
        if (it && it->selectable)
            select_child(it, cur.shift);
    } else if (!cur.shift) {
        select_child(nullptr, false);
    }
}


ztd::CallOnDtor GraphicsView::do_transform() const {
    auto raii = cur.translate(-position.x, -position.y);
    return raii;
}


void GraphicsView::do_paint() {
    auto raii = cur.translate(-position.x, -position.y);
    glPushMatrix();

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glViewport(position.x + bbox.x0, position.y + bbox.y0, bbox.nx, bbox.ny);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(2.f, 2.f, -.001f);
    glTranslatef(-.5f, -.5f, 1.f);
    glScalef(1.f / bbox.nx, 1.f / bbox.ny, 1.f);

    glTranslatef(translate.x, translate.y, 1.f);

    paint();
    for (auto const &child: children) {
        child->do_paint();
    }

    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glPopMatrix();
}


}  // namespace z2::UI
