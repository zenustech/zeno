#include <zs/editor/UI/GraphicsView.h>


namespace zs::editor::UI {


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
        //constexpr float SPEEDUP = 1.375f;
        translate.x += cur.dx;// * SPEEDUP;
        translate.y += cur.dy;// * SPEEDUP;

    } else if (cur.lmb) {
        for (auto const &child: children_selected) {
            if (child->draggable) {
                child->position = {
                    child->position.x + cur.dx,
                    child->position.y + cur.dy,
                };
            }
        }
    }
}


void GraphicsView::on_event(Event_Mouse e) {
    Widget::on_event(e);

    if (e.btn == 2) {
        if (e.down == true) {
            cur.focus_on(this);
        } else {
            cur.focus_on(nullptr);
        }
    }

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


void GraphicsView::on_event(Event_Scroll e) {
    Widget::on_event(e);

    float old_scaling = scaling;
    scaling *= std::pow(1.3f, e.dy);

    Point curpos(cur.x, cur.y);
    translate = translate + curpos * (old_scaling - scaling);
    /*
     * given c, s, s'
     * c' = t + c * s = t' + c * s'
     * show t' = ?
     * t' = t + c * (s - s')
     */
}


ztd::dtor_function GraphicsView::do_transform() const {
    auto offs = position + translate;
    auto raii = cur.translate(-offs.x, -offs.y, 1 / scaling);
    return raii;
}


void GraphicsView::do_paint() {
    auto raii = do_transform();

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glViewport(position.x + bbox.x0, position.y + bbox.y0, bbox.nx, bbox.ny);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glScalef(2.f, 2.f, -.001f);
    glTranslatef(-.5f, -.5f, 1.f);
    glScalef(1.f / bbox.nx, 1.f / bbox.ny, 1.f);

    paint();

    glTranslatef(translate.x, translate.y, 1.f);
    glScalef(scaling, scaling, 1.f);

    for (auto const &child: children) {
        if (!child->hidden)
            child->do_paint();
    }

    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glPopMatrix();
}


}  // namespace zs::editor::UI
