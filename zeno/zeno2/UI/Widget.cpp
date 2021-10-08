#include <zeno2/UI/Widget.h>


namespace zeno2::UI {


void Widget::remove_all_children() {
    for (auto &child: children) {
        child->parent = nullptr;
        children_gc.push_back(std::move(child));
    }
}

bool Widget::remove_child(Widget *ptr) {
    for (auto it = children.begin(); it != children.end(); it++) {
        auto &child = *it;
        if (child.get() == ptr) {
            child->parent = nullptr;
            // transfer ownership to gc (also set child to null):
            children_gc.push_back(std::move(child));
            children.erase(it);
            return true;
        }
    }
    return false;
}

bool hovered = false;
bool pressed[3] = {false, false, false};

Widget *Widget::child_at(Point p) const {
    Widget *found = nullptr;
    for (auto const &child: children) {
        if (child->contains_point(p - child->position)) {
            if (!found || child->zvalue >= found->zvalue) {
                found = child.get();
            }
        }
    }
    return found;
}

Widget *Widget::item_at(Point p) const {
    return item_at(p, [] (Widget *) {
        return true;
    });
}

Widget *Widget::item_at(Point p, std::function<bool(Widget *)> filter) const {
    Widget *found = nullptr;
    float found_zvalue = 0.0f;
    for (auto const &child: children) {
        if (auto pt = p - child->position; child->contains_point(pt)) {
            if (auto it = child->item_at(pt); it && filter(it)) {
                auto it_zvalue = child->absolute_zvalue();
                if (!found || it_zvalue >= found_zvalue) {
                    found = it;
                    found_zvalue = it_zvalue;
                }
            }
        }
    }
    if (found) return found;
    return const_cast<Widget *>(this);
}

void Widget::on_event(Event_Hover e) {
}

void Widget::on_event(Event_Motion e) {
}

void Widget::on_event(Event_Mouse e) {
    pressed[e.btn] = e.down;
}

void Widget::on_event(Event_Key e) {
}

void Widget::on_event(Event_Char e) {
}

void Widget::on_event(Event_Scroll e) {
}

void Widget::on_generic_event(Event e) {
    std::visit([this] (auto e) {
        on_event(e);
    }, e);
}

bool Widget::contains_point(Point p) const {
    return bbox.contains(p.x, p.y);
}

void Widget::after_update() {
    bool has_any;
    do {
        has_any = false;
        for (auto it = children.begin(); it != children.end(); it++) {
            if (!*it) {
                children.erase(it);
                has_any = true;
                break;
            }
        }
    } while (has_any);

    for (auto const &child: children) {
        child->after_update();
    }
    children_gc.clear();
}

ztd::dtor_function Widget::do_transform() const {
    return cur.translate(-position.x, -position.y);
}

void Widget::do_update_event() {
    auto raii = do_transform();

    if (auto child = child_at({cur.x, cur.y}); child) {
        child->do_update_event();
    }

    for (auto const &e: cur.events) {
        on_generic_event(e);
    }

    if (cur.dx || cur.dy) {
        on_event(Event_Motion{.x = -1, .y = -1});
    }
}

void Widget::do_update() {
    auto old_hovered = hovered;
    hovered = cur.focus_widget == this ||
        contains_point({cur.x - position.x, cur.y - position.y});

    auto raii = do_transform();

    for (auto const &child: children) {
        child->do_update();
    }

    if (!old_hovered && hovered) {
        on_event(Event_Hover{.enter = true});
    } else if (old_hovered && !hovered) {
        on_event(Event_Hover{.enter = false});
    }
}

void Widget::do_paint() {
    auto raii = do_transform();
    glPushMatrix();
    glTranslatef(position.x, position.y, zvalue);
    paint();
    for (auto const &child: children) {
        if (!child->hidden)
            child->do_paint();
    }
    glPopMatrix();
}

void Widget::paint() const {
}


}  // namespace zeno2::UI
