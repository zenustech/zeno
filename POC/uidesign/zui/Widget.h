#pragma once


#include "CursorState.h"


template <class T>
inline T notnull(T &&t) {
    if (!t) throw std::bad_optional_access();
    return t;
}

struct Object {
    Object() = default;
    Object(Object const &) = delete;
    Object &operator=(Object const &) = delete;
    Object(Object &&) = delete;
    Object &operator=(Object &&) = delete;
    virtual ~Object() = default;
};


struct Widget : Object {
    Widget *parent = nullptr;
    std::list<std::unique_ptr<Widget>> children;
    std::vector<std::unique_ptr<Widget>> children_gc;
    Point position{0, 0};
    float zvalue{0};

    template <class T, class ...Ts>
    T *add_child(Ts &&...ts) {
        auto p = std::make_unique<T>(std::forward<Ts>(ts)...);
        T *raw_p = p.get();
        p->parent = this;
        children.push_back(std::move(p));
        return raw_p;
    }

    void remove_all_children() {
        for (auto &child: children) {
            child->parent = nullptr;
            children_gc.push_back(std::move(child));
        }
    }

    bool remove_child(Widget *ptr) {
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

    virtual Widget *child_at(Point p) const {
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

    virtual Widget *item_at(Point p) const {
        if (!contains_point(p)) {
            return nullptr;
        }
        Widget *found = nullptr;
        float found_zvalue = 0.0f;
        for (auto const &child: children) {
            if (auto it = child->item_at(p - child->position)) {
                auto it_zvalue = child->absolute_zvalue();
                if (!found || it_zvalue >= found_zvalue) {
                    found = it;
                    found_zvalue = it_zvalue;
                }
            }
        }
        if (found) return found;
        return const_cast<Widget *>(this);
    }

    float absolute_zvalue() const {
        return parent ? zvalue : parent->absolute_zvalue() + zvalue;
    }

    virtual void on_event(Event_Hover e) {
    }

    virtual void on_event(Event_Motion e) {
    }

    virtual void on_event(Event_Mouse e) {
        pressed[e.btn] = e.down;
    }

    virtual void on_event(Event_Key e) {
    }

    virtual void on_event(Event_Char e) {
    }

    virtual void on_generic_event(Event e) {
        std::visit([this] (auto e) {
            on_event(e);
        }, e);
    }

    AABB bbox{0, 0, 10, 10};

    virtual bool contains_point(Point p) const {
        return bbox.contains(p.x, p.y);
    }

    virtual void after_update() {
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

    virtual void do_update_event() {
        auto raii = cur.translate(-position.x, -position.y);

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

    virtual void do_update() {
        auto raii = cur.translate(-position.x, -position.y);

        auto old_hovered = hovered;
        hovered = contains_point({cur.x, cur.y});

        for (auto const &child: children) {
            child->do_update();
        }

        if (!old_hovered && hovered) {
            on_event(Event_Hover{.enter = true});
        } else if (old_hovered && !hovered) {
            on_event(Event_Hover{.enter = false});
        }
    }

    virtual void do_paint() {
        auto raii = cur.translate(-position.x, -position.y);
        glPushMatrix();
        glTranslatef(position.x, position.y, zvalue);
        paint();
        for (auto const &child: children) {
            child->do_paint();
        }
        glPopMatrix();
    }

    virtual void paint() const {}
};
