#pragma once


#include <zeno/UI/GraphicsLineItem.h>


struct UiDopLink : GraphicsLineItem {
    UiDopOutputSocket *from_socket;
    UiDopInputSocket *to_socket;

    UiDopLink(UiDopOutputSocket *from_socket, UiDopInputSocket *to_socket)
        : from_socket(from_socket), to_socket(to_socket)
    {
        from_socket->attach_link(this);
        to_socket->attach_link(this);
        selectable = true;
        zvalue = 1.f;
    }

    Point get_from_position() const override {
        return from_socket->position + from_socket->get_parent()->position;
    }

    Point get_to_position() const override {
        return to_socket->position + to_socket->get_parent()->position;
    }

    UiDopGraph *get_parent() const {
        return (UiDopGraph *)(parent);
    }
};


struct UiDopPendingLink : GraphicsLineItem {
    UiDopSocket *socket;

    UiDopPendingLink(UiDopSocket *socket)
        : socket(socket)
    {
        zvalue = 3.f;
    }

    Color get_line_color() const override {
        return {0.75f, 0.5f, 0.375f};
    }

    Point get_from_position() const override {
        return socket->position + socket->get_parent()->position;
    }

    Point get_to_position() const override {
        return {cur.x, cur.y};
    }

    UiDopGraph *get_parent() const {
        return (UiDopGraph *)(parent);
    }

    Widget *item_at(Point p) const override {
        return nullptr;
    }
};
