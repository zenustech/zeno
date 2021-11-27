#pragma once


#include <z2/UI/GraphicsWidget.h>
#include <z2/UI/Color.h>


namespace z2::UI {


struct GraphicsLineItem : GraphicsWidget {
    static constexpr float LW = 5.f;

    virtual Point get_from_position() const = 0;
    virtual Point get_to_position() const = 0;

    bool contains_point(Point p) const override;
    virtual Color get_line_color() const;
    void paint() const override;
};


}  // namespace z2::UI
