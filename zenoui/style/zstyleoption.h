#ifndef __ZSTYLEOPTION_H__
#define __ZSTYLEOPTION_H__

#include <QtWidgets>

class ZStyleOptionToolButton : public QStyleOptionToolButton
{
public:
    enum ArrowOption
    {
        NO_ARROW,
        DOWNARROW,
        RIGHTARROW,
    };

    enum IconOption
    {
        NO_ICON,
        ICON_UP,
        ICON_LEFT
    };

    QString text;
    int buttonOpts;
    int roundCorner;
    Qt::Orientation orientation;
    bool hideText;
    bool buttonEnabled;
    bool bDown;
    bool bTextUnderIcon;
    ArrowOption m_arrowOption;
    IconOption m_iconOption;
    QColor textColor;
    QColor textHoverColor;
    QColor textDownColor;
    QColor borderColor;
    QColor borderInColor;
    QColor hoveredBgColor;
    QColor selectedBgColor;
    QColor ActiveBgColor;
    QBrush bgBrush;

    ZStyleOptionToolButton();
    ZStyleOptionToolButton(const ZStyleOptionToolButton& other) { *this = other; }
};

class ZStyleOptionComboBox : public QStyleOptionComboBox
{
public:
    ZStyleOptionComboBox();
    ZStyleOptionComboBox(const QStyleOptionComboBox &opt);

    //border
    QColor bdrNormal, bdrHoverd, bdrSelected;

    //line edit text
    QColor clrText;

    //background
    QColor clrBackground;
    QColor clrBgHovered;

    //arrow button
    QColor btnNormal, btnHovered, btnDown;

    //item
    QColor itemNormal, itemHovered, itemText;

    int textMargin;
};

#endif