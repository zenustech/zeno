#ifndef __ZENO_STYLE_H__
#define __ZENO_STYLE_H__

#include <QtWidgets>

class ZStyleOptionToolButton;

class ZenoStyle : public QProxyStyle
{
    Q_OBJECT
    typedef QProxyStyle base;
public:
    enum ZenoComplexControl
    {
        CC_ZenoToolButton = CC_CustomBase + 1,
        CC_ZenoComboBox,
    };
    enum ZenoSubControl
    {
        SC_ZenoToolButtonIcon = SC_CustomBase + 1,
        SC_ZenoToolButtonText,
        SC_ZenoToolButtonArrow
    };
    enum ZenoPixmetrics
    {
        PM_ButtonLeftMargin = PM_CustomBase + 1,
        PM_ButtonTopMargin,
        PM_ButtonRightMargin,
        PM_ButtonBottomMargin
    };

    ZenoStyle();
    ~ZenoStyle();

    void drawComplexControl(ComplexControl control, const QStyleOptionComplex* option, QPainter* painter, const QWidget* widget = nullptr) const override;
    void drawPrimitive(PrimitiveElement pe, const QStyleOption* opt, QPainter* p,
        const QWidget* w = nullptr) const override;
    void drawControl(ControlElement element, const QStyleOption* opt, QPainter* p,
        const QWidget* w = nullptr) const override;
    QRect subControlRect(ComplexControl cc, const QStyleOptionComplex* opt, SubControl sc, const QWidget* widget) const override;
    int styleHint(StyleHint sh, const QStyleOption* opt = nullptr, const QWidget* w = nullptr,
        QStyleHintReturn* shret = nullptr) const override;
    int pixelMetric(PixelMetric m, const QStyleOption* opt = nullptr, const QWidget* widget = nullptr) const override;
    QRect subElementRect(SubElement element, const QStyleOption* option, const QWidget* widget) const override;
    void drawItemText(QPainter* painter, const QRect& rect, int flags, const QPalette& pal, bool enabled,
        const QString& text, QPalette::ColorRole textRole = QPalette::NoRole) const override;

private:
    void drawZenoLineEdit(PrimitiveElement pe, const QStyleOption* opt, QPainter* p, const QWidget* widget) const;
    void drawZenoToolButton(const ZStyleOptionToolButton* option, QPainter* painter, const QWidget* widget) const;
    void drawDropdownArrow(QPainter* painter, QRect downArrowRect) const;
    void drawNewItemMenu(const QStyleOptionMenuItem* menuitem, QPainter* p, const QWidget* w) const;
    void drawMenuItem(ControlElement element, const QStyleOption* opt, QPainter* p, const QWidget* w) const;
};

#endif
