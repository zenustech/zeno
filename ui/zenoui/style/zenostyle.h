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
        CC_ZenoCheckBoxBar,
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
    enum ZenoPrimitiveElement
    {
        PE_ComboBoxDropdownButton = PE_CustomBase + 1,
        PE_ComboBoxLineEdit,
    };

    enum ZenoControlElement {
        CE_ZenoComboBoxLabel = CE_CustomBase + 1,
        CE_TabBarTabUnderline,
        CE_TabBarTabCloseBtn,
    };

    ZenoStyle();
    ~ZenoStyle();

    static qreal dpiScaled(qreal value);
    static QSize dpiScaledSize(const QSize &value);

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
    QSize sizeFromContents(ContentsType type, const QStyleOption* option, const QSize& size, const QWidget* widget) const override;

private:
    void drawZenoToolButton(const ZStyleOptionToolButton* option, QPainter* painter, const QWidget* widget) const;
    void drawDropdownArrow(QPainter* painter, QRect downArrowRect, bool isDown) const;
    void drawCheckBox(QPainter* painter, QRect rect, bool bHover, Qt::CheckState state) const;
};

#endif
