#ifndef __ZENO_STYLE_H__
#define __ZENO_STYLE_H__

#include <QtWidgets>

class ZStyleOptionToolButton;

class ZenoStyle : public QProxyStyle
{
    Q_OBJECT
    typedef QProxyStyle base;
public:
    //temp: remove these enum later, because it causes chaos and infinite recursive.
    enum ZenoComplexControl
    {
        CC_ZenoToolButton = CC_MdiControls + 1,
        CC_ZenoComboBox,        //-268435454
        CC_ZenoCheckBoxBar,     //-268435453
    };
    enum ZenoSubControl
    {
        SC_ZenoToolButtonIcon = SC_MdiCloseButton + 1, //-268435455
        SC_ZenoToolButtonText,  //-268435454
        SC_ZenoToolButtonArrow
    };
    enum ZenoPixmetrics
    {
        PM_ButtonLeftMargin = PM_TitleBarButtonSize + 1,
        PM_ButtonTopMargin,
        PM_ButtonRightMargin,
        PM_ButtonBottomMargin,
        PM_IconTextSpacing,
    };
    enum ZenoPrimitiveElement
    {
        PE_ComboBoxDropdownButton = PE_IndicatorTabTearRight + 1,
        PE_ComboBoxLineEdit,
    };

    enum ZenoControlElement {
        CE_ZenoComboBoxLabel = CE_ShapedFrame + 1,
        CE_TabBarTabUnderline,
        CE_TabBarTabCloseBtn,
    };

    ZenoStyle();
    ~ZenoStyle();

    static qreal dpiScaled(qreal value);
    static QSize dpiScaledSize(const QSize &value);
    static QSizeF dpiScaledSize(const QSizeF& sz);
    static QMargins dpiScaledMargins(const QMargins& margins);
    static QString dpiScaleSheet(const QString &sheet);

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
