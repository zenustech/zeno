#ifndef __ZTOOLBUTTON_H__
#define __ZTOOLBUTTON_H__

class ZStyleOptionToolButton;

#include <QtWidgets>

class ZToolButton : public QWidget
{
    Q_OBJECT
public:
    enum ButtonOption
    {
        Opt_Undefined = 0x0000,
        Opt_HasIcon = 0x0001 << 0,
        Opt_HasText = 0x0001 << 1,
        Opt_DownArrow = 0x0001 << 2,
        Opt_RightArrow = 0x0001 << 3,
        Opt_TextUnderIcon = 0x0001 << 4,
        Opt_TextRightToIcon = 0x0001 << 5,
        Opt_Checkable = 0x0001 << 6,
        Opt_NoBackground = 0x0001 << 7,
        Opt_UpRight = 0x0001 << 8
    };
    ZToolButton(
        int option,
        const QIcon& icon = QIcon(),
        const QSize& iconSize = QSize(),
        const QString& text = QString(),
        QWidget* parent = nullptr);
    virtual ~ZToolButton();

public:
    QString text() const;
    QIcon icon() const;
    QSize iconSize() const;
    void setCheckable(bool bCheckable);
    bool isChecked() const;
    bool isDown() const;
    bool isPressed() const;
    bool isHovered() const;
    int buttonOption() const;
    virtual QSize sizeHint() const;
    void setBackgroundClr(const QColor& hoverClr, const QColor& downClr, const QColor& checkedClr);

public slots:
    void setText(const QString& text);
    void setIcon(const QIcon& icon);
    void setIconSize(const QSize& size);
    void showToolTip();
    void setChecked(bool bChecked);
    void setDown(bool bDown);
    void setButtonOptions(int style);
    void setShortcut(QString);
    virtual void updateIcon();

signals:
    void clicked();
    void LButtonClicked();
    void LButtonPressed();
    void RButtonClicked();
    void RButtonPressed();

protected:
    virtual bool event(QEvent* e);
    virtual void enterEvent(QEvent* e);
    virtual void leaveEvent(QEvent* e);
    virtual void paintEvent(QPaintEvent* event);
    virtual void initStyleOption(ZStyleOptionToolButton* option) const;
    virtual void mousePressEvent(QMouseEvent* e);
    virtual void mouseReleaseEvent(QMouseEvent* e);

    virtual void initColors(ZStyleOptionToolButton* option) const;
    virtual QBrush backgrondColor(QStyle::State state) const;

private:
    void setCustomTip(QString tip);
    void setPressed(bool bPressed);
    QString getCustomTip() const;

    QString m_text;
    QString m_customTip;
    QSize m_iconSize;
    QIcon m_icon;
    QFont m_font;

    QColor m_clrBgHover;
    QColor m_clrBgDown;
    QColor m_clrBgChecked;

    int m_options;
    int m_roundCorner;
    bool m_bHideText;
    
    bool m_bHovered;
    bool m_bChecked;
    bool m_bDown;
    bool m_bPressed;
};

#endif
