#ifndef __ZTOOLBUTTON_H__
#define __ZTOOLBUTTON_H__

class ZStyleOptionToolButton;

#include <QtWidgets>

class ZToolButton : public QWidget
{
    Q_OBJECT
public:
    struct AnimationInfo {
      QColor mBackColor;

      bool mOnOff;
      int mAnimationPeriod;
      QVariantAnimation *posAnimation;

      QPointF m_RightPos;
      QPointF m_LeftPos; 
      QRectF mButtonRect;
      float BtnWidth;
    };
    enum ButtonOption
    {
        Opt_Undefined = 0x0000,
        Opt_HasIcon = 0x0001 << 0,
        Opt_HasText = 0x0001 << 1,
        Opt_DownArrow = 0x0001 << 2,
        Opt_RightArrow = 0x0001 << 3,
        Opt_TextUnderIcon = 0x0001 << 4,
        Opt_TextRightToIcon = 0x0001 << 5,
        Opt_TextLeftToIcon = 0x0001 << 6,
        Opt_Checkable = 0x0001 << 7,
        Opt_NoBackground = 0x0001 << 8,
        Opt_UpRight = 0x0001 << 9,
        Opt_SwitchAnimation = 0x0001 << 10
    };
    ZToolButton(QWidget* parent = nullptr);
    ZToolButton(
        int option,
        const QIcon& icon = QIcon(),
        const QSize& iconSize = QSize(),
        const QString& text = QString(),
        QWidget* parent = nullptr);
    ZToolButton(
        int option,
        const QString& icon = "",
        const QString& iconOn = "",
        const QSize& iconSize = QSize(),
        const QString& text = QString(),
        QWidget* parent = nullptr);
    virtual ~ZToolButton();

public:
    QString text() const;
    QIcon icon() const;
    QSize iconSize() const;
    QMargins margins() const;
    void setCheckable(bool bCheckable);
    void toggle(bool bOn);
    bool isChecked() const;
    bool isDown() const;
    bool isPressed() const;
    bool isHovered() const;
    int buttonOption() const;
    virtual QSize sizeHint() const override;
    void setBackgroundClr(const QColor& normalClr, const QColor& hoverClr, const QColor& downClr, const QColor& checkedClr);
    void setTextClr(const QColor &normal, const QColor &hover, const QColor &normalOn, const QColor &hoverOn);
    void setMargins(const QMargins& margins);
    void setRadius(int radius);
    void setFont(const QFont& font);
    void initAnimation();

public slots:
    void setText(const QString& text);
    void setIcon(const QSize& size, QString icon, QString iconHover, QString iconOn, QString iconOnHover);
    void setIconSize(const QSize& size);
    void setChecked(bool bChecked);
    void setDown(bool bDown);
    void setButtonOptions(int style);
    void setShortcut(QKeySequence);
    virtual void updateIcon();

signals:
    void clicked();
    void toggled(bool);
    void LButtonClicked();
    void LButtonPressed();
    void RButtonClicked();
    void RButtonPressed();

protected:
    virtual bool event(QEvent* e) override;
    virtual void enterEvent(QEvent* e) override;
    virtual void leaveEvent(QEvent* e) override;
    virtual void paintEvent(QPaintEvent* event) override;
    virtual void initStyleOption(ZStyleOptionToolButton* option) const;
    virtual void mousePressEvent(QMouseEvent* e) override;
    virtual void mouseReleaseEvent(QMouseEvent* e) override;

    virtual void initColors(ZStyleOptionToolButton* option) const;
    virtual QBrush backgrondColor(QStyle::State state) const;

private:
    void setCustomTip(QString tip);
    void setPressed(bool bPressed);
    QString getCustomTip() const;
    QBrush textColor(QStyle::State state) const;
    void initDefault();

    QMargins m_margins;
    QString m_text;
    QString m_customTip;
    QSize m_iconSize;
    QIcon m_icon;
    QIcon m_iconOn;
    QFont m_font;

    QColor m_clrBgNormalHover;
    QColor m_clrBgOn;
    QColor m_clrBgOnHovered;
    QColor m_clrBgNormal;

    QColor m_clrText, m_clrTextHover, m_clrTextOn, m_clrTextOnHover;

    int m_radius;
    int m_options;
    int m_roundCorner;
    bool m_bHideText;
    
    bool m_bHovered;
    bool m_bChecked;
    bool m_bDown;
    bool m_bPressed;

    AnimationInfo animInfo;
};

#endif
