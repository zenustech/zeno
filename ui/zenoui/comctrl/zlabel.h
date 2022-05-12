#ifndef __ZLABEL_H__
#define __ZLABEL_H__

#include <QtWidgets>

class ZIconLabel : public QLabel
{
    Q_OBJECT
public:
    ZIconLabel(QWidget* pLabel = nullptr);
    void setIcons(const QSize& sz, const QString& iconEnable, const QString& iconHover, const QString& iconNormalOn = QString(), const QString& iconHoverOn = QString(), const QString& iconDisable = QString());
    void toggle(bool bToggle = true);

protected:
    void paintEvent(QPaintEvent* e) override;
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

signals:
    void clicked();
    void toggled(bool);

private:
    void onClicked();
    void onToggled();
    void updateIcon();

    QIcon m_icon;
    QSize m_iconSz;
    bool m_bToggled;
    bool m_bHovered;
    bool m_bClicked;
    bool m_bToggleable;
};

class ZTextLabel : public QLabel
{
    Q_OBJECT
public:
    ZTextLabel(QWidget* parent = nullptr);
    ZTextLabel(const QString& text, QWidget* parent = nullptr);
    void setTextColor(const QColor& clr);
    void setBackgroundColor(const QColor& clr);
    void setUnderline(bool bUnderline);
    void setUnderlineOnHover(bool bUnderline);

protected:
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

signals:
    void clicked();
    void rightClicked();

private:
    QColor m_normal;
    bool m_bUnderlineHover;
    bool m_bUnderline;
};

#endif