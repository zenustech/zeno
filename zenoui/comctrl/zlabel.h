#ifndef __ZLABEL_H__
#define __ZLABEL_H__

#include <QLabel>
#include <QIcon>

class ZIconLabel : public QLabel
{
    Q_OBJECT
public:
    ZIconLabel(QWidget* pLabel = nullptr);
    void setIcons(const QSize& sz, const QString& iconEnable, const QString& iconHover, const QString& iconNormalOn = QString(), const QString& iconHoverOn = QString(), const QString& iconDisable = QString());

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

protected:
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

signals:
    void clicked();

private:
    QColor m_normal;
};

#endif