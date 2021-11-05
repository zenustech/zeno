#ifndef __ZLABEL_H__
#define __ZLABEL_H__

#include <QLabel>
#include <QIcon>

class ZLabel : public QLabel
{
    Q_OBJECT
public:
    ZLabel(QWidget* pLabel = nullptr);
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

#endif