#ifndef __ZLABEL_H__
#define __ZLABEL_H__

#include <QtWidgets>

class ZDebugLabel : public QLabel
{
    Q_OBJECT
public:
    ZDebugLabel(QWidget* parent = nullptr);
    ZDebugLabel(const QString& text, QWidget* parent = nullptr);
    void adjustText(const QString& text);

private:
    QString m_text;
};

class ZIconLabel : public QLabel
{
    Q_OBJECT
public:
    ZIconLabel(QWidget* pLabel = nullptr);
    void setIcons(const QSize& sz, const QString& iconEnable, const QString& iconHover, const QString& iconNormalOn = QString(), const QString& iconHoverOn = QString(), const QString& iconDisable = QString());
    void setIcons(const QString& iconIdle, const QString& iconLight);
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
    void setHoverCursor(Qt::CursorShape shape);
    void setTextColor(const QColor& clr);
    void setBackgroundColor(const QColor& clr);
    void setUnderline(bool bUnderline);
    void setUnderlineOnHover(bool bUnderline);
    void setTransparent(bool btransparent);
    void setEnterCursor(Qt::CursorShape shape);

protected:
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

signals:
    void clicked();
    void rightClicked();

private:
    QColor m_normal;
    Qt::CursorShape m_hoverCursor;
    bool m_bUnderlineHover;
    bool m_bUnderline;
};

#endif