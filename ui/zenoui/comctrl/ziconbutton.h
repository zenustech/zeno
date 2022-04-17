#ifndef __ZICONBUTTON_H__
#define __ZICONBUTTON_H__

#include <QLabel>
#include <QIcon>

class ZIconButton : public QLabel
{
    Q_OBJECT
    typedef QLabel _base;
    enum BTN_STATE {
        STATE_NORMAL,
        STATE_HOVERED,
        STATE_CLICKED,
    };

public:
    ZIconButton(QIcon icon, const QSize &sz, const QColor &hoverClr, const QColor &selectedClr, bool bCheckable = false, QWidget *pLabel = nullptr);
    void setChecked(bool bChecked);
    QSize sizeHint() const override;

protected:
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent* event) override;

signals:
    void clicked();
    void toggled(bool);

private:
    QIcon m_icon;
    QSize m_size;
    QColor m_hover;
    QColor m_selected;
    int m_state;
    bool m_bCheckable;
    bool m_bChecked;
};

#endif