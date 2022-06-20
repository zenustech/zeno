#ifndef __ZCHECKBOX_BAR_H__
#define __ZCHECKBOX_BAR_H__

#include <QtWidgets>

class ZStyleOptionCheckBoxBar;

class ZCheckBoxBar : public QWidget
{
    Q_OBJECT
public:
    ZCheckBoxBar(QWidget* parent = nullptr);
    Qt::CheckState checkState() const;
    void setCheckState(Qt::CheckState state);

signals:
    void stateChanged(int);

protected:
    void paintEvent(QPaintEvent* event);
    void enterEvent(QEvent *event);
    void leaveEvent(QEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

private:
    void initStyleOption(ZStyleOptionCheckBoxBar* option);

    Qt::CheckState m_checkState;
    bool m_bHover;
};

#endif