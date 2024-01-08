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
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    void initStyleOption(ZStyleOptionCheckBoxBar* option);

    Qt::CheckState m_checkState;
    bool m_bHover;
};

#endif
