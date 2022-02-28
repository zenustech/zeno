#ifndef __ZMENUBAR_H__
#define __ZMENUBAR_H__

#include <QtWidgets>

class ZMenuBar : public QMenuBar
{
    Q_OBJECT
public:
    ZMenuBar(QWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
};

#endif