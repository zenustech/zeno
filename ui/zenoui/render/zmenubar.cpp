#include "zmenubar.h"
#include "zmenu.h"


ZMenuBar::ZMenuBar(QWidget* parent)
    : QMenuBar(parent)
{
    QPalette palette;
    palette.setBrush(QPalette::Window, QColor(58, 58, 58));
    setPalette(palette);
}

void ZMenuBar::paintEvent(QPaintEvent* event)
{
    QMenuBar::paintEvent(event);
}

void ZMenuBar::mousePressEvent(QMouseEvent* event)
{
    QMenuBar::mousePressEvent(event);
}