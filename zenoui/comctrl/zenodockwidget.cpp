#include "zenodockwidget.h"


ZenoDockWidget::ZenoDockWidget(const QString &title, QWidget *parent, Qt::WindowFlags flags)
    : _base(title, parent, flags) {
    init();
}

ZenoDockWidget::ZenoDockWidget(QWidget *parent, Qt::WindowFlags flags)
    : _base(parent, flags) {
    init();
}

ZenoDockWidget::~ZenoDockWidget()
{

}

void ZenoDockWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(50, 50, 50));
    _base::paintEvent(event);
}

void ZenoDockWidget::init()
{
    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(38, 38, 38));
    palette.setBrush(QPalette::WindowText, QColor());
    setPalette(palette);
}