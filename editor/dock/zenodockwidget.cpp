#include "zenodockwidget.h"


ZenoDockWidget::ZenoDockWidget(const QString& title, QWidget* parent, Qt::WindowFlags flags)
    : QDockWidget(title, parent, flags)
{
}

ZenoDockWidget::ZenoDockWidget(QWidget* parent, Qt::WindowFlags flags)
    : QDockWidget(parent, flags)
{
}

ZenoDockWidget::~ZenoDockWidget()
{

}