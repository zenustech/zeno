#include "zmenu.h"


ZMenu::ZMenu(QWidget* parent)
    : QMenu(parent)
{
}

ZMenu::ZMenu(const QString& title, QWidget* parent)
    : QMenu(title, parent)
{
}

void ZMenu::paintEvent(QPaintEvent* event)
{
    QMenu::paintEvent(event);
}