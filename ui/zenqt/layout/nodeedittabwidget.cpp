#include "nodeedittabwidget.h"
#include "widgets/zlabel.h"
#include "style/zenostyle.h"
#include <QtSvg/QSvgRenderer>
#include <zeno/utils/log.h>
#include "widgets/zaddtabbar.h"
#include "widgets/zicontoolbutton.h"


NodeEditorTabWidget::NodeEditorTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    setAutoFillBackground(true);
    QPalette pal;
    pal.setColor(QPalette::Window, QColor(255, 0, 0));
    pal.setBrush(QPalette::Background, QColor(255, 0, 0));
    setPalette(pal);
}

NodeEditorTabWidget::~NodeEditorTabWidget()
{
}

void NodeEditorTabWidget::enterEvent(QEvent* event)
{
    QTabWidget::enterEvent(event);
}

void NodeEditorTabWidget::mousePressEvent(QMouseEvent* event)
{
    QTabWidget::mousePressEvent(event);
}

void NodeEditorTabWidget::mouseReleaseEvent(QMouseEvent* event)
{
    QTabWidget::mouseReleaseEvent(event);
}

void NodeEditorTabWidget::mouseMoveEvent(QMouseEvent* event)
{
    QTabWidget::mouseMoveEvent(event);
}

bool NodeEditorTabWidget::eventFilter(QObject* watched, QEvent* event)
{
    return QTabWidget::eventFilter(watched, event);
}

void NodeEditorTabWidget::leaveEvent(QEvent* event)
{
    QTabWidget::leaveEvent(event);
}

void NodeEditorTabWidget::paintEvent(QPaintEvent* e)
{
    //节点编辑器的tabbar旁边的空白区域的背景色，根本没法在qss指定，已经被dock的css控制了，导致新建的浮动控件没有这个css颜色，所以显示为白色
    //因此只能强行在paintEvent写死
    QPainter painter(this);
    painter.fillRect(rect(), QColor(31, 31, 31));
}