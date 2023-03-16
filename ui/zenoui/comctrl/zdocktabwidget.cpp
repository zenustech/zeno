#include "zdocktabwidget.h"
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <QtSvg/QSvgRenderer>
#include <zeno/utils/log.h>
#include <zenoui/comctrl/zaddtabbar.h>
#include <zenoui/comctrl/zicontoolbutton.h>
#include <zenoedit/zenoapplication.h>


ZDockTabWidget::ZDockTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    initStyleSheet();

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor("#2D3239"));
    setAutoFillBackground(true);
    setPalette(pal);

    ZAddTabBar* pTabbar = new ZAddTabBar;
    setTabBar(pTabbar);
    connect(pTabbar, SIGNAL(addBtnClicked()), this, SIGNAL(addClicked()));
    connect(pTabbar, SIGNAL(layoutBtnClicked()), this, SIGNAL(layoutBtnClicked()));

    connect(pTabbar, &ZAddTabBar::tabCloseRequested, this, [=](int index) {
        removeTab(index);
        emit tabClosed(index);
    });

    setDocumentMode(true);
    setMouseTracking(true);

    QFont font = zenoApp->font();
    font.setPointSize(10);
    tabBar()->setFont(font);
    tabBar()->setDrawBase(false);
    tabBar()->setMouseTracking(true);
    tabBar()->installEventFilter(this);
}

ZDockTabWidget::~ZDockTabWidget()
{
}

void ZDockTabWidget::enterEvent(QEvent* event)
{
    QTabWidget::enterEvent(event);
}

void ZDockTabWidget::mousePressEvent(QMouseEvent* event)
{
    QTabWidget::mousePressEvent(event);
}

void ZDockTabWidget::mouseReleaseEvent(QMouseEvent* event)
{
    QTabWidget::mouseReleaseEvent(event);
}

void ZDockTabWidget::mouseMoveEvent(QMouseEvent* event)
{
    QTabWidget::mouseMoveEvent(event);
}

bool ZDockTabWidget::eventFilter(QObject* watched, QEvent* event)
{
    return QTabWidget::eventFilter(watched, event);
}

void ZDockTabWidget::leaveEvent(QEvent* event)
{
    QTabWidget::leaveEvent(event);
}

void ZDockTabWidget::initStyleSheet()
{
}

void ZDockTabWidget::paintEvent(QPaintEvent* e)
{
    QTabWidget::paintEvent(e);
}