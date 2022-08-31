#include "zdocktabwidget.h"
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <QtSvg/QSvgRenderer>
#include <zeno/utils/log.h>
#include <zenoui/comctrl/zaddtabbar.h>
#include <zenoui/comctrl/zicontoolbutton.h>


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

    setDocumentMode(true);
    setMouseTracking(true);

    QFont font("Segoe UI Bold", 10);
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
    setStyleSheet("\
            QTabBar {\
                background-color: #22252C;\
                border-bottom: 1px solid rgb(24, 29, 33);\
                border-right: 0px;\
            }\
            \
            QTabBar::tab {\
                background: #22252C;\
                color: #737B85;\
                border-top: 0px solid rgb(24,29,33);\
                border-right: 1px solid rgb(24, 29, 33);\
                border-bottom: 1px solid rgb(24, 29, 33);\
            }\
            \
            QTabBar::tab:top {\
                padding: 2px 16px 3px 16px;\
            }\
            \
            QTabBar::tab:selected {\
                background: #2D3239;\
	            color: #C3D2DF;\
                border-bottom: 0px;\
            }\
            \
            QTabBar::close-button {\
                image: url(:/icons/closebtn.svg);\
                subcontrol-position: right;\
            }\
            QTabBar::close-button:hover {\
                image: url(:/icons/closebtn_on.svg);\
            }"
    );
}

void ZDockTabWidget::paintEvent(QPaintEvent* e)
{
    QTabWidget::paintEvent(e);
}