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

    //ZIconToolButton* pMoreBtn = new ZIconToolButton(":/icons/layout.svg", ":/icons/layout-on.svg");
    //setCornerWidget(pMoreBtn);
    pal = tabBar()->palette();
    pal.setBrush(QPalette::Dark, QColor(255,0,0));
    tabBar()->setPalette(pal);

    setDocumentMode(true);
    setMouseTracking(true);
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
	            font-weight: bold;\
                background-color: #22252C;\
                border-bottom: 1px solid rgb(24, 29, 33);\
                border-right: 0px;\
            }\
            \
            QTabWidget::pane { /* The tab widget frame */\
                border: 1px;\
                background-color: rgb(255,255,255);\
            }\
            \
            QTabBar::tab {\
                background: #22252C;\
                color: #737B85;\
                border-top: 0px solid rgb(24,29,33);\
                border-right: 1px solid rgb(24, 29, 33);\
                border-bottom: 1px solid rgb(24, 29, 33);\
                font: 10pt 'Segoe UI Bold';\
                /*margin-right: 1px;*/\
            }\
            \
            QTabBar::tab:first {\
                border-left: 0px solid rgb(24, 29, 33);\
            }\
            \
            QTabBar::tab:top {\
                /*margin-right: 1px;*/\
                padding: 2px 16px 3px 16px;\
            }\
            \
            QTabBar::tab:top:first {\
                margin-left: 0px;\
            }\
            \
            QTabBar::tab:!selected { font-weight: normal; }\
            \
            QTabBar::tab:selected {\
                background: #2D3239;\
	            color: #C3D2DF;\
                border-right: 1px solid rgb(24, 29, 33);\
                border-top: 0px solid rgb(24, 29, 33);\
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