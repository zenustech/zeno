#include "zdocktabwidget.h"
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <QtSvg/QSvgRenderer>
#include <zeno/utils/log.h>


ZDockTabWidget::ZDockTabWidget(QWidget* parent)
    : QTabWidget(parent)
    , m_bHovered(false)
{
    initStyleSheet();

    ZIconLabel* pMoreBtn = new ZIconLabel;
    pMoreBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(27, 27)), ":/icons/more.svg", ":/icons/more_on.svg");

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor("#22252C"));
    setAutoFillBackground(true);
    setPalette(pal);

    setCornerWidget(pMoreBtn);
    pal = tabBar()->palette();
    pal.setBrush(QPalette::Dark, QColor(24, 29, 33));
    tabBar()->setPalette(pal);

    //QToolButton* tb = new QToolButton();
    //tb->setText("+");
    //int nTabs = tabBar()->count();
    //addTab(new QLabel("Add tabs by pressing \"+\""), QString());
    //setTabEnabled(nTabs, false);
    //tabBar()->setTabButton(nTabs, QTabBar::RightSide, tb);

    setDocumentMode(false);
    this->setMouseTracking(true);
    tabBar()->setMouseTracking(true);
    tabBar()->installEventFilter(this);
}

ZDockTabWidget::~ZDockTabWidget()
{
}

int ZDockTabWidget::addTab(QWidget* widget, const QString& label)
{
    //widget->installEventFilter(this);
    return QTabWidget::addTab(widget, label);
}

int ZDockTabWidget::addTab(QWidget* widget, const QIcon& icon, const QString& label)
{
    //widget->installEventFilter(this);
    return QTabWidget::addTab(widget, icon, label);
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
    if (buttonRect().contains(event->pos()))
    {
        emit addClicked();
    }
}

void ZDockTabWidget::mouseMoveEvent(QMouseEvent* event)
{
    QTabWidget::mouseMoveEvent(event);
    QPoint pt = event->pos();
    QRect rc = buttonRect();
    m_bHovered = rc.contains(pt);
    update();
}

bool ZDockTabWidget::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == tabBar() && event->type() == QEvent::MouseMove)
    {
        QMouseEvent* pMouseEvent = static_cast<QMouseEvent*>(event);
        QPoint pt = pMouseEvent->pos();
        QRect rc = buttonRect();
        m_bHovered = rc.contains(pt);
        update();
    }
    return QTabWidget::eventFilter(watched, event);
}

void ZDockTabWidget::leaveEvent(QEvent* event)
{
    QTabWidget::leaveEvent(event);
    m_bHovered = false;
    update();
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
	            font: 12px;\
	            /*margin-right: 1px;*/\
            }\
            \
            QTabBar::tab:first {\
                border-left: 0px solid rgb(24, 29, 33);\
            }\
            \
            QTabBar::tab:top {\
	            /*margin-right: 1px;*/\
                padding: 7px 16px 7px 16px;\
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

QRect ZDockTabWidget::buttonRect()
{
    int w = tabBar()->width();
    int h = tabBar()->height();
    int x = tabBar()->x() + w;
    int y = tabBar()->y();

    int buttonWidth = h;
    int buttonHeight = h;
    int xoffset = ZenoStyle::dpiScaled(7);
    int yoffset = ZenoStyle::dpiScaled(3);

    QRect rc(QPoint(x, y), QPoint(x + buttonWidth - 1, y + buttonHeight - 1));
    return rc;
}

void ZDockTabWidget::paintEvent(QPaintEvent* e)
{
    QTabWidget::paintEvent(e);

    //draw add button
    int w = tabBar()->width();
    int h = tabBar()->height();
    int x = tabBar()->x() + w;
    int y = tabBar()->y();

    int buttonWidth = h;
    int buttonHeight = h;

    QPainter p(this);
    p.save();

    p.setPen(QColor(24, 29, 33));
    p.setBrush(Qt::NoBrush);

    QLine l1(QPoint(x, y + buttonHeight - 1), QPoint(x + buttonWidth - 1, y + buttonHeight - 1));
    QLine l2(QPoint(x + buttonWidth - 1, y + buttonHeight - 1), QPoint(x + buttonWidth - 1, y));

    QVector<QLine> lines;
    lines.append(l1);
    lines.append(l2);
    p.drawLines(lines);

    int xoffset = ZenoStyle::dpiScaled(7);
    int yoffset = ZenoStyle::dpiScaled(3);

    QRect rc = buttonRect();
    rc.adjust(xoffset, xoffset, -xoffset, -xoffset);

    QString iconPath = m_bHovered ? ":/icons/add-on.svg" : ":/icons/add.svg";
    QPixmap px(iconPath);

    p.drawPixmap(rc, px);
    p.restore();
}