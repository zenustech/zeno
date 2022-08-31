#include "zaddtabbar.h"
#include <zenoui/comctrl/zicontoolbutton.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>


ZAddTabBar::ZAddTabBar(QWidget* parent)
    : QTabBar(parent)
{
    m_pAddBtn = new ZIconLabel(this);
    m_pAddBtn->setIcons(":/icons/addpanel.svg", ":/icons/addpanel-on.svg");

    m_pLayoutBtn = new ZIconLabel(this);
    m_pLayoutBtn->setIcons(":/icons/layout.svg", ":/icons/layout-on.svg");

    setDrawBase(false);

    connect(m_pAddBtn, SIGNAL(clicked()), this, SIGNAL(addBtnClicked()));
    connect(m_pLayoutBtn, SIGNAL(clicked()), this, SIGNAL(layoutBtnClicked()));
}

ZAddTabBar::~ZAddTabBar()
{
}

QSize ZAddTabBar::sizeHint() const
{
    QSize sz = QTabBar::sizeHint();
    return QSize(sz.width(), sz.height());
}

void ZAddTabBar::resizeEvent(QResizeEvent* event)
{
    QTabBar::resizeEvent(event);
    setGeomForAddBtn();
    setGeomForLayoutBtn();
}

void ZAddTabBar::tabLayoutChange()
{
    QTabBar::tabLayoutChange();
    setGeomForAddBtn();
    setGeomForLayoutBtn();
}

void ZAddTabBar::mousePressEvent(QMouseEvent* e)
{
    QTabBar::mousePressEvent(e);
}

void ZAddTabBar::setGeomForAddBtn()
{
    int nWidth = 0;
    for (int i = 0; i < this->count(); i++)
    {
        nWidth += this->tabRect(i).width();
    }

    QSize sz = m_pAddBtn->sizeHint();

    int h = geometry().top();
    int w = this->width();
    int x = 0;
    static int sLeftMargin = ZenoStyle::dpiScaled(5);
    if (nWidth > w)
        x = w - 54;
    else
        x = nWidth + sLeftMargin;
    int y = geometry().top() + (geometry().height() - sz.height()) / 2 + 1;

    m_pAddBtn->move(x, y);
}

void ZAddTabBar::setGeomForLayoutBtn()
{
    QSize sz = m_pLayoutBtn->sizeHint();
    int x = geometry().right() - sz.width();
    int y = geometry().top() + (geometry().height() - sz.height()) / 2;
    m_pLayoutBtn->move(x, y);
}
