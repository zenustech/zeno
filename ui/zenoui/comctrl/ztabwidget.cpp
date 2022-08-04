#include "ztabwidget.h"


ZTabWidget::ZTabWidget(QWidget* parent)
    : QWidget(parent)
    , m_stack(nullptr)
    , m_pTabbar(nullptr)
{
    init();
}

ZTabWidget::~ZTabWidget()
{

}

void ZTabWidget::init()
{
    m_stack = new QStackedWidget(this);
    m_stack->setLineWidth(0);
    m_stack->setSizePolicy(QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred, QSizePolicy::TabWidget));
    connect(m_stack, SIGNAL(widgetRemoved(int)), this, SLOT(_removeTab(int)));
}

int ZTabWidget::addTab(const QString& label, const QIcon& icon)
{
    return -1;
}

int ZTabWidget::addTab(const QString& label)
{
    return -1;
}

void ZTabWidget::removeTab(int index)
{

}

void ZTabWidget::_removeTab(int index)
{
    
}

QString ZTabWidget::tabText(int index) const
{
    return m_pTabbar->tabText(index);
}

void ZTabWidget::setTabText(int index, const QString& text)
{

}

void ZTabWidget::setCurrentIndex(int index)
{

}

void ZTabWidget::setCurrentIndex(const QString& text)
{

}

void ZTabWidget::paintEvent(QPaintEvent* event)
{

}
