#include "ztabbar.h"
#include "ziconbutton.h"
#include "../style/zenostyle.h"


ZTabBar::ZTabBar(QWidget* parent)
    : QWidget(parent)
    , m_currentIndex(-1)
    , m_bCloseBtn(true)
{

}

ZTabBar::~ZTabBar()
{

}

Tab ZTabBar::at(int index)
{
    return m_tabList.at(index);
}

int ZTabBar::addTab(const QString& text)
{
    return insertTab(-1, text);
}

int ZTabBar::addTab(const QIcon& icon, const QString& text)
{
    return insertTab(-1, icon, text);
}

int ZTabBar::insertTab(int index, const QString& text)
{
    return insertTab(index, QIcon(), text);
}

int ZTabBar::insertTab(int index, const QIcon& icon, const QString& text)
{
    if (validIndex(index)) {
        index = m_tabList.count();
        m_tabList.append(Tab(icon, text));
    }
    else {
        m_tabList.insert(index, Tab(icon, text));
    }

    if (m_tabList.count() == 1)
    {
        setCurrentIndex(index);
    }
    else if (index <= m_currentIndex) {
        ++m_currentIndex;
    }

    if (m_bCloseBtn) {
        ZIconButton* closeButton = new ZIconButton(QIcon(":/icons/closebtn.svg"), ZenoStyle::dpiScaledSize(QSize(20, 20)), QColor(60, 60, 60), QColor(60, 60, 60));
        connect(closeButton, SIGNAL(clicked()), this, SLOT(_closeTab()));
    }

    return -1;
}

void ZTabBar::_closeTab()
{
    QObject* object = sender();
    int tabToClose = -1;
    for (int i = 0; i < m_tabList.count(); ++i)
    {
        if (m_tabList.at(i).pCloseButton == object)
        {
            tabToClose = i;
            break;
        }
    }
    if (tabToClose != -1)
        emit tabCloseBtnClicked(tabToClose);
}

void ZTabBar::moveTab(int srcIndex, int dstIndex)
{

}

void ZTabBar::removeTab(int index)
{
    if (validIndex(index))
    {
        if (m_tabList[index].pCloseButton)
        {
            m_tabList[index].pCloseButton->hide();
            m_tabList[index].pCloseButton->deleteLater();
            m_tabList[index].pCloseButton = nullptr;
        }
        int newIndex = -1;
        m_tabList.removeAt(index);
        if (index == m_currentIndex) {
            newIndex = m_tabList.count() - 1;
            if (newIndex >= 0)
            {
                setCurrentIndex(newIndex);
            }
        }
        //update issues.
    }
}

QString ZTabBar::tabText(int index) const
{
    return m_tabList.at(index).text;
}

void ZTabBar::setTabText(int index, const QString& text)
{
    if (validIndex(index))
    {
        m_tabList[index].text = text;
    }
}

void ZTabBar::setCurrentIndex(int index)
{

}

void ZTabBar::refresh()
{

}

void ZTabBar::paintEvent(QPaintEvent* e)
{

}
