#include "ztabdockwidget.h"
#include <zenoui/comctrl/zdocktabwidget.h>
#include "zenoapplication.h"
#include "../panel/zenodatapanel.h"
#include "panel/zenoproppanel.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include "viewport/viewportwidget.h"
#include "nodesview/zenographseditor.h"
#include <zenoui/comctrl/zlabel.h>
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "docktabcontent.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zicontoolbutton.h>
#include <zenoui/model/modelrole.h>


ZTabDockWidget::ZTabDockWidget(ZenoMainWindow* parent, Qt::WindowFlags flags)
    : _base(parent, flags)
    , m_tabWidget(new ZDockTabWidget)
    , m_plblName(nullptr)
    , m_pLineEdit(nullptr)
{
    setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    setWidget(m_tabWidget);

    connect(this, SIGNAL(maximizeTriggered()), parent, SLOT(onMaximumTriggered()));
    connect(this, SIGNAL(splitRequest(bool)), parent, SLOT(onSplitDock(bool)));

    setTitleBarWidget(new QWidget(this));
    connect(m_tabWidget, SIGNAL(addClicked()), this, SLOT(onAddTabClicked()));
    connect(m_tabWidget, SIGNAL(layoutBtnClicked()), this, SLOT(onDockOptionsClicked()));
}

ZTabDockWidget::~ZTabDockWidget()
{

}

void ZTabDockWidget::setCurrentWidget(PANEL_TYPE type)
{
    QWidget* wid = createTabWidget(type);
    if (wid)
    {
        int idx = m_tabWidget->addTab(wid, type2Title(type));
        m_tabWidget->setCurrentIndex(idx);
    }
}

QWidget* ZTabDockWidget::createTabWidget(PANEL_TYPE type)
{
    ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
    switch (type)
    {
        case PANEL_NODE_PARAMS:
        {
            DockContent_Parameter* wid = new DockContent_Parameter;
            return wid;
        }
        case PANEL_VIEW:
        {
            return new DisplayWidget;
        }
        case PANEL_EDITOR:
        {
            return new DockContent_Editor;
        }
        case PANEL_NODE_DATA:
        {
            return new ZenoSpreadsheet;
        }
        case PANEL_LOG:
        {
            return new ZlogPanel;
        }
    }
    return nullptr;
}

QString ZTabDockWidget::type2Title(PANEL_TYPE type)
{
    switch (type)
    {
    case PANEL_VIEW:        return tr("View");
    case PANEL_EDITOR:      return tr("Editor");
    case PANEL_NODE_PARAMS: return tr("Parameter");
    case PANEL_NODE_DATA:   return tr("Data");
    case PANEL_LOG:         return tr("Logger");
    default:
        return "";
    }
}

PANEL_TYPE ZTabDockWidget::title2Type(const QString& title)
{
    PANEL_TYPE type = PANEL_EMPTY;
    if (title == tr("Parameter")) {
        type = PANEL_NODE_PARAMS;
    }
    else if (title == tr("View")) {
        type = PANEL_VIEW;
    }
    else if (title == tr("Editor")) {
        type = PANEL_EDITOR;
    }
    else if (title == tr("Data")) {
        type = PANEL_NODE_DATA;
    }
    else if (title == tr("Logger")) {
        type = PANEL_LOG;
    }
    return type;
}

void ZTabDockWidget::onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (DockContent_Parameter* prop = qobject_cast<DockContent_Parameter*>(wid))
        {
            prop->onNodesSelected(subgIdx, nodes, select);
        }
    }
}

void ZTabDockWidget::onPrimitiveSelected(const std::unordered_set<std::string>& primids)
{

}

void ZTabDockWidget::onUpdateViewport(const QString& action)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (DisplayWidget* pView = qobject_cast<DisplayWidget*>(wid))
        {
            pView->updateFrame(action);
        }
    }
}

void ZTabDockWidget::onRunFinished()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (DisplayWidget* pView = qobject_cast<DisplayWidget*>(wid))
        {
            pView->onFinished();
        }
    }
}

void ZTabDockWidget::onRun()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
        if (DisplayWidget* pView = qobject_cast<DisplayWidget*>(m_tabWidget->widget(i)))
            pView->onRun();
}

void ZTabDockWidget::onRecord()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
        if (DisplayWidget *pView = qobject_cast<DisplayWidget *>(m_tabWidget->widget(i)))
            pView->onRecord();
}

void ZTabDockWidget::onKill()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
        if (DisplayWidget *pView = qobject_cast<DisplayWidget *>(m_tabWidget->widget(i)))
            pView->onKill();
}

void ZTabDockWidget::onPlayClicked(bool bChecked)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
        if (DisplayWidget *pView = qobject_cast<DisplayWidget *>(m_tabWidget->widget(i)))
            pView->onPlayClicked(bChecked);
}

void ZTabDockWidget::onSliderValueChanged(int frame)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
        if (DisplayWidget *pView = qobject_cast<DisplayWidget *>(m_tabWidget->widget(i)))
            pView->onSliderValueChanged(frame);
}

void ZTabDockWidget::onFinished()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
        if (DisplayWidget *pView = qobject_cast<DisplayWidget *>(m_tabWidget->widget(i)))
            pView->onFinished();
}

void ZTabDockWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(36, 36, 36));
    _base::paintEvent(event);
}

bool ZTabDockWidget::event(QEvent* event)
{
    return _base::event(event);
}

void ZTabDockWidget::onDockOptionsClicked()
{
    QMenu* menu = new QMenu(this);
    QFont font("HarmonyOS Sans", 12);
    font.setBold(false);
    menu->setFont(font);

    QAction* pSplitHor = new QAction("Split Left/Right");
    QAction* pSplitVer = new QAction("Split Top/Bottom");
    QAction* pMaximize = new QAction("Maximize");
    QAction* pFloatWin = new QAction("Float Window");
    QAction* pCloseLayout = new QAction("Close Layout");

    connect(pMaximize, SIGNAL(triggered()), this, SIGNAL(maximizeTriggered()));
    connect(pFloatWin, SIGNAL(triggered()), this, SLOT(onFloatTriggered()));
    connect(pCloseLayout, SIGNAL(triggered()), this, SLOT(close()));
    connect(pSplitHor, &QAction::triggered, this, [=]() {
        emit splitRequest(true);
        });
    connect(pSplitVer, &QAction::triggered, this, [=]() {
        emit splitRequest(false);
        });

    menu->addAction(pSplitHor);
    menu->addAction(pSplitVer);
    menu->addSeparator();
    menu->addAction(pMaximize);
    menu->addAction(pFloatWin);
    menu->addSeparator();
    menu->addAction(pCloseLayout);
    menu->exec(QCursor::pos());
}

void ZTabDockWidget::onMaximizeTriggered()
{

}

void ZTabDockWidget::onFloatTriggered()
{

}

void ZTabDockWidget::onAddTabClicked()
{
    QMenu* menu = new QMenu(this);
    QFont font("HarmonyOS Sans", 12);
    font.setBold(false);
    menu->setFont(font);

    static QList<QString> panels = { tr("Parameter"), tr("View"), tr("Editor"), tr("Data"), tr("Logger") };
    for (QString name : panels)
    {
        QAction* pAction = new QAction(name);
        connect(pAction, &QAction::triggered, this, [=]() {
            PANEL_TYPE type = title2Type(name);
            QWidget* wid = createTabWidget(type);
            if (wid)
            {
                int idx = m_tabWidget->addTab(wid, name);
                m_tabWidget->setCurrentIndex(idx);
            }
        });
        menu->addAction(pAction);
    }
    menu->exec(QCursor::pos());
}

void ZTabDockWidget::init(ZenoMainWindow* pMainWin)
{
    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(38, 38, 38));
    palette.setBrush(QPalette::WindowText, QColor());
    setPalette(palette);
    //...
}

bool ZTabDockWidget::isTopLevelWin()
{
    return false;
}