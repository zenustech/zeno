#include "zenodockwidget.h"
#include "zenodocktitlewidget.h"
#include "zenomainwindow.h"
#include <comctrl/ziconbutton.h>
#include <comctrl/ztoolbutton.h>
#include "nodesview/zenographseditor.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "../panel/zenoproppanel.h"
#include <zenoui/model/modelrole.h>
#include "util/log.h"
#include "panel/zenospreadsheet.h"
#include "viewport/viewportwidget.h"
#include "viewport/zenovis.h"
#include <zenovis/ObjectsManager.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zdocktabwidget.h>


////////////////////////////////////////////////////////////////////////////////////////////
ZenoDockWidget::ZenoDockWidget(const QString &title, QWidget *parent, Qt::WindowFlags flags)
    : _base(title, parent, flags)
    , m_type(DOCK_EMPTY)
{
    ZenoMainWindow* pMainWin = qobject_cast<ZenoMainWindow*>(parent);
    init(pMainWin);
}

ZenoDockWidget::ZenoDockWidget(QWidget *parent, Qt::WindowFlags flags)
    : _base(parent, flags)
    , m_type(DOCK_EMPTY)
{
    ZenoMainWindow* pMainWin = qobject_cast<ZenoMainWindow*>(parent);
    init(pMainWin);
}

ZenoDockWidget::~ZenoDockWidget()
{
}

void ZenoDockWidget::setWidget(DOCK_TYPE type, QWidget* widget)
{
    m_type = type;
    if (m_type == DOCK_NODE_PARAMS)
    {
        m_tabWidget = new ZDockTabWidget;
        m_tabWidget->addTab(widget, "Parameter");

        QWidget* pWidget1 = new QWidget;
        pWidget1->setAutoFillBackground(true);
        QPalette pal = pWidget1->palette();
        pal.setColor(QPalette::Window, QColor(44,51,58));
        pWidget1->setPalette(pal);

        QWidget* pWidget2 = new QWidget;
        pWidget2->setAutoFillBackground(true);
        pal = pWidget2->palette();
        pal.setColor(QPalette::Window, QColor(44, 51, 58));
        pWidget2->setPalette(pal);

        m_tabWidget->addTab(pWidget1, "Parameter");
        m_tabWidget->addTab(pWidget2, "Parameter");
        _base::setWidget(m_tabWidget);
        setTitleBarWidget(new QWidget(this));

        return;
    }

    _base::setWidget(widget);
	m_type = type;
    ZenoDockTitleWidget* pTitleWidget = nullptr;
	if (m_type == DOCK_EDITOR)
	{
        ZenoEditorDockTitleWidget* pEditorTitle = new ZenoEditorDockTitleWidget;
        pEditorTitle->setupUi();
        pEditorTitle->initModel();
        setTitleBarWidget(pEditorTitle);
        pTitleWidget = pEditorTitle;
        ZenoGraphsEditor* pEditor = qobject_cast<ZenoGraphsEditor*>(widget);
        ZASSERT_EXIT(pEditor);
        connect(pEditorTitle, SIGNAL(actionTriggered(QAction*)), pEditor, SLOT(onMenuActionTriggered(QAction*)));
	}
    else if (m_type == DOCK_NODE_PARAMS)
    {
        ZenoPropDockTitleWidget* pPropTitle = new ZenoPropDockTitleWidget;
        pPropTitle->setupUi();
        pTitleWidget = pPropTitle;
    }
    else if (m_type == DOCK_VIEW)
    {
        ZenoViewDockTitle* pViewTitle = new ZenoViewDockTitle;
        pViewTitle->setupUi();
        pTitleWidget = pViewTitle;
        if (DisplayWidget* pViewWid = qobject_cast<DisplayWidget*>(widget))
        {
            connect(pViewTitle, &ZenoDockTitleWidget::actionTriggered, this, [=](QAction* action) {
                //TODO: unify the tr specification.
                if (action->text() == ZenoViewDockTitle::tr("Record Video")) {
                    pViewWid->onRecord();
                }
            });
        }
    }
	else
	{
        pTitleWidget = new ZenoDockTitleWidget;
        pTitleWidget->setupUi();
	}
    setTitleBarWidget(pTitleWidget);

	connect(pTitleWidget, SIGNAL(dockOptionsClicked()), this, SLOT(onDockOptionsClicked()));
	connect(pTitleWidget, SIGNAL(dockSwitchClicked(DOCK_TYPE)), this, SIGNAL(dockSwitchClicked(DOCK_TYPE)));
    connect(pTitleWidget, SIGNAL(doubleClicked()), this, SLOT(onFloatTriggered()));
}

void ZenoDockWidget::onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
    if (m_type == DOCK_NODE_PARAMS) {
        ZenoPropPanel* panel = nullptr;
        if (QTabWidget* pTabWidget = qobject_cast<QTabWidget*>(widget()))
        {
            for (int i = 0; i < pTabWidget->count(); i++)
            {
                panel = qobject_cast<ZenoPropPanel*>(pTabWidget->widget(i));
                if (panel)
                    break;
            }
        }
        else if (panel = qobject_cast<ZenoPropPanel*>(widget()))
        {
        }

        ZASSERT_EXIT(panel);

        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        ZenoPropDockTitleWidget* pPropTitle = qobject_cast<ZenoPropDockTitleWidget*>(titleBarWidget());
        if (pPropTitle)
        {
            if (select) {
                const QModelIndex& idx = nodes[0];
                QString nodeName = pModel->data2(subgIdx, idx, ROLE_OBJNAME).toString();
                pPropTitle->setTitle(nodeName);
            }
            else {
                pPropTitle->setTitle(tr("property"));
            }
        }
        panel->reset(pModel, subgIdx, nodes, select);
    }
    else if (m_type == DOCK_NODE_DATA) {
        ZenoSpreadsheet* panel = qobject_cast<ZenoSpreadsheet*>(widget());
        ZASSERT_EXIT(panel);

        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        if (select) {
            const QModelIndex& idx = nodes[0];
            QString nodeId = pModel->data2(subgIdx, idx, ROLE_OBJID).toString();
            auto *scene = Zenovis::GetInstance().getSession()->get_scene();
            scene->selected.clear();
            std::string nodeid = nodeId.toStdString();
            for (auto const &[key, ptr]: scene->objectsMan->pairs()) {
                if (nodeid == key.substr(0, key.find_first_of(':'))) {
                    scene->selected.insert(key);
                }
            }
            ZenoMainWindow* mainWin = zenoApp->getMainWindow();
            mainWin->onPrimitiveSelected(scene->selected);
            zenoApp->getMainWindow()->updateViewport();
        }
        else {
            panel->clear();
        }
    }
}

DOCK_TYPE ZenoDockWidget::type() const
{
    return m_type;
}

void ZenoDockWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(36, 36, 36));
    _base::paintEvent(event);
}

bool ZenoDockWidget::event(QEvent* event)
{
    switch (event->type())
    {
        case QEvent::MouseButtonDblClick:
        {
        //    onFloatTriggered();
            if (m_type == DOCK_EDITOR || m_type == DOCK_VIEW)
                return true;
        }
        case QEvent::NonClientAreaMouseButtonDblClick:
        {
            // for the case of dblclicking the titlebar of float top-level window.
            if (isTopLevelWin())
                return true;
        }
    }
    return _base::event(event);
}

bool ZenoDockWidget::isTopLevelWin()
{
    Qt::WindowFlags flags = windowFlags();
    return flags & m_newFlags;
}

void ZenoDockWidget::init(ZenoMainWindow* pMainWin)
{
    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(38, 38, 38));
    palette.setBrush(QPalette::WindowText, QColor());
    setPalette(palette);
    if (!windowTitle().isEmpty())
    {
        ZenoDockTitleWidget* pTitleWidget = new ZenoDockTitleWidget;
        pTitleWidget->setupUi();
        setTitleBarWidget(pTitleWidget);
        connect(pTitleWidget, SIGNAL(dockOptionsClicked()), this, SLOT(onDockOptionsClicked()));
        connect(pTitleWidget, SIGNAL(dockSwitchClicked(DOCK_TYPE)), this, SIGNAL(dockSwitchClicked(DOCK_TYPE)));
    }
    else {
        //setTitleBarWidget(new QWidget(this));
    }
    connect(this, SIGNAL(dockSwitchClicked(DOCK_TYPE)), pMainWin, SLOT(onDockSwitched(DOCK_TYPE)));
}

void ZenoDockWidget::onDockOptionsClicked()
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

void ZenoDockWidget::onMaximizeTriggered()
{
    QMainWindow* pMainWin = qobject_cast<QMainWindow*>(parent());
    for (auto pObj : pMainWin->children())
    {
        if (ZenoDockWidget* pOtherDock = qobject_cast<ZenoDockWidget*>(pObj))
        {
            if (pOtherDock != this)
            {
                pOtherDock->close();
            }
        }
    }
}

void ZenoDockWidget::onFloatTriggered()
{
    if (isFloating())
    {
        if (m_type == DOCK_EDITOR || m_type == DOCK_VIEW)
        {
            setWindowFlags(m_oldFlags);
            ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
            //need redock
            pMainWin->restoreDockWidget(this);
            bool bVisible = isVisible();
            if (!bVisible)
            {
                setVisible(true);
            }
        }
        setFloating(false);
    }
    else
    {
        setFloating(true);
        m_oldFlags = windowFlags();
        if (m_type == DOCK_EDITOR)
        {
            setParent(nullptr);
            m_newFlags = Qt::CustomizeWindowHint | Qt::Window |
                         Qt::WindowMinimizeButtonHint |
                         Qt::WindowMaximizeButtonHint |
                         Qt::WindowCloseButtonHint;
            setWindowFlags(m_newFlags);
            show();
        }
        else if (m_type == DOCK_VIEW)
        {
            //reinitialize glview is not allowed.
            setParent(nullptr);
            m_newFlags = Qt::CustomizeWindowHint | Qt::Window | Qt::WindowMinimizeButtonHint |
                         Qt::WindowMaximizeButtonHint;
            setWindowFlags(m_newFlags);
            show();
        }
    }
}


void ZenoDockWidget::onPrimitiveSelected(const std::unordered_set <std::string> &primids) {
    if (m_type != DOCK_NODE_DATA) {
        return;
    }
    ZenoSpreadsheet* panel = qobject_cast<ZenoSpreadsheet*>(widget());
    ZASSERT_EXIT(panel);
    if (primids.size() == 1) {
        panel->setPrim(*primids.begin());
    }
    else {
        panel->clear();
    }
}
