#include "zenodockwidget.h"
#include "zenodocktitlewidget.h"
#include "zenomainwindow.h"
#include <comctrl/ziconbutton.h>
#include <comctrl/ztoolbutton.h>
#include "nodesview/zenographseditor.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"


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
	}
	else
	{
        pTitleWidget = new ZenoDockTitleWidget;
        pTitleWidget->setupUi();
        setTitleBarWidget(pTitleWidget);
	}
	connect(pTitleWidget, SIGNAL(dockOptionsClicked()), this, SLOT(onDockOptionsClicked()));
	connect(pTitleWidget, SIGNAL(dockSwitchClicked(DOCK_TYPE)), this, SIGNAL(dockSwitchClicked(DOCK_TYPE)));
}

void ZenoDockWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(36, 36, 36));
    _base::paintEvent(event);
}

void ZenoDockWidget::init(ZenoMainWindow* pMainWin)
{
    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(38, 38, 38));
    palette.setBrush(QPalette::WindowText, QColor());
    setPalette(palette);
    ZenoDockTitleWidget* pTitleWidget = new ZenoDockTitleWidget;
    pTitleWidget->setupUi();
    setTitleBarWidget(pTitleWidget);
	connect(pTitleWidget, SIGNAL(dockOptionsClicked()), this, SLOT(onDockOptionsClicked()));
	connect(pTitleWidget, SIGNAL(dockSwitchClicked(DOCK_TYPE)), this, SIGNAL(dockSwitchClicked(DOCK_TYPE)));
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
        setFloating(false);
    }
    else
    {
        setFloating(true);
        //setParent(nullptr);
        //setWindowFlags(Qt::CustomizeWindowHint |
        //    Qt::Window |
        //    Qt::WindowMinimizeButtonHint |
        //    Qt::WindowMaximizeButtonHint |
        //    Qt::WindowCloseButtonHint);
        //show();
    }
}