#include "zenodockwidget.h"
#include <QtWidgets/private/qdockwidget_p.h>
#include <comctrl/ziconbutton.h>
#include <comctrl/ztoolbutton.h>
//#include <style/zenostyle.h>


ZenoDockTitleWidget::ZenoDockTitleWidget(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setSpacing(0);
    pLayout->setContentsMargins(0, 0, 0, 0);

    QHBoxLayout* pHLayout = new QHBoxLayout;

    ZToolButton* pDockSwitchBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/dockOption.svg"), QSize(16, 16));
    pDockSwitchBtn->setMargins(QMargins(10, 10, 10, 10));
    pDockSwitchBtn->setBackgroundClr(QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

    ZToolButton* pDockOptionsBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/dockOption.svg"), QSize(16, 16));
    pDockOptionsBtn->setMargins(QMargins(10, 10, 10, 10));
    pDockOptionsBtn->setBackgroundClr(QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));

    pHLayout->addWidget(pDockSwitchBtn);
    pHLayout->addStretch();
    pHLayout->addWidget(pDockOptionsBtn);
    pHLayout->setContentsMargins(0, 0, 0, 0);
    pHLayout->setMargin(0);

    QFrame* pLine = new QFrame;
    pLine->setFrameShape(QFrame::HLine);
    pLine->setFrameShadow(QFrame::Plain);
    QPalette pal = pLine->palette();
    pal.setBrush(QPalette::WindowText, QColor(36, 36, 36));
    pLine->setPalette(pal);
    pLine->setFixedHeight(1);       //dpi scaled?
    pLine->setLineWidth(1);

    pLayout->addLayout(pHLayout);
    pLayout->addWidget(pLine);

    setLayout(pLayout);

    connect(pDockOptionsBtn, SIGNAL(clicked()), this, SIGNAL(dockOptionsClicked()));
    connect(pDockSwitchBtn, SIGNAL(clicked()), this, SLOT(onDockSwitchClicked()));
}

ZenoDockTitleWidget::~ZenoDockTitleWidget()
{
}

QSize ZenoDockTitleWidget::sizeHint() const
{
    QSize sz = QWidget::sizeHint();
    return sz;
}

void ZenoDockTitleWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(58, 58, 58));
    QPen pen(QColor(44, 50, 49), 2);
    painter.setPen(pen);
}

void ZenoDockTitleWidget::onDockSwitchClicked()
{
	QMenu* menu = new QMenu(this);
	QFont font("HarmonyOS Sans SC", 12);
	font.setBold(false);
	menu->setFont(font);
	QAction* pSwitchEditor = new QAction("Editor");
	QAction* pSwitchView = new QAction("View");
    QAction* pSwitchNodeParam = new QAction("parameter");
    QAction* pSwitchNodeData = new QAction("data");
	menu->addAction(pSwitchEditor);
	menu->addAction(pSwitchView);
    menu->addAction(pSwitchNodeParam);
    menu->addAction(pSwitchNodeData);
    connect(pSwitchEditor, &QAction::triggered, this, [=]() {
        emit dockSwitchClicked(DOCK_EDITOR);
    });
    connect(pSwitchView, &QAction::triggered, this, [=]() {
        emit dockSwitchClicked(DOCK_VIEW);
    });
	connect(pSwitchNodeParam, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_NODE_PARAMS);
	});
	connect(pSwitchNodeData, &QAction::triggered, this, [=]() {
		emit dockSwitchClicked(DOCK_NODE_DATA);
	});

    menu->exec(QCursor::pos());
}


ZenoDockWidget::ZenoDockWidget(const QString &title, QWidget *parent, Qt::WindowFlags flags)
    : _base(title, parent, flags)
{
    ZenoMainWindow* pMainWin = qobject_cast<ZenoMainWindow*>(parent);
    init(pMainWin);
}

ZenoDockWidget::ZenoDockWidget(QWidget *parent, Qt::WindowFlags flags)
    : _base(parent, flags)
{
    ZenoMainWindow* pMainWin = qobject_cast<ZenoMainWindow*>(parent);
    init(pMainWin);
}

ZenoDockWidget::~ZenoDockWidget()
{
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
    setTitleBarWidget(pTitleWidget);
    connect(pTitleWidget, SIGNAL(dockOptionsClicked()), this, SLOT(onDockOptionsClicked()));
    connect(pTitleWidget, SIGNAL(dockSwitchClicked(DOCK_TYPE)), this, SIGNAL(dockSwitchClicked(DOCK_TYPE)));
    connect(this, SIGNAL(dockSwitchClicked(DOCK_TYPE)), pMainWin, SLOT(onDockSwitched(DOCK_TYPE)));
}

void ZenoDockWidget::onDockOptionsClicked()
{
    QMenu* menu = new QMenu(this);
    QFont font("HarmonyOS Sans SC", 12);
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