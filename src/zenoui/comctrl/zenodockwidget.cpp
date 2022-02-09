#include "zenodockwidget.h"
#include <QtWidgets/private/qdockwidget_p.h>
#include <comctrl/ziconbutton.h>
#include <comctrl/ztoolbutton.h>
#include <style/zenostyle.h>


ZenoDockTitleWidget::ZenoDockTitleWidget(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setSpacing(0);
    pLayout->setContentsMargins(0, 0, 0, 0);

    QHBoxLayout* pHLayout = new QHBoxLayout;
    ZToolButton* pDockBtn = new ZToolButton(ZToolButton::Opt_HasIcon, QIcon(":/icons/dockOption.svg"), QSize(16, 16));
    pDockBtn->setMargins(QMargins(10, 10, 10, 10));
    pDockBtn->setBackgroundClr(QColor(51, 51, 51), QColor(51, 51, 51), QColor(51, 51, 51));
    pHLayout->addStretch();
    pHLayout->addWidget(pDockBtn);
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

    connect(pDockBtn, SIGNAL(clicked()), this, SIGNAL(dockOptionsClicked()));
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


ZenoDockWidget::ZenoDockWidget(const QString &title, QWidget *parent, Qt::WindowFlags flags)
    : _base(title, parent, flags)
{
    init();
}

ZenoDockWidget::ZenoDockWidget(QWidget *parent, Qt::WindowFlags flags)
    : _base(parent, flags)
{
    init();
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

void ZenoDockWidget::init()
{
    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(38, 38, 38));
    palette.setBrush(QPalette::WindowText, QColor());
    setPalette(palette);
    ZenoDockTitleWidget* pTitleWidget = new ZenoDockTitleWidget;
    setTitleBarWidget(pTitleWidget);
    connect(pTitleWidget, SIGNAL(dockOptionsClicked()), this, SLOT(onDockOptionsClicked()));
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