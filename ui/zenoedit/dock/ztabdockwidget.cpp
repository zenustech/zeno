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
            QWidget* wid = new QWidget;

            QHBoxLayout* pToolLayout = new QHBoxLayout;
            pToolLayout->setContentsMargins(ZenoStyle::dpiScaled(8), ZenoStyle::dpiScaled(4),
                ZenoStyle::dpiScaled(4), ZenoStyle::dpiScaled(4));

            ZIconLabel* pIcon = new ZIconLabel();
            pIcon->setIcons(ZenoStyle::dpiScaledSize(QSize(20, 20)), ":/icons/nodeclr-yellow.svg", "");

            m_plblName = new QLabel("");
            m_plblName->setFont(QFont("Segoe UI Bold", 10));
            m_plblName->setMinimumWidth(ZenoStyle::dpiScaled(128));
            QPalette palette = m_plblName->palette();
            palette.setColor(m_plblName->foregroundRole(), QColor("#A3B1C0"));
            m_plblName->setPalette(palette);

            m_pLineEdit = new QLineEdit;
            m_pLineEdit->setText("");
            m_pLineEdit->setProperty("cssClass", "zeno2_2_lineedit");
            m_pLineEdit->setReadOnly(true);

            ZIconToolButton* pFixBtn = new ZIconToolButton(":/icons/fixpanel.svg", ":/icons/fixpanel-on.svg");
            ZIconToolButton* pWikiBtn = new ZIconToolButton(":/icons/wiki.svg", ":/icons/wiki-on.svg");
            ZIconToolButton* pSettingBtn = new ZIconToolButton(":/icons/settings.svg", ":/icons/settings-on.svg");

            pToolLayout->addWidget(pIcon);
            pToolLayout->addWidget(m_plblName);
            pToolLayout->addWidget(m_pLineEdit);
            pToolLayout->addStretch();
            pToolLayout->addWidget(pFixBtn);
            pToolLayout->addWidget(pWikiBtn);
            pToolLayout->addWidget(pSettingBtn);
            pToolLayout->setSpacing(9);

            QVBoxLayout* pVLayout = new QVBoxLayout;
            pVLayout->addLayout(pToolLayout);
            pVLayout->setContentsMargins(0, 0, 0, 0);
            pVLayout->setSpacing(0);

            ZenoPropPanel* prop = new ZenoPropPanel;
            pVLayout->addWidget(prop);
            wid->setLayout(pVLayout);
            return wid;
        }
        case PANEL_VIEW:
        {
            return new DisplayWidget;
        }
        case PANEL_EDITOR:
        {
            return new ZenoGraphsEditor(pMainWin);
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
        if (ZenoPropPanel* prop = wid->findChild<ZenoPropPanel*>())
        {
            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
            prop->reset(pModel, subgIdx, nodes, select);

            if (!nodes.isEmpty())
            {
                const QModelIndex& idx = nodes[0];
                if (select) {
                    m_plblName->setText(idx.data(ROLE_OBJNAME).toString());
                    m_pLineEdit->setText(idx.data(ROLE_OBJID).toString());
                }
                else{
                    m_plblName->setText("");
                    m_pLineEdit->setText("");
                }
            }
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