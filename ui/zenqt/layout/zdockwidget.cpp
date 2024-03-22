#include "zdockwidget.h"
#include "layout/zdockwidget.h"
#include "zenoapplication.h"
#include "../panel/zenodatapanel.h"
#include "panel/zenoimagepanel.h"
#include "panel/zenoproppanel.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include "../panel/zenocommandparamspanel.h"
#include "../panel/zenoopenpathpanel.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "nodeeditor/gv/zenographseditor.h"
#include "widgets/zlabel.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "style/zenostyle.h"
#include "widgets/zicontoolbutton.h"
#include "uicommon.h"
#include <zenovis/ObjectsManager.h>
#include "util/uihelper.h"
#include "util/apphelper.h"
#include "viewport/optixviewport.h"
#include "layout/zdocktabwidget.h"
#include "zassert.h"
#include "viewport/zenovis.h"


ZDockWidget::ZDockWidget(ZenoMainWindow* mainWin, Qt::WindowFlags flags)
    : _base(mainWin, flags)
    , m_tabWidget(new ZDockTabWidget)
{
    setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);

    /*there is no way to control the docklayout of qt, so
      we have to construct a widget to paint the border.
     */
    QWidget* pProxyWid = new QWidget;
    QPalette pal = pProxyWid->palette();
    pal.setColor(QPalette::Window, QColor("#000000"));
    pProxyWid->setAutoFillBackground(true);
    pProxyWid->setPalette(pal);

    QVBoxLayout *pLayout = new QVBoxLayout;
    pLayout->setContentsMargins(1, 1, 1, 1);
    pLayout->addWidget(m_tabWidget);
    pProxyWid->setLayout(pLayout);

    setWidget(pProxyWid);

    connect(this, SIGNAL(maximizeTriggered()), mainWin, SLOT(onMaximumTriggered()));
    connect(this, SIGNAL(splitRequest(bool)), mainWin, SLOT(onSplitDock(bool)));
    connect(this, SIGNAL(closeRequest()), mainWin, SLOT(onCloseDock()));

    setTitleBarWidget(new QWidget(this));
    connect(m_tabWidget, SIGNAL(addClicked()), this, SLOT(onAddTabClicked()));
    connect(m_tabWidget, SIGNAL(layoutBtnClicked()), this, SLOT(onDockOptionsClicked()));
    //connect(this, SIGNAL(dockLocationChanged(Qt::DockWidgetArea)),
    //    mainWin, SLOT(onDockLocationChanged(Qt::DockWidgetArea)));
    connect(m_tabWidget, &ZDockTabWidget::tabAboutToClose, this, [=](int i) {
        DockToolbarWidget* pToolWid = qobject_cast<DockToolbarWidget*>(m_tabWidget->widget(i));
        if (pToolWid)
            pToolWid->onTabAboutToClose();
    });

    connect(m_tabWidget, &ZDockTabWidget::tabClosed, this, [=]() {
        if (m_tabWidget->count() == 0) {
            this->close();
        }
    });
    connect(m_tabWidget, &ZDockTabWidget::currentChanged, this, [=](int index) {
        if (DockContent_View* view = qobject_cast<DockContent_View*>(m_tabWidget->widget(index)))
        {
            if (DisplayWidget* dpview = view->getDisplayWid())
            {
                ZenoMainWindow* main = zenoApp->getMainWindow();
                ZASSERT_EXIT(main);
                QVector<DisplayWidget*> views = main->viewports();
                for (auto view : views)
                {
                    view->setIsCurrent(false);
                }
                dpview->setIsCurrent(true);
            }
        }
        });
}

ZDockWidget::~ZDockWidget()
{

}

void ZDockWidget::setCurrentWidget(PANEL_TYPE type)
{
    m_debugPanel = type;
    QWidget* wid = createTabWidget(type);
    if (wid)
    {
        int idx = m_tabWidget->addTab(wid, type2Title(type));
        m_tabWidget->setCurrentIndex(idx);
    }
}

int ZDockWidget::count() const
{
    return m_tabWidget ? m_tabWidget->count() : 0;
}

QWidget* ZDockWidget::widget(int i) const
{
    if (i < 0 || i >= count())
        return nullptr;

    return m_tabWidget->widget(i);
}

QWidget* ZDockWidget::widget() const
{
    return m_tabWidget;
}

void ZDockWidget::testCleanupGL()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(0);
        if (DockContent_View* pDis = qobject_cast<DockContent_View*>(wid)) {
            DisplayWidget* pWid = pDis->getDisplayWid();
            if (pWid)
                pWid->testCleanUp();
        }
    }
}

QVector<DisplayWidget*> ZDockWidget::viewports() const
{
    QVector<DisplayWidget*> views;
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (DockContent_View* pView = qobject_cast<DockContent_View*>(wid))
        {
            views.append(pView->getDisplayWid());
        }
    }
    return views;
}

DisplayWidget* ZDockWidget::getUniqueViewport() const
{
    if (1 == count())
    {
        QWidget* wid = m_tabWidget->widget(0);
        if (DockContent_View* pView = qobject_cast<DockContent_View*>(wid))
        {
            return pView->getDisplayWid();
        }
    }
    return nullptr;
}

ZenoGraphsEditor* ZDockWidget::getAnyEditor() const
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(0);
        if (DockContent_Editor* pEditor = qobject_cast<DockContent_Editor*>(wid))
        {
            return pEditor->getEditor();
        }
    }
    return nullptr;
}

QWidget* ZDockWidget::createTabWidget(PANEL_TYPE type)
{
    ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
    switch (type)
    {
        case PANEL_NODE_PARAMS:
        {
            DockContent_Parameter* wid = new DockContent_Parameter;
            wid->initUI();
            return wid;
        }
        case PANEL_GL_VIEW:
        {
            DockContent_View* wid = new DockContent_View(true);
            wid->initUI();

            DisplayWidget* pDisplay = wid->getDisplayWid();
            ZASSERT_EXIT(pDisplay, nullptr);
            Zenovis* pZenoVis = pDisplay->getZenoVis();
            ZASSERT_EXIT(pZenoVis, nullptr);
            pZenoVis->initializeGL();
            return wid;
        }
        case PANEL_EDITOR:
        {
            DockContent_Editor* wid = new DockContent_Editor;
            wid->initUI();
            return wid;
        }
        case PANEL_NODE_DATA:
        {
            return new ZenoSpreadsheet;
        }
        case PANEL_LIGHT:
        {
            return new ZenoLights;
        }
        case PANEL_LOG:
        {
            DockContent_Log* wid = new DockContent_Log;
            wid->initUI();
            return wid;
        }
        case PANEL_IMAGE:
        {
            return new ZenoImagePanel;
        }
        case PANEL_OPTIX_VIEW:
        {
            DockContent_View* wid = new DockContent_View(false);
            wid->initUI();
            return wid;
        }
        case PANEL_COMMAND_PARAMS:
        {
            //TODO
            //return new ZenoCommandParamsPanel;
            break;
        }
        case PANEL_OPEN_PATH:
        {
            return new ZenoOpenPathPanel;
        }
    }
    return nullptr;
}

QString ZDockWidget::type2Title(PANEL_TYPE type)
{
    switch (type)
    {
    case PANEL_GL_VIEW:        return tr("Scene Viewport");
    case PANEL_EDITOR:      return tr("Node Editor");
    case PANEL_NODE_PARAMS: return tr("Parameter");
    case PANEL_NODE_DATA:   return tr("Spreadsheet");
    case PANEL_LOG:         return tr("Log");
    case PANEL_LIGHT:       return tr("Light");
    case PANEL_OPTIX_VIEW:  return tr("Optix");
    case PANEL_IMAGE: return tr("Image");
    case PANEL_COMMAND_PARAMS: return tr("Command Params");
    case PANEL_OPEN_PATH: return tr("Open Path");
    default:
        return "";
    }
}

PANEL_TYPE ZDockWidget::title2Type(const QString& title)
{
    PANEL_TYPE type = PANEL_EMPTY;
    if (title == tr("Parameter") || title == "Parameter") {
        type = PANEL_NODE_PARAMS;
    }
    else if (title == tr("View") || title == "View" || title == tr("Scene Viewport") || title == "Scene Viewport") {
        type = PANEL_GL_VIEW;
    }
    else if (title == tr("Editor") || title == "Editor" || title == tr("Node Editor") || title == "Node Editor") {
        type = PANEL_EDITOR;
    }
    else if (title == tr("Data") || title == "Data" || title == tr("Spreadsheet") || title == "Spreadsheet") {
        type = PANEL_NODE_DATA;
    }
    else if (title == tr("Logger") || title == "Logger" || title == tr("Log") || title == "Log") {
        type = PANEL_LOG;
    }
    else if (title == tr("Light")|| title == "Light") {
        type = PANEL_LIGHT;
    }
    else if (title == tr("Image")|| title == "Image") {
        type = PANEL_IMAGE;
    }
    else if (title == tr("Optix")|| title == "Optix") {
        type = PANEL_OPTIX_VIEW;
    }
    else if (title == tr("Command Params") || title == "Command Params") {
        type = PANEL_COMMAND_PARAMS;
    }
    else if (title == tr("Open Path") || title == "Open Path") {
        type = PANEL_OPEN_PATH;
    }
    return type;
}

void ZDockWidget::onNodesSelected(GraphModel* subgraph, const QModelIndexList& nodes, bool select)
{
    //if (nodes.count() <= 0)
    //    return;
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (DockContent_Parameter* prop = qobject_cast<DockContent_Parameter*>(wid))
        {
            if (select && nodes.size() <= 1) {
                prop->onNodesSelected(subgraph, nodes, select);
            }
        }
        else if (ZenoSpreadsheet* panel = qobject_cast<ZenoSpreadsheet*>(wid))
        {
            if (select && nodes.size() == 1)
            {
                const QModelIndex &idx = nodes[0];
                QString nodeId = idx.data(ROLE_NODE_NAME).toString();

                ZenoMainWindow *pWin = zenoApp->getMainWindow();
                ZASSERT_EXIT(pWin);
                QVector<DisplayWidget *> views = pWin->viewports();
                for (auto pDisplay : views)
                {
                    auto pZenoVis = pDisplay->getZenoVis();
                    ZASSERT_EXIT(pZenoVis);
                    auto *scene = pZenoVis->getSession()->get_scene();
                    scene->selected.clear();
                    std::string nodeid = nodeId.toStdString();
                    for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
                        if (nodeid == key.substr(0, key.find_first_of(':'))) {
                            scene->selected.insert(key);
                        }
                    }
                    onPrimitiveSelected(scene->selected);
                    pDisplay->updateFrame();
                }
            }
        } 
        else if (DockContent_Editor *editor = qobject_cast<DockContent_Editor *>(wid)) {
            if (select && nodes.size() <= 1)
            {
                editor->getEditor()->showFloatPanel(subgraph, nodes);
            }
        }
        else if (DockContent_View *view = qobject_cast<DockContent_View *>(wid)) {
            view->getDisplayWid()->onNodeSelected(subgraph, nodes, select);
        }
        else if (ZenoImagePanel *image = qobject_cast<ZenoImagePanel *>(wid))
        {
            if (select && nodes.size() == 1)
            {
                const QModelIndex &idx = nodes[0];
                image->setPrim(idx.data(ROLE_NODE_NAME).toString().toStdString());
            }
            if (!select)
            {
                image->clear();
            }
        }
    }
}

void ZDockWidget::onPrimitiveSelected(const std::unordered_set<std::string>& primids)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (ZenoSpreadsheet* panel = qobject_cast<ZenoSpreadsheet*>(wid))
        {
            if (primids.size() == 1) {
                panel->setPrim(*primids.begin());
            }
            else {
                panel->clear();
            }
        }
    }
}

void ZDockWidget::updateLights()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (ZenoLights* panel = qobject_cast<ZenoLights*>(wid))
        {
            panel->updateLights();
        }
    }
}

void ZDockWidget::onUpdateViewport(const QString& action)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        if (DockContent_View* pView = qobject_cast<DockContent_View*>(m_tabWidget->widget(i)))
        {
            DisplayWidget* pWid = pView->getDisplayWid();
            ZASSERT_EXIT(pWid);
            pWid->updateFrame(action);
        }
    }
}

void ZDockWidget::onPlayClicked(bool bChecked)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        if (DockContent_View* pView = qobject_cast<DockContent_View*>(m_tabWidget->widget(i)))
        {
            DisplayWidget* pWid = pView->getDisplayWid();
            ZASSERT_EXIT(pWid);
            pWid->onPlayClicked(bChecked);
        }
    }
}

void ZDockWidget::onSliderValueChanged(int frame)
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        if (DockContent_View* pView = qobject_cast<DockContent_View*>(m_tabWidget->widget(i)))
        {
            DisplayWidget* pWid = pView->getDisplayWid();
            ZASSERT_EXIT(pWid);
            pWid->onSliderValueChanged(frame);
        }
    }
}

void ZDockWidget::onRunFinished()
{
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        if (DockContent_View* pView = qobject_cast<DockContent_View*>(m_tabWidget->widget(i)))
        {
            DisplayWidget* pWid = pView->getDisplayWid();
            ZASSERT_EXIT(pWid);
            pWid->onRunFinished();
        }
        else if (DockContent_Editor* pView = qobject_cast<DockContent_Editor*>(m_tabWidget->widget(i)))
        {
            pView->runFinished();
        }
    }
}

void ZDockWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(36, 36, 36));
    _base::paintEvent(event);
}

bool ZDockWidget::event(QEvent* event)
{
    return _base::event(event);
}

void ZDockWidget::onDockOptionsClicked()
{
    QMenu* menu = new QMenu(this);
    QFont font = QApplication::font();
    font.setBold(false);
    menu->setFont(font);

    QAction* pSplitHor = new QAction(tr("Split Left/Right"));
    QAction* pSplitVer = new QAction(tr("Split Top/Bottom"));
    QAction* pMaximize = new QAction(tr("Maximize"));
    QAction* pFloatWin = new QAction(tr("Float Window"));
    QAction* pCloseLayout = new QAction(tr("Close Layout"));

    connect(pMaximize, SIGNAL(triggered()), this, SIGNAL(maximizeTriggered()));
    connect(pFloatWin, SIGNAL(triggered()), this, SLOT(onFloatTriggered()));
    connect(pCloseLayout, &QAction::triggered, this, [=](){
        emit closeRequest();
        });
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

void ZDockWidget::onMaximizeTriggered()
{

}

void ZDockWidget::onFloatTriggered()
{
    if (isFloating()) 
    {
        setWindowFlags(m_oldFlags);
        ZenoMainWindow *pMainWin = zenoApp->getMainWindow();
        //need redock
        pMainWin->restoreDockWidget(this);
        bool bVisible = isVisible();
        if (!bVisible) {
            setVisible(true);
        }
        setFloating(false);
    } 
    else 
    {
        setFloating(true);
        m_oldFlags = windowFlags();

        setParent(nullptr);
        m_newFlags = Qt::CustomizeWindowHint | Qt::Window | Qt::WindowMinimizeButtonHint |
                        Qt::WindowMaximizeButtonHint | Qt::WindowCloseButtonHint;

        QString filePath = zenoApp->graphsManager()->zsgPath();
        QString winTitle = AppHelper::nativeWindowTitle(filePath);
        auto mainWin = zenoApp->getMainWindow();
        if (mainWin)
            mainWin->updateNativeWinTitle(winTitle);
        setWindowIcon(QIcon(":/icons/zeno-logo.png"));

        setWindowFlags(m_newFlags);
        show();
    }
}

void ZDockWidget::onAddTabClicked()
{
    QMenu* menu = new QMenu(this);
    QFont font = QApplication::font();
    font.setBold(false);
    menu->setFont(font);

    static QList<QString> panels = { tr("Parameter"), tr("Scene Viewport"), tr("Node Editor"), tr("Spreadsheet"), tr("Log"), tr("Image"), tr("Optix"), tr("Command Params"), tr("Open Path") };
    for (QString name : panels)
    {
        QAction* pAction = new QAction(name);
        connect(pAction, &QAction::triggered, this, [=]() {
            PANEL_TYPE type = title2Type(name);
#ifdef ZENO_OPTIX_PROC
            if (type == PANEL_OPTIX_VIEW)
            {
                ZTcpServer* pServer = zenoApp->getServer();
                pServer->startOptixProc();
                return;
            }
#endif
            QWidget* wid = createTabWidget(type);
            if (wid)
            {
                int idx = m_tabWidget->addTab(wid, name);
                switch (panels.indexOf(name)) {
                case 0: m_debugPanel = PANEL_NODE_PARAMS; break;
                case 1: m_debugPanel = PANEL_GL_VIEW; break;
                case 2: m_debugPanel = PANEL_EDITOR; break;
                case 3: m_debugPanel = PANEL_NODE_DATA; break;
                case 4: m_debugPanel = PANEL_LOG; break;
                case 5: m_debugPanel = PANEL_LIGHT; break;
                case 6: m_debugPanel = PANEL_IMAGE; break;
                case 7: m_debugPanel = PANEL_OPTIX_VIEW; break;
                case 8: m_debugPanel = PANEL_COMMAND_PARAMS; break;
                case 9: m_debugPanel = PANEL_OPEN_PATH; break;
                }
                m_tabWidget->setCurrentIndex(idx);
            }
        });
        menu->addAction(pAction);
    }
    menu->exec(QCursor::pos());
}

void ZDockWidget::onAddTab(PANEL_TYPE type)
{
    QWidget *wid = createTabWidget(type);
    if (wid) {
        QString name = type2Title(type);
        int idx = m_tabWidget->addTab(wid, name);
        m_debugPanel = type;
        m_tabWidget->setCurrentIndex(idx);
    }
}

void ZDockWidget::onAddTab(PANEL_TYPE type, DockContentWidgetInfo info)
{
    onAddTab(type);
    QWidget* wid = m_tabWidget->currentWidget();
    if (DockContent_View* view = qobject_cast<DockContent_View*>(wid))
    {
        view->getDisplayWid()->setViewWidgetInfo(info);
        view->setResComboBoxIndex(info.comboboxindex);
    }
}

void ZDockWidget::onMenuActionTriggered(QAction* pAction, bool bTriggered)
{
    if (!pAction)
        return;

    int actionType = pAction->property("ActionType").toInt();
    for (int i = 0; i < m_tabWidget->count(); i++)
    {
        QWidget* wid = m_tabWidget->widget(i);
        if (DockContent_Parameter* prop = qobject_cast<DockContent_Parameter*>(wid))
        {
        }
        if (DockContent_View* pView = qobject_cast<DockContent_View*>(wid))
        {
            pView->onCommandDispatched(pAction, bTriggered);
        }
        if (DockContent_Editor* pEditor = qobject_cast<DockContent_Editor*>(wid))
        {
            //undo/redo issues
            pEditor->onCommandDispatched(pAction, bTriggered);
        }
        if (ZenoSpreadsheet* pSpreadsheet = qobject_cast<ZenoSpreadsheet*>(wid))
        {

        }
        if (ZlogPanel* logPanel = qobject_cast<ZlogPanel*>(wid))
        {

        }
    }
}

void ZDockWidget::init(ZenoMainWindow* pMainWin)
{
    QPalette palette = this->palette();
    palette.setBrush(QPalette::Window, QColor(38, 38, 38));
    palette.setBrush(QPalette::WindowText, QColor());
    setPalette(palette);
    //...
}

bool ZDockWidget::isTopLevelWin()
{
    return false;
}